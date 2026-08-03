"""Tabular finite-K InfoNCE training and optimizer diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .core import logsumexp, normalize_probability

Array = np.ndarray


@dataclass(frozen=True)
class TrainResult:
    """A checkpoint from tabular InfoNCE training.

    ``score`` is the score used for evaluation.  It equals ``last_score`` for
    ordinary training and the Polyak tail average when tail averaging is
    enabled.  Keeping both lets the optimizer-floor ablation measure the gain
    from averaging without running a second stochastic trajectory.
    """

    score: Array
    last_score: Array
    losses: Array
    learning_rates: Array
    steps: int
    positive_pairs_seen: int
    effective_batch_size: int
    averaged_iterates: int
    averaging_start_step: Optional[int]


def sample_categorical_rows(
    rng: np.random.Generator,
    rows_prob: Array,
    rows: Array,
) -> Array:
    cdf = np.cumsum(rows_prob, axis=1)
    cdf[:, -1] = 1.0
    u = rng.random(rows.size)
    return np.sum(u[:, None] > cdf[rows], axis=1)


def sample_infonce_batch(
    rng: np.random.Generator,
    p_x: Array,
    channel: Array,
    *,
    batch_size: int,
    K: int,
) -> Tuple[Array, Array]:
    n = p_x.size
    x_pos = rng.choice(n, size=batch_size, p=p_x)
    y = sample_categorical_rows(rng, channel, x_pos)
    x_neg = rng.choice(n, size=(batch_size, K), p=p_x)
    return np.concatenate([x_pos[:, None], x_neg], axis=1), y


def empirical_infonce_loss(score: Array, candidates: Array, y: Array) -> float:
    logits = score[candidates, y[:, None]]
    return float(np.mean(logsumexp(logits, axis=1) - logits[:, 0]))


def learning_rate_at_step(
    step: int,
    total_steps: int,
    *,
    base_learning_rate: float,
    schedule: str = "constant",
    min_learning_rate: float = 0.0,
    warmup_steps: int = 0,
    milestone_fractions: Sequence[float] = (0.2, 0.5, 0.8),
    decay_gamma: float = 0.25,
) -> float:
    """Return the learning rate for a one-indexed optimizer step.

    Supported schedules are:

    ``constant``
        Keep the base learning rate throughout training.
    ``cosine``
        Cosine decay from the base rate to ``min_learning_rate``.
    ``linear``
        Linear decay from the base rate to ``min_learning_rate``.
    ``multistep``
        Multiply the base rate by ``decay_gamma`` after each progress
        milestone.  Milestones are fractions of the post-warmup trajectory.

    A short linear warmup can be combined with every schedule.
    """
    if total_steps <= 0 or step <= 0 or step > total_steps:
        raise ValueError("step must lie in 1, ..., total_steps")
    if base_learning_rate <= 0:
        raise ValueError("base_learning_rate must be positive")
    if min_learning_rate < 0 or min_learning_rate > base_learning_rate:
        raise ValueError("min_learning_rate must lie in [0, base_learning_rate]")
    if warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError("warmup_steps must lie in [0, total_steps)")

    if warmup_steps > 0 and step <= warmup_steps:
        return base_learning_rate * (step / warmup_steps)

    schedule = schedule.lower().strip()
    post_warmup_steps = max(total_steps - warmup_steps, 1)
    progress = (step - warmup_steps - 1) / max(post_warmup_steps - 1, 1)
    progress = float(np.clip(progress, 0.0, 1.0))

    if schedule == "constant":
        return float(base_learning_rate)
    if schedule == "cosine":
        factor = 0.5 * (1.0 + np.cos(np.pi * progress))
        return float(min_learning_rate + (base_learning_rate - min_learning_rate) * factor)
    if schedule == "linear":
        return float(base_learning_rate + progress * (min_learning_rate - base_learning_rate))
    if schedule == "multistep":
        milestones = tuple(float(x) for x in milestone_fractions)
        if any(x <= 0 or x >= 1 for x in milestones) or any(
            b <= a for a, b in zip(milestones, milestones[1:])
        ):
            raise ValueError("milestone_fractions must be strictly increasing in (0, 1)")
        if not 0 < decay_gamma < 1:
            raise ValueError("decay_gamma must lie in (0, 1)")
        decays = sum(progress >= milestone for milestone in milestones)
        return float(max(min_learning_rate, base_learning_rate * decay_gamma**decays))
    raise ValueError(
        f"unknown learning-rate schedule {schedule!r}; "
        "choose constant, cosine, linear, or multistep"
    )


def _resolve_tail_average_start(
    *,
    max_steps: int,
    tail_average_fraction: Optional[float],
    tail_average_start_step: Optional[int],
) -> Optional[int]:
    if tail_average_fraction is not None and tail_average_start_step is not None:
        raise ValueError(
            "specify at most one of tail_average_fraction and tail_average_start_step"
        )
    if tail_average_fraction is not None:
        if not 0 <= tail_average_fraction < 1:
            raise ValueError("tail_average_fraction must lie in [0, 1)")
        return max(1, int(np.ceil(tail_average_fraction * max_steps)))
    if tail_average_start_step is not None:
        start = int(tail_average_start_step)
        if start <= 0 or start > max_steps:
            raise ValueError("tail_average_start_step must lie in 1, ..., max_steps")
        return start
    return None


def _batch_gradient_and_loss(
    score: Array,
    rng: np.random.Generator,
    p_x: Array,
    channel: Array,
    *,
    microbatch_size: int,
    gradient_accumulation_steps: int,
    K: int,
) -> tuple[Array, float]:
    """Estimate one gradient using one or more independent microbatches."""
    if microbatch_size <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("batch sizes and accumulation counts must be positive")
    grad = np.zeros_like(score)
    loss_sum = 0.0
    total_examples = microbatch_size * gradient_accumulation_steps
    n_y = score.shape[1]

    for _ in range(gradient_accumulation_steps):
        candidates, y = sample_infonce_batch(
            rng,
            p_x,
            channel,
            batch_size=microbatch_size,
            K=K,
        )
        logits = score[candidates, y[:, None]]
        log_norm = logsumexp(logits, axis=1, keepdims=True)
        probs = np.exp(logits - log_norm)
        loss_sum += float(np.sum(np.squeeze(log_norm, axis=1) - logits[:, 0]))

        grad_logits = probs
        grad_logits[:, 0] -= 1.0
        flat_index = candidates * n_y + y[:, None]
        grad += np.bincount(
            flat_index.ravel(),
            weights=grad_logits.ravel(),
            minlength=score.size,
        ).reshape(score.shape)

    grad /= total_examples
    return grad, loss_sum / total_examples


def train_tabular_infonce_checkpoints(
    p_x: Array,
    channel: Array,
    *,
    K: int,
    checkpoint_steps: Iterable[int],
    batch_size: int,
    learning_rate: float,
    seed: int,
    gauge_center: bool = True,
    record_every: int = 10,
    learning_rate_schedule: str = "constant",
    min_learning_rate: float = 0.0,
    warmup_steps: int = 0,
    lr_milestone_fractions: Sequence[float] = (0.2, 0.5, 0.8),
    lr_decay_gamma: float = 0.25,
    tail_average_fraction: Optional[float] = None,
    tail_average_start_step: Optional[int] = None,
    gradient_accumulation_steps: int = 1,
) -> Dict[int, TrainResult]:
    """Train once and return score snapshots at requested optimizer steps.

    Parameters added for the optimizer-floor study:

    ``learning_rate_schedule``
        Controls constant, cosine, linear, or multistep decay.
    ``tail_average_fraction`` / ``tail_average_start_step``
        Enables Polyak-Ruppert averaging of all score iterates after a burn-in.
        At checkpoints before the averaging start, ``score`` is the last
        iterate.  Thereafter it is the running tail average.
    ``gradient_accumulation_steps``
        Forms a larger effective batch while keeping the memory footprint of
        ``batch_size``.  The effective batch size is their product.

    Using one trajectory makes all checkpoints paired within a
    problem/seed/K/optimizer configuration.
    """
    p_x = normalize_probability(p_x, name="p_x")
    W = np.asarray(channel, dtype=np.float64)
    if W.shape[0] != p_x.size or np.any(W <= 0) or not np.allclose(
        W.sum(axis=1), 1.0
    ):
        raise ValueError("channel must have full support and stochastic rows")

    checkpoints = tuple(sorted({int(step) for step in checkpoint_steps}))
    if not checkpoints or checkpoints[0] <= 0:
        raise ValueError("checkpoint_steps must contain positive integers")
    if record_every <= 0:
        raise ValueError("record_every must be positive")

    max_steps = checkpoints[-1]
    averaging_start = _resolve_tail_average_start(
        max_steps=max_steps,
        tail_average_fraction=tail_average_fraction,
        tail_average_start_step=tail_average_start_step,
    )
    effective_batch_size = int(batch_size) * int(gradient_accumulation_steps)

    rng = np.random.default_rng(seed)
    score = 0.01 * rng.standard_normal(W.shape)
    m = np.zeros_like(score)
    v = np.zeros_like(score)
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8
    losses: List[float] = []
    learning_rates: List[float] = []
    results: Dict[int, TrainResult] = {}
    checkpoint_set = set(checkpoints)

    average_sum = np.zeros_like(score)
    average_count = 0

    for t in range(1, max_steps + 1):
        grad, batch_loss = _batch_gradient_and_loss(
            score,
            rng,
            p_x,
            W,
            microbatch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            K=K,
        )
        lr_t = learning_rate_at_step(
            t,
            max_steps,
            base_learning_rate=learning_rate,
            schedule=learning_rate_schedule,
            min_learning_rate=min_learning_rate,
            warmup_steps=warmup_steps,
            milestone_fractions=lr_milestone_fractions,
            decay_gamma=lr_decay_gamma,
        )
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * grad * grad
        score -= lr_t * (m / (1.0 - beta1**t)) / (
            np.sqrt(v / (1.0 - beta2**t)) + adam_eps
        )
        if gauge_center:
            score -= np.sum(p_x[:, None] * score, axis=0, keepdims=True)

        if averaging_start is not None and t >= averaging_start:
            average_sum += score
            average_count += 1

        if t == 1 or t % record_every == 0 or t in checkpoint_set:
            losses.append(float(batch_loss))
            learning_rates.append(float(lr_t))

        if t in checkpoint_set:
            if average_count > 0:
                evaluation_score = average_sum / average_count
            else:
                evaluation_score = score
            results[t] = TrainResult(
                score=evaluation_score.copy(),
                last_score=score.copy(),
                losses=np.asarray(losses, dtype=np.float64).copy(),
                learning_rates=np.asarray(learning_rates, dtype=np.float64).copy(),
                steps=t,
                positive_pairs_seen=t * effective_batch_size,
                effective_batch_size=effective_batch_size,
                averaged_iterates=average_count,
                averaging_start_step=averaging_start,
            )

    return results


def train_tabular_infonce(
    p_x: Array,
    channel: Array,
    *,
    K: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    gauge_center: bool = True,
    record_every: int = 10,
    learning_rate_schedule: str = "constant",
    min_learning_rate: float = 0.0,
    warmup_steps: int = 0,
    lr_milestone_fractions: Sequence[float] = (0.2, 0.5, 0.8),
    lr_decay_gamma: float = 0.25,
    tail_average_fraction: Optional[float] = None,
    tail_average_start_step: Optional[int] = None,
    gradient_accumulation_steps: int = 1,
) -> TrainResult:
    """Single-checkpoint wrapper around :func:`train_tabular_infonce_checkpoints`."""
    return train_tabular_infonce_checkpoints(
        p_x,
        channel,
        K=K,
        checkpoint_steps=(steps,),
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        gauge_center=gauge_center,
        record_every=record_every,
        learning_rate_schedule=learning_rate_schedule,
        min_learning_rate=min_learning_rate,
        warmup_steps=warmup_steps,
        lr_milestone_fractions=lr_milestone_fractions,
        lr_decay_gamma=lr_decay_gamma,
        tail_average_fraction=tail_average_fraction,
        tail_average_start_step=tail_average_start_step,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )[int(steps)]


def critic_mse_modulo_y_gauge(
    score: Array,
    target: Array,
    p_x: Array,
    p_y: Array,
) -> float:
    p_x = normalize_probability(p_x, name="p_x")
    p_y = normalize_probability(p_y, name="p_y")
    residual = np.asarray(score) - np.asarray(target)
    residual -= np.sum(p_x[:, None] * residual, axis=0, keepdims=True)
    return float(np.sum(p_x[:, None] * p_y[None, :] * residual * residual))
