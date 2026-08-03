import numpy as np

from itw_rd.generators import make_generic_channel
from itw_rd.infonce import (
    learning_rate_at_step,
    train_tabular_infonce_checkpoints,
)
from itw_rd.optimization_ablation import (
    OptimizationAblationConfig,
    OptimizerVariant,
    _checkpoint_plan,
)


def test_learning_rate_schedules_have_expected_endpoints():
    base = 0.04
    minimum = 0.0005
    assert learning_rate_at_step(
        1,
        100,
        base_learning_rate=base,
        schedule="constant",
    ) == base
    assert np.isclose(
        learning_rate_at_step(
            1,
            100,
            base_learning_rate=base,
            schedule="cosine",
            min_learning_rate=minimum,
        ),
        base,
    )
    assert np.isclose(
        learning_rate_at_step(
            100,
            100,
            base_learning_rate=base,
            schedule="cosine",
            min_learning_rate=minimum,
        ),
        minimum,
    )
    final_multistep = learning_rate_at_step(
        100,
        100,
        base_learning_rate=base,
        schedule="multistep",
        min_learning_rate=minimum,
        milestone_fractions=(0.2, 0.5, 0.8),
        decay_gamma=0.25,
    )
    assert np.isclose(final_multistep, max(minimum, base * 0.25**3))


def test_tail_average_and_last_iterate_are_both_returned():
    problem = make_generic_channel(n=7, seed=2, matched=False)
    result = train_tabular_infonce_checkpoints(
        problem.p_x,
        problem.channel,
        K=4,
        checkpoint_steps=(20,),
        batch_size=32,
        learning_rate=0.03,
        learning_rate_schedule="constant",
        tail_average_fraction=0.5,
        seed=19,
    )[20]
    assert result.averaging_start_step == 10
    assert result.averaged_iterates == 11
    assert result.score.shape == result.last_score.shape == (7, 7)
    assert not np.allclose(result.score, result.last_score)


def test_gradient_accumulation_sets_effective_batch_and_pair_count():
    problem = make_generic_channel(n=6, seed=5, matched=True)
    result = train_tabular_infonce_checkpoints(
        problem.p_x,
        problem.channel,
        K=2,
        checkpoint_steps=(12,),
        batch_size=16,
        gradient_accumulation_steps=4,
        learning_rate=0.02,
        seed=7,
    )[12]
    assert result.effective_batch_size == 64
    assert result.positive_pairs_seen == 12 * 64


def test_pair_budget_plan_matches_effective_batch():
    variant = OptimizerVariant(
        name="large",
        batch_size=512,
        gradient_accumulation_steps=4,
    )
    config = OptimizationAblationConfig(
        matched_modes=("matched",),
        k_values=(4,),
        seeds=(0,),
        checkpoint_unit="positive_pairs",
        checkpoints=(25_600, 51_200),
        variants=(variant,),
    )
    steps, mapping = _checkpoint_plan(config, variant)
    assert steps == (13, 25)
    assert mapping[25_600] == 13
    assert mapping[51_200] == 25
