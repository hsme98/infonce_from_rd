# Result interpretation

## What the experiments establish

The generic-channel experiment directly tests the theorem-facing claim. It
starts from a coupling that was not generated from a known distortion, learns
an InfoNCE critic, and recovers the original channel by solving the induced
fixed-output RD problem. The gauge stress test distinguishes fixed-output RD
from free-output RD: only the former quotients out the same output-only
ambiguity as InfoNCE.

The Markov experiment supplies a controlled operational consequence. At low
information rates, the InfoNCE-induced distortion preferentially preserves the
macro-state; micro-state and individual identity require successively larger
rates. The learned hierarchy nearly matches the oracle information-density
distortion.

The CIFAR experiment supplies real-data evidence. The normalized critic space
retains both superclass and fine-class information at lower rates than pixel
MSE or an untrained network over a broad part of the RD curve. The pre-projection
representation is slightly stronger, which is reported explicitly. CIFAR does
not show a sharp coarse-before-fine phase transition, so the claim is semantic
rate efficiency rather than universal hierarchical ordering.

## What the experiments do not establish

- Larger `K` has the same population target but not the same finite-training
  efficiency or compute cost.
- The CIFAR experiment does not reconstruct the unknown augmentation channel;
  exact cross-optimality is tested by the generic-channel experiment.
- The supervised representation is not a fair self-supervised baseline.
- Characteristic or injective kernels provide identifiability statements, but
  intentional representation invariances can make raw-space injectivity
  undesirable.
