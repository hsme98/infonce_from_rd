# Experiment definitions

## Generic positive-pair channels

A random full-support joint coupling is generated either with matched marginals
or with distinct prescribed input and output marginals. Finite-`K` tabular
InfoNCE is trained for `K in {1, 4, 32}`. The exponentiated learned score is
Sinkhorn-scaled to the prescribed marginals, yielding the recovered
fixed-output RD coupling.

The experiment reports:

- critic MSE after minimizing over the output-only InfoNCE gauge;
- weighted row KL from the true positive-pair channel;
- the exact fixed-output RD objective-gap identity;
- invariance under `d(x,y) -> d(x,y) + lambda g(y)`;
- the corresponding failure of invariance for free-output RD.

The paper configuration uses ten seeds and nine checkpoints along one optimizer
trajectory per `(marginal mode, seed, K)`. Multistep learning-rate decay is used
because constant-step Adam reaches a stochastic last-iterate floor.

## Hierarchical Markov coarse-graining

Each seed creates a 64-state stationary reversible chain with four
macro-communities, four micro-communities per macro-community, and four states
per micro-community. Positive pairs are adjacent states; negatives follow the
stationary marginal.

After InfoNCE training, the recovered channel induces a canonical distortion.
Marginal-preserving RD is solved over a multiplier grid. Information retained
at each rate is decomposed into:

- macro-community information;
- micro-community information beyond the macro label;
- state identity information beyond the micro label.

Oracle information-density, Hamming, and random-spherical distortions are
reported for comparison.

## CIFAR-100 semantic rate-distortion

A CIFAR-adapted ResNet-18 is trained with SimCLR. The projection head output
`z` is normalized and used in the NT-Xent objective. A separate supervised
ResNet-18 is trained only as a reference ceiling. On a balanced test subset,
the following distances are evaluated:

- `simclr_z`: normalized InfoNCE critic space;
- `simclr_h`: normalized pre-projection encoder space;
- `pixel`: raw-pixel MSE;
- `random_z`: untrained architecture-matched projection space;
- `supervised_h`: supervised encoder reference.

For each distance and RD multiplier, log-domain Sinkhorn solves the empirical
self-coupling problem with uniform marginals. The resulting coupling is scored
by its mutual-information rate and by normalized mutual information between
CIFAR-100 superclass and fine-class labels. Labels are used only in this final
evaluation.

## Optimizer-floor ablation

Two comparison conventions are run:

1. fixed number of positive pairs;
2. fixed number of optimizer updates.

The interventions are constant learning rate, cosine decay, multistep decay,
Polyak tail averaging, effective batch 2048 through gradient accumulation, and
selected combinations. This isolates stochastic last-iterate variance from
minibatch variance and update-count effects.
