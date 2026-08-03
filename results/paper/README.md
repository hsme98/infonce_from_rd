# Bundled reported results

This directory contains compact numerical summaries from the runs reported in
the accompanying paper and README. It intentionally excludes CIFAR model
checkpoints and feature tensors because those artifacts are large.

- `generic_channel/`: ten-seed multistep optimizer records and theorem
diagnostics.
- `markov/`: ten-seed aggregate metrics and the representative seed-3 arrays.
- `cifar/`: aggregate curves, thresholds, and feature diagnostics over three
training seeds.
- `optimizer_ablation/`: fixed-pair and fixed-step optimizer study summaries.

Run `python scripts/rebuild_bundled_figures.py` to regenerate the figures in
`assets/` from these files.
