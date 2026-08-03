# Sequential reproducibility

All orchestration is local and sequential. No scheduler, job array, distributed
training, or multi-GPU setup is required.

## Resume behavior

CIFAR training writes `checkpoint_latest.pt` periodically and resumes from it.
A validated `checkpoint_final.pt` is skipped. Per-seed RD outputs are also
skipped when their configuration fingerprint matches. Changing a configuration
while reusing the same output directory raises an error rather than silently
mixing results.

## Recommended workflow

1. Run `pytest -q` and the synthetic smoke test.
2. Run the `pilot` profile.
3. Inspect the generated curves and feature diagnostics.
4. Run the `paper` profile on one CUDA GPU.
5. Validate outputs with `scripts/validate_results.py`.
6. Archive the exact configuration files with the output directory.

## Determinism

`--deterministic` asks PyTorch for deterministic behavior where supported.
Exact bitwise reproducibility can still depend on the PyTorch, CUDA, cuDNN, and
GPU versions. Numerical theorem checks in the tabular experiments use NumPy
`float64` and are stable to machine precision.
