# Validation performed for this public archive

The cleaned sequential repository was checked with:

```text
pytest -q
19 passed
```

Additional execution checks:

- the quick generic-channel and Markov profiles completed sequentially;
- `scripts/validate_results.py` accepted the quick tabular outputs;
- the offline synthetic CIFAR RD smoke test completed and produced curves,
  thresholds, diagnostics, and a PDF figure;
- the quick fixed-pair and fixed-step optimizer ablations completed;
- all bundled README figures were regenerated from the included numerical
  summaries.

The full CIFAR neural training was not rerun while packaging this archive. The
bundled CIFAR aggregate files are the completed three-seed results supplied for
the paper; the public sequential pipeline is the scheduler-free rewrite of the
code used to produce them.
