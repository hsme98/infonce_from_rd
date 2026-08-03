#!/usr/bin/env python3
"""Run the hierarchical Markov coarse-graining experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from itw_rd.config import load_json
from itw_rd.experiment2 import Experiment2Config, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper/markov.json")
    parser.add_argument("--outdir")
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()

    config = Experiment2Config.from_dict(load_json(args.config))
    if args.outdir:
        config.outdir = args.outdir
    if args.seeds:
        config.seeds = tuple(args.seeds)
        if config.representative_seed not in config.seeds:
            config.representative_seed = config.seeds[0]
    result = run(config)
    print(f"Markov experiment complete: {Path(config.outdir).resolve()}")
    print(f"Aggregated {len(result['seeds'])} independent chains.")


if __name__ == "__main__":
    main()
