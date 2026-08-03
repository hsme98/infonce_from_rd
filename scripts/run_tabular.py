#!/usr/bin/env python3
"""Run the generic-channel and Markov experiments sequentially."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from itw_rd.config import load_json
from itw_rd.experiment1 import Experiment1Config, run as run_generic
from itw_rd.experiment2 import Experiment2Config, run as run_markov


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["quick", "pilot", "paper"], default="paper")
    parser.add_argument("--output-root", default="results/reproduced/tabular")
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()

    profile = REPO_ROOT / "configs" / args.profile
    output_root = Path(args.output_root)

    generic = Experiment1Config.from_dict(load_json(profile / "generic_channel.json"))
    markov = Experiment2Config.from_dict(load_json(profile / "markov.json"))
    generic.outdir = str(output_root / "generic_channel")
    markov.outdir = str(output_root / "markov")
    if args.seeds:
        generic.seeds = tuple(args.seeds)
        markov.seeds = tuple(args.seeds)
        if markov.representative_seed not in markov.seeds:
            markov.representative_seed = markov.seeds[0]

    print("[1/2] Generic positive-pair channels", flush=True)
    run_generic(generic)
    print("[2/2] Hierarchical Markov coarse-graining", flush=True)
    run_markov(markov)
    print(f"Tabular experiments complete: {output_root.resolve()}")


if __name__ == "__main__":
    main()
