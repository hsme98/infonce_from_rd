#!/usr/bin/env python3
"""Run the generic-channel inverse fixed-output RD experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from itw_rd.config import load_json
from itw_rd.experiment1 import Experiment1Config, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper/generic_channel.json")
    parser.add_argument("--outdir")
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()

    config = Experiment1Config.from_dict(load_json(args.config))
    if args.outdir:
        config.outdir = args.outdir
    if args.seeds:
        config.seeds = tuple(args.seeds)
    result = run(config)
    print(f"Generic-channel experiment complete: {Path(config.outdir).resolve()}")
    print(f"Recorded {len(result['records'])} training checkpoints.")


if __name__ == "__main__":
    main()
