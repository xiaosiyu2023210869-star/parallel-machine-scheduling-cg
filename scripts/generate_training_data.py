#!/usr/bin/env python3
"""Generate K-sweep training records for the ML selector."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=Path, default=ROOT / "data" / "instances_training.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "training")
    parser.add_argument("--time-limit", type=float, default=1800.0)
    args = parser.parse_args()
    os.environ["PMCG_INSTANCES_FILE"] = str(args.instances)
    os.environ["PMCG_OUTPUT_DIR"] = str(args.output_dir)
    from pmcg import training
    training.TIME_LIMIT = args.time_limit
    training.main()


if __name__ == "__main__":
    main()
