#!/usr/bin/env python3
"""Compare fixed-K and ML-selected-K column generation."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=Path, default=ROOT / "data" / "instances_ml.json")
    parser.add_argument("--training-data", type=Path, default=ROOT / "data" / "training_data.xlsx")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "ml-assisted")
    parser.add_argument("--time-limit", type=float, default=1800.0)
    parser.add_argument("--baseline-k", type=int, default=30)
    parser.add_argument("--machines", type=int, default=3)
    parser.add_argument("--disable-online-learning", action="store_true")
    args = parser.parse_args()
    if args.machines != 3 and args.output_dir == ROOT / "outputs" / "ml-assisted":
        args.output_dir = ROOT / "outputs" / f"ml-assisted-m{args.machines}"
    os.environ["PMCG_INSTANCES_FILE"] = str(args.instances)
    os.environ["PMCG_TRAINING_DATA"] = str(args.training_data)
    os.environ["PMCG_OUTPUT_DIR"] = str(args.output_dir)
    from pmcg import ml_assisted
    ml_assisted.TIME_LIMIT = args.time_limit
    ml_assisted.BASELINE_K = args.baseline_k
    ml_assisted.m = args.machines
    ml_assisted.ONLINE_LEARNING_ENABLED = not args.disable_online_learning
    ml_assisted.main()


if __name__ == "__main__":
    main()
