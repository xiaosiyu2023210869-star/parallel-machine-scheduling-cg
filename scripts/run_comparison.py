#!/usr/bin/env python3
"""Run the four-method benchmark for a selected instance scale."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=("small", "medium", "large"), default="small")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--time-limit", type=float)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--machines", type=int, default=3)
    parser.add_argument("--tool-change-interval", type=int)
    parser.add_argument("--tool-change-time", type=float)
    parser.add_argument("--show-plots", action="store_true")
    args = parser.parse_args()

    defaults = {
        "small": (600.0, 4, 2.0),
        "medium": (1800.0, 2, 4.0),
        "large": (1800.0, 2, 4.0),
    }
    default_time, default_c, default_theta = defaults[args.scale]
    instances_file = ROOT / "data" / f"instances_{args.scale}.json"
    output_dir = args.output_dir or ROOT / "outputs" / f"comparison-{args.scale}"
    os.environ["PMCG_INSTANCES_FILE"] = str(instances_file)
    os.environ["PMCG_OUTPUT_DIR"] = str(output_dir)
    os.environ["PMCG_RUNTIME_DIR"] = str(ROOT)
    if args.show_plots:
        os.environ["PMCG_SHOW_PLOTS"] = "1"

    if args.scale == "small":
        from pmcg import small_comparison as engine
    else:
        from pmcg import comparison as engine

    engine.TIME_LIMIT = args.time_limit or default_time
    engine.K_FIXED = args.k
    engine.m = args.machines
    engine.c = args.tool_change_interval or default_c
    engine.theta = args.tool_change_time if args.tool_change_time is not None else default_theta
    engine.main()


if __name__ == "__main__":
    main()
