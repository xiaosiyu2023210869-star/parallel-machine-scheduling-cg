#!/usr/bin/env python3
"""Append a WSPT-based heuristic baseline to saved CG comparison results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pmcg.wspt_baseline import append_wspt_to_saved_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--saved-comparison",
        type=Path,
        required=True,
        help="Saved comparison workbook/CSV containing the existing CG rows.",
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-xlsx", type=Path, default=ROOT / "outputs" / "wspt-baseline" / "comparison_with_wspt.xlsx")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "outputs" / "wspt-baseline" / "comparison_with_wspt.csv")
    parser.add_argument(
        "--machines",
        type=int,
        help="Override the machine count. Omit this to use the saved row value or the experiment default.",
    )
    args = parser.parse_args()

    wspt_rows = append_wspt_to_saved_results(
        args.saved_comparison,
        args.data_dir,
        args.output_xlsx,
        args.output_csv,
        machines=args.machines,
    )
    print(f"Added {len(wspt_rows)} WSPT-BR rows.")
    print(f"Wrote {args.output_xlsx}")
    if args.output_csv is not None:
        print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
