#!/usr/bin/env python3
"""Run fixed-K CG under a sweep of requested machine counts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pmcg.parameters import effective_machine_count, machine_count_adjusted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=("small", "medium", "large"), default="small")
    parser.add_argument("--instances", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--time-limit", type=float)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--machines", type=int, nargs="+", default=[3, 6, 9])
    parser.add_argument("--instance-ids", type=int, nargs="+")
    parser.add_argument("--tool-change-interval", type=int)
    parser.add_argument("--tool-change-time", type=float)
    return parser.parse_args()


def defaults_for_scale(scale: str) -> tuple[float, int, float]:
    if scale == "small":
        return 1800.0, 4, 2.0
    return 1800.0, 2, 4.0


def read_finished(
    path: Path,
    instances_by_id: dict[int, dict[str, object]],
) -> dict[tuple[int, int], dict[str, object]]:
    if not path.exists():
        return {}
    finished: dict[tuple[int, int], dict[str, object]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                inst_id = int(float(row["instance_id"]))
                requested_m = int(float(row.get("requested_m") or row["m"]))
                n = int(instances_by_id[inst_id]["n"])
                actual_m = int(float(row.get("m") or effective_machine_count(requested_m, n)))
                objective = float(row["objective"])
                status = str(row.get("status") or row.get("cg_status") or "")
            except (KeyError, TypeError, ValueError):
                continue
            if (
                actual_m == effective_machine_count(requested_m, n)
                and math.isfinite(objective)
                and status
            ):
                finished[(inst_id, requested_m)] = row
    return finished


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "instance_id",
        "n",
        "requested_m",
        "m",
        "machine_count_adjusted",
        "K",
        "time_limit",
        "objective",
        "incumbent_objective",
        "time_sec",
        "status",
        "iterations",
        "num_columns",
        "initial_columns",
        "initial_objective",
        "last_rmp_obj",
        "pricing_status",
        "integer_status",
        "objective_source",
        "checkpoint_saved_at",
    ]
    extras = sorted({key for row in rows for key in row if key not in fieldnames})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + extras)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    default_time, default_c, default_theta = defaults_for_scale(args.scale)
    instances_path = args.instances or ROOT / "data" / f"instances_{args.scale}.json"
    output_csv = args.output_csv or ROOT / "outputs" / "m-sensitivity" / f"cg_m_sensitivity_{args.scale}.csv"
    c = args.tool_change_interval or default_c
    theta = args.tool_change_time if args.tool_change_time is not None else default_theta

    os.environ["PMCG_RUNTIME_DIR"] = str(ROOT)
    os.environ.setdefault("PMCG_OUTPUT_DIR", str(output_csv.parent))

    all_instances = json.loads(instances_path.read_text(encoding="utf-8"))
    instances_by_id = {int(inst["id"]): inst for inst in all_instances}
    instances = all_instances
    if args.instance_ids:
        selected_ids = set(args.instance_ids)
        missing = selected_ids.difference(instances_by_id)
        if missing:
            raise ValueError(f"Unknown instance ids: {sorted(missing)}")
        instances = [inst for inst in all_instances if int(inst["id"]) in selected_ids]
    finished = read_finished(output_csv, instances_by_id)
    rows = list(finished.values())
    completed = set(finished)

    for inst in instances:
        inst_id = int(inst["id"])
        p = [float(value) for value in inst["p"]]
        w = [float(value) for value in inst["w"]]
        n = int(inst["n"])
        time_limit = (
            args.time_limit
            if args.time_limit is not None
            else float(inst.get("time_limit", default_time))
        )
        for requested_m in args.machines:
            key = (inst_id, int(requested_m))
            if key in completed:
                print(f"instance={inst_id}, requested_m={requested_m} already finished; skip.")
                continue
            m_eff = effective_machine_count(requested_m, n)
            adjusted = machine_count_adjusted(requested_m, m_eff)
            suffix = f" (requested m={requested_m} adjusted to m={m_eff})" if adjusted else ""
            print(f"Running instance={inst_id}, n={n}, m={m_eff}, K={args.k}{suffix}", flush=True)
            from pmcg.ml_assisted import run_fixed_k_column_generation

            result = run_fixed_k_column_generation(
                p,
                w,
                requested_m,
                args.k,
                time_limit,
                tool_change_interval=c,
                tool_change_time=theta,
            )
            row = {
                "instance_id": inst_id,
                "n": n,
                "requested_m": int(requested_m),
                "m": m_eff,
                "machine_count_adjusted": adjusted,
                "K": args.k,
                "time_limit": time_limit,
                **result,
                "checkpoint_saved_at": datetime.now().isoformat(timespec="seconds"),
            }
            rows = [old for old in rows if not (int(float(old["instance_id"])) == inst_id and int(float(old.get("requested_m") or old["m"])) == int(requested_m))]
            rows.append(row)
            completed.add(key)
            write_rows(output_csv, rows)

    write_rows(output_csv, rows)
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
