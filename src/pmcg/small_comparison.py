#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined comparison script.

This script keeps the fixed n/p/w instances and the recorded results for the
recorded first-three-method results from the source experiment, then runs the
same fixed-K column-generation implementation as the ML workflow.

Important parameter choices are kept from compare.html so the CG run is
comparable with the recorded MIP results:
    m = 3, c = 4, theta = 2.0, TIME_LIMIT = 600
"""

import csv
import gc
import json
import math
import os
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME_DIR = Path(os.environ.get("PMCG_RUNTIME_DIR", PROJECT_ROOT))
_SCRIPT_DIR = _RUNTIME_DIR
(_SCRIPT_DIR / ".cache").mkdir(parents=True, exist_ok=True)
(_SCRIPT_DIR / ".matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_SCRIPT_DIR / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(_SCRIPT_DIR / ".matplotlib"))
os.environ["LC_ALL"] = "C"
os.environ["LANG"] = "C"
os.environ["LC_CTYPE"] = "C"

# ================== Fixed parameters from compare.html ==================
m = 3
c = 4
theta = 2.0
TIME_LIMIT = 600


# ================== Shared fixed-K column generation parameter ==================
K_FIXED = 30

OUTPUT_DIR = Path(os.environ.get("PMCG_OUTPUT_DIR", PROJECT_ROOT / "outputs" / "comparison-small"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = OUTPUT_DIR / "comparison_results.csv"
OUTPUT_PNG = OUTPUT_DIR / "comparison_results.png"
CHECKPOINT_TIME_FIELD = "checkpoint_saved_at"
METHODS = ("FullMIP", "Case11MIP", "Case12MIP", "CG")


# ================== Fixed instances and recorded first-three-method results ==================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INSTANCES_FILE = PROJECT_ROOT / "data" / "instances_small.json"
INSTANCES_FILE = Path(os.environ.get("PMCG_INSTANCES_FILE", DEFAULT_INSTANCES_FILE))
INSTANCES = json.loads(INSTANCES_FILE.read_text(encoding="utf-8"))


# ================== Result helpers ==================
def format_obj(value):
    if value is None or math.isinf(value):
        return "inf"
    return f"{value:.1f}"


def is_finished_result(result):
    if not result:
        return False
    objective = result.get("objective")
    if objective is None:
        return False
    try:
        if math.isinf(float(objective)):
            return False
    except (TypeError, ValueError):
        return False
    status = str(result.get("status") or "").strip().lower()
    return status not in {"", "none", "nan", "not_started"}


def recorded_result(item):
    time_sec = float(item["time_sec"])
    return {
        "objective": float(item["objective"]),
        "time_sec": time_sec,
        "status": "RECORDED_FROM_COMPARE_HTML",
        "timed_out": time_sec >= TIME_LIMIT - 1e-3,
        "bound": None,
        "gap": None,
        "sol_count": None,
    }


def initial_results():
    results = {}
    for inst in INSTANCES:
        inst_id = inst["id"]
        results[inst_id] = {
            "n": inst["n"],
            "p": inst["p"],
            "w": inst["w"],
            "methods": {
                method: recorded_result(values)
                for method, values in inst["recorded"].items()
            },
        }
    return results


def parse_csv_value(value):
    if value is None or value == "":
        return None
    if isinstance(value, str) and value.strip().lower() in {"none", "nan"}:
        return None
    return value


def parse_bool(value):
    value = parse_csv_value(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def parse_int(value):
    value = parse_csv_value(value)
    if value is None:
        return None
    return int(float(value))


def parse_float(value):
    value = parse_csv_value(value)
    if value is None:
        return None
    return float(value)


# ================== Canonical fixed-K column generation ==================
def column_generation_with_time_limit(p, w, m_val, k_fixed, time_limit):
    from .ml_assisted import run_fixed_k_column_generation

    return run_fixed_k_column_generation(
        p,
        w,
        m_val,
        k_fixed,
        time_limit,
        tool_change_interval=c,
        tool_change_time=theta,
    )


# ================== Saving and plotting ==================
def mark_checkpoint(result):
    result[CHECKPOINT_TIME_FIELD] = datetime.now().isoformat(timespec="seconds")
    return result


def load_checkpoint(filename=OUTPUT_CSV):
    results = initial_results()
    if not Path(filename).exists():
        print(f"No checkpoint found at {filename}. Starting with recorded MIP results.")
        return results

    loaded = 0
    skipped = 0
    with open(filename, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                inst_id = parse_int(row.get("instance_id"))
                method = parse_csv_value(row.get("method"))
            except (TypeError, ValueError):
                skipped += 1
                continue
            if inst_id not in results or method != "CG":
                continue

            result = {
                "objective": parse_float(row.get("objective")),
                "time_sec": parse_float(row.get("time_sec")),
                "status": parse_csv_value(row.get("status")),
                "timed_out": parse_bool(row.get("timed_out")),
                "bound": parse_float(row.get("bound")),
                "gap": parse_float(row.get("gap")),
                "sol_count": parse_int(row.get("sol_count")),
                "K": parse_int(row.get("K")),
                "iterations": parse_int(row.get("iterations")),
                "num_columns": parse_int(row.get("num_columns")),
                "initial_columns": parse_int(row.get("initial_columns")),
                "initial_schedules": parse_int(row.get("initial_schedules")),
                "pricing_status": parse_csv_value(row.get("pricing_status")),
                "pricing_sol_count": parse_int(row.get("pricing_sol_count")),
                "integer_status": parse_csv_value(row.get("integer_status")),
                "incumbent_status": parse_csv_value(row.get("incumbent_status")),
                "objective_source": parse_csv_value(row.get("objective_source")),
                "last_integer_update_iter": parse_int(row.get("last_integer_update_iter")),
                "initial_objective": parse_float(row.get("initial_objective")),
                "last_rmp_obj": parse_float(row.get("last_rmp_obj")),
                "best_reduced_cost": parse_float(row.get("best_reduced_cost")),
                "final_int_bound": parse_float(row.get("final_int_bound")),
                "final_int_gap": parse_float(row.get("final_int_gap")),
                "final_int_sol_count": parse_int(row.get("final_int_sol_count")),
                CHECKPOINT_TIME_FIELD: parse_csv_value(row.get(CHECKPOINT_TIME_FIELD)),
            }
            if not is_finished_result(result):
                skipped += 1
                continue
            results[inst_id]["methods"]["CG"] = result
            loaded += 1

    print(f"Loaded {loaded} finished CG records from {filename}; ignored {skipped} incomplete rows.")
    return results


def flatten_records(results):
    rows = []
    for inst in INSTANCES:
        inst_id = inst["id"]
        if inst_id not in results:
            continue
        n = inst["n"]
        for method in METHODS:
            result = results[inst_id]["methods"].get(method)
            if not result:
                continue
            row = {
                "instance_id": inst_id,
                "n": n,
                "method": method,
                "objective": result.get("objective"),
                "time_sec": result.get("time_sec"),
                "status": result.get("status"),
                "timed_out": result.get("timed_out"),
                "bound": result.get("bound"),
                "gap": result.get("gap"),
                "sol_count": result.get("sol_count"),
                "K": result.get("K"),
                "iterations": result.get("iterations"),
                "num_columns": result.get("num_columns"),
                "initial_columns": result.get("initial_columns"),
                "initial_schedules": result.get("initial_schedules"),
                "pricing_status": result.get("pricing_status"),
                "pricing_sol_count": result.get("pricing_sol_count"),
                "integer_status": result.get("integer_status"),
                "incumbent_status": result.get("incumbent_status"),
                "objective_source": result.get("objective_source"),
                "last_integer_update_iter": result.get("last_integer_update_iter"),
                "initial_objective": result.get("initial_objective"),
                "last_rmp_obj": result.get("last_rmp_obj"),
                "best_reduced_cost": result.get("best_reduced_cost"),
                "final_int_bound": result.get("final_int_bound"),
                "final_int_gap": result.get("final_int_gap"),
                "final_int_sol_count": result.get("final_int_sol_count"),
                CHECKPOINT_TIME_FIELD: result.get(CHECKPOINT_TIME_FIELD),
                "p": inst["p"],
                "w": inst["w"],
            }
            rows.append(row)
    return rows


def save_csv(results, filename=OUTPUT_CSV):
    rows = flatten_records(results)
    fieldnames = [
        "instance_id",
        "n",
        "method",
        "objective",
        "time_sec",
        "status",
        "timed_out",
        "bound",
        "gap",
        "sol_count",
        "K",
        "iterations",
        "num_columns",
        "initial_columns",
        "initial_schedules",
        "pricing_status",
        "pricing_sol_count",
        "integer_status",
        "incumbent_status",
        "objective_source",
        "last_integer_update_iter",
        "initial_objective",
        "last_rmp_obj",
        "best_reduced_cost",
        "final_int_bound",
        "final_int_gap",
        "final_int_sol_count",
        CHECKPOINT_TIME_FIELD,
        "p",
        "w",
    ]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(results, filename=OUTPUT_PNG):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        print(f"Skipping plot because matplotlib/numpy is unavailable: {exc}")
        return

    ns = [inst["n"] for inst in INSTANCES]
    labels = ["Full MIP", "Case1.1 MIP", "Case1.2 MIP", "Column Generation"]
    markers = ["o", "s", "^", "D"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for method, label, marker in zip(METHODS, labels, markers):
        vals = []
        for inst in INSTANCES:
            obj = results.get(inst["id"], {}).get("methods", {}).get(method, {}).get("objective")
            vals.append(np.nan if obj is None or math.isinf(obj) else obj)
        ax.plot(ns, vals, marker=marker, label=label)
    ax.set_xlabel("Number of jobs")
    ax.set_ylabel("Total Weighted Completion Time")
    ax.set_title("Objective Value Comparison")
    ax.legend()
    ax.grid(True)

    ax = axes[1]
    for method, label, marker in zip(METHODS, labels, markers):
        times = []
        for inst in INSTANCES:
            value = results.get(inst["id"], {}).get("methods", {}).get(method, {}).get("time_sec")
            times.append(np.nan if value is None else value)
        ax.plot(ns, times, marker=marker, label=label)
    ax.set_xlabel("Number of jobs")
    ax.set_ylabel("Solving Time (seconds)")
    ax.set_title("Computation Time Comparison")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close(fig)


def main():
    results = load_checkpoint()
    save_csv(results)

    for inst in INSTANCES:
        inst_id = inst["id"]
        n = inst["n"]
        p = inst["p"]
        w = inst["w"]

        print(f"\n{'=' * 20} n={n} {'=' * 20}")
        print(f"p = {p}")
        print(f"w = {w}")

        for method in ("FullMIP", "Case11MIP", "Case12MIP"):
            r = results[inst_id]["methods"][method]
            print(
                f"{method} recorded from compare.html. "
                f"obj={format_obj(r['objective'])}, time={r['time_sec']:.2f}s"
            )

        if is_finished_result(results[inst_id]["methods"].get("CG")):
            r = results[inst_id]["methods"]["CG"]
            print(
                f"Column Generation already finished, skip. obj={format_obj(r['objective'])}, "
                f"status={r['status']}, iters={r.get('iterations')}, cols={r.get('num_columns')}"
            )
        else:
            print(f"Solving Column Generation with improved code, fixed K={K_FIXED} ...", flush=True)
            results[inst_id]["methods"]["CG"] = mark_checkpoint(
                column_generation_with_time_limit(p, w, m, K_FIXED, TIME_LIMIT)
            )
            r = results[inst_id]["methods"]["CG"]
            print(
                f"  obj={format_obj(r['objective'])}, time={r['time_sec']:.2f}s, "
                f"status={r['status']}, iters={r['iterations']}, cols={r['num_columns']}, "
                f"source={r['objective_source']}"
            )
            save_csv(results)
            gc.collect()

    save_csv(results)
    plot_results(results)
    print(f"\nResults saved to {OUTPUT_CSV}")
    print(f"Plot saved to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
