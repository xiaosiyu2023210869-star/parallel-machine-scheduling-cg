# -*- coding: utf-8 -*-
"""
Four-method comparison on the fixed instances used by fixed_column_generation.py.

Method 1: Full MIP
Method 2: Case 1.1 MIP with nonincreasing weight order
Method 3: Case 1.2 MIP with nondecreasing processing-time order
Method 4: The canonical fixed-K column generation used by the ML workflow.

Both the direct MIP methods and the column-generation method use a 1800 second
wall-clock limit per instance/method. Column generation keeps an incumbent
integer-master objective during the run and returns the current incumbent if the
time budget is exhausted.
"""

import csv
import gc
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME_DIR = Path(os.environ.get("PMCG_RUNTIME_DIR", PROJECT_ROOT))
_CACHE_DIR = _RUNTIME_DIR / ".cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
(_CACHE_DIR / ".cache").mkdir(parents=True, exist_ok=True)
(_CACHE_DIR / ".matplotlib").mkdir(parents=True, exist_ok=True)
os.environ["LC_ALL"] = "C"
os.environ["LANG"] = "C"
os.environ["LC_CTYPE"] = "C"
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / ".matplotlib"))

import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import numpy as np

from .parameters import effective_machine_count, machine_count_adjusted


# ================== Global parameters ==================
m = 3
c = 2
theta = 4.0

TIME_LIMIT = 1800
K_FIXED = 30

OUTPUT_DIR = Path(os.environ.get("PMCG_OUTPUT_DIR", PROJECT_ROOT / "outputs" / "comparison"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = OUTPUT_DIR / "comparison_results.csv"
OUTPUT_PNG = OUTPUT_DIR / "comparison_results.png"
CHECKPOINT_TIME_FIELD = "checkpoint_saved_at"
METHODS = ("FullMIP", "Case11MIP", "Case12MIP", "CG")


# ================== Fixed instances ==================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INSTANCES_FILE = PROJECT_ROOT / "data" / "instances_medium.json"
INSTANCES_FILE = Path(os.environ.get("PMCG_INSTANCES_FILE", DEFAULT_INSTANCES_FILE))
instances = json.loads(INSTANCES_FILE.read_text(encoding="utf-8"))


# ================== Shared helpers ==================
def status_name(status):
    return {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
    }.get(status, str(status))


def safe_model_attr(model, attr):
    try:
        return getattr(model, attr)
    except (AttributeError, gp.GurobiError):
        return None


def mip_result(model, elapsed):
    obj = float(model.ObjVal) if model.SolCount > 0 else float("inf")
    bound = safe_model_attr(model, "ObjBound")
    gap = safe_model_attr(model, "MIPGap")
    return {
        "objective": obj,
        "time_sec": elapsed,
        "status": status_name(model.Status),
        "sol_count": int(model.SolCount),
        "bound": None if bound is None else float(bound),
        "gap": None if gap is None else float(gap),
        "timed_out": model.Status == GRB.TIME_LIMIT,
    }


def compute_column(seq, p, w):
    times = []
    cur = 0.0
    for pos, job in enumerate(seq):
        if pos > 0 and pos % c == 0:
            cur += theta
        cur += p[job]
        times.append(cur)
    cost = sum(w[job] * times[i] for i, job in enumerate(seq))
    return times, cost


def configure_mip(model):
    model.setParam("MIPFocus", 1)
    model.setParam("Heuristics", 0.5)
    model.setParam("Presolve", 2)
    model.setParam("Cuts", 2)


def greedy_ordered_machine_sequences(p, w, m_val, order_key):
    sorted_jobs = sorted(range(len(p)), key=order_key)
    machine_seq = [[] for _ in range(m_val)]

    for job in sorted_jobs:
        best_machine = None
        best_delta = float("inf")
        for i, seq in enumerate(machine_seq):
            _, cost_new = compute_column(seq + [job], p, w)
            _, cost_old = compute_column(seq, p, w) if seq else ([], 0.0)
            delta = cost_new - cost_old
            if delta < best_delta:
                best_delta = delta
                best_machine = i
        machine_seq[best_machine].append(job)

    return machine_seq


def apply_mip_start(X, F, C, p, w, machine_seq):
    n = len(p)
    job_completion = {}

    for i, seq in enumerate(machine_seq):
        cur = 0.0
        for k in range(1, n + 1):
            has_prev = k > 1 and k - 1 <= len(seq)
            current_job = seq[k - 1] if k <= len(seq) else None
            if k > 1 and (k - 1) % c == 0 and has_prev:
                cur += theta
            if current_job is not None:
                cur += p[current_job]
                job_completion[current_job] = cur
            F[i, k].Start = cur

        for k, job in enumerate(seq, start=1):
            X[i, job, k].Start = 1.0

    for j in range(n):
        C[j].Start = job_completion.get(j, 0.0)


# ================== Method 1: Full MIP ==================
def solve_full_MIP(p, w, m_val=None):
    n = len(p)
    m_val = effective_machine_count(m if m_val is None else m_val, n)
    J = range(n)
    M_list = range(m_val)
    K = range(1, n + 1)
    M_big = sum(p) + (n // c + 1) * theta

    model = gp.Model("FullMIP")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", TIME_LIMIT)
    configure_mip(model)

    X = model.addVars(M_list, J, K, vtype=GRB.BINARY, name="X")
    F = model.addVars(M_list, K, lb=0.0, name="F")
    C = model.addVars(J, lb=0.0, name="C")

    for j in J:
        model.addConstr(gp.quicksum(X[i, j, k] for i in M_list for k in K) == 1)
    for i in M_list:
        for k in K:
            model.addConstr(gp.quicksum(X[i, j, k] for j in J) <= 1)
        for k in K[:-1]:
            model.addConstr(
                gp.quicksum(X[i, j, k + 1] for j in J)
                <= gp.quicksum(X[i, j, k] for j in J)
            )

    for i in M_list:
        model.addConstr(F[i, 1] == gp.quicksum(p[j] * X[i, j, 1] for j in J))
        for k in K[1:]:
            prev_has = gp.quicksum(X[i, j, k - 1] for j in J)
            if (k - 1) % c == 0:
                model.addConstr(
                    F[i, k]
                    == F[i, k - 1]
                    + theta * prev_has
                    + gp.quicksum(p[j] * X[i, j, k] for j in J)
                )
            else:
                model.addConstr(
                    F[i, k]
                    == F[i, k - 1] + gp.quicksum(p[j] * X[i, j, k] for j in J)
                )

    for i in M_list:
        for j in J:
            for k in K:
                model.addConstr(C[j] >= F[i, k] - M_big * (1 - X[i, j, k]))

    apply_mip_start(
        X,
        F,
        C,
        p,
        w,
        greedy_ordered_machine_sequences(p, w, m_val, lambda j: (p[j] / w[j], j)),
    )
    model.setObjective(gp.quicksum(w[j] * C[j] for j in J), GRB.MINIMIZE)
    start = time.monotonic()
    model.optimize()
    elapsed = time.monotonic() - start

    result = mip_result(model, elapsed)
    model.dispose()
    return result


# ================== Method 2: Case 1.1 MIP ==================
def solve_case1_1_MIP(p, w, m_val=None):
    n = len(p)
    m_val = effective_machine_count(m if m_val is None else m_val, n)
    J = range(n)
    M_list = range(m_val)
    K = range(1, n + 1)
    M_big = sum(p) + (n // c + 1) * theta

    model = gp.Model("Case1.1MIP")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", TIME_LIMIT)
    configure_mip(model)

    X = model.addVars(M_list, J, K, vtype=GRB.BINARY, name="X")
    F = model.addVars(M_list, K, lb=0.0, name="F")
    C = model.addVars(J, lb=0.0, name="C")

    for j in J:
        model.addConstr(gp.quicksum(X[i, j, k] for i in M_list for k in K) == 1)
    for i in M_list:
        for k in K:
            model.addConstr(gp.quicksum(X[i, j, k] for j in J) <= 1)
        for k in K[:-1]:
            model.addConstr(
                gp.quicksum(X[i, j, k + 1] for j in J)
                <= gp.quicksum(X[i, j, k] for j in J)
            )

    for i in M_list:
        model.addConstr(F[i, 1] == gp.quicksum(p[j] * X[i, j, 1] for j in J))
        for k in K[1:]:
            prev_has = gp.quicksum(X[i, j, k - 1] for j in J)
            if (k - 1) % c == 0:
                model.addConstr(
                    F[i, k]
                    == F[i, k - 1]
                    + theta * prev_has
                    + gp.quicksum(p[j] * X[i, j, k] for j in J)
                )
            else:
                model.addConstr(
                    F[i, k]
                    == F[i, k - 1] + gp.quicksum(p[j] * X[i, j, k] for j in J)
                )

    for i in M_list:
        for j in J:
            for k in K:
                model.addConstr(C[j] >= F[i, k] - M_big * (1 - X[i, j, k]))

    for i in M_list:
        for k in K[:-1]:
            model.addConstr(
                gp.quicksum(w[j] * X[i, j, k] for j in J)
                >= gp.quicksum(w[j] * X[i, j, k + 1] for j in J)
            )

    apply_mip_start(
        X,
        F,
        C,
        p,
        w,
        greedy_ordered_machine_sequences(p, w, m_val, lambda j: (-w[j], p[j], j)),
    )
    model.setObjective(gp.quicksum(w[j] * C[j] for j in J), GRB.MINIMIZE)
    start = time.monotonic()
    model.optimize()
    elapsed = time.monotonic() - start

    result = mip_result(model, elapsed)
    model.dispose()
    return result


# ================== Method 3: Case 1.2 MIP ==================
def solve_case1_2_MIP(p, w, m_val=None):
    n = len(p)
    m_val = effective_machine_count(m if m_val is None else m_val, n)
    J = range(n)
    M_list = range(m_val)
    K = range(1, n + 1)
    M_big = sum(p) + (n // c + 1) * theta

    model = gp.Model("Case1.2MIP")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", TIME_LIMIT)
    configure_mip(model)

    X = model.addVars(M_list, J, K, vtype=GRB.BINARY, name="X")
    F = model.addVars(M_list, K, lb=0.0, name="F")
    C = model.addVars(J, lb=0.0, name="C")

    for j in J:
        model.addConstr(gp.quicksum(X[i, j, k] for i in M_list for k in K) == 1)
    for i in M_list:
        for k in K:
            model.addConstr(gp.quicksum(X[i, j, k] for j in J) <= 1)
        for k in K[:-1]:
            model.addConstr(
                gp.quicksum(X[i, j, k + 1] for j in J)
                <= gp.quicksum(X[i, j, k] for j in J)
            )

    for i in M_list:
        model.addConstr(F[i, 1] == gp.quicksum(p[j] * X[i, j, 1] for j in J))
        for k in K[1:]:
            prev_has = gp.quicksum(X[i, j, k - 1] for j in J)
            if (k - 1) % c == 0:
                model.addConstr(
                    F[i, k]
                    == F[i, k - 1]
                    + theta * prev_has
                    + gp.quicksum(p[j] * X[i, j, k] for j in J)
                )
            else:
                model.addConstr(
                    F[i, k]
                    == F[i, k - 1] + gp.quicksum(p[j] * X[i, j, k] for j in J)
                )

    for i in M_list:
        for j in J:
            for k in K:
                model.addConstr(C[j] >= F[i, k] - M_big * (1 - X[i, j, k]))

    p_max = max(p)
    for i in M_list:
        for k in K[:-1]:
            next_has = gp.quicksum(X[i, j, k + 1] for j in J)
            model.addConstr(
                gp.quicksum(p[j] * X[i, j, k] for j in J)
                <= gp.quicksum(p[j] * X[i, j, k + 1] for j in J)
                + p_max * (1 - next_has)
            )

    apply_mip_start(
        X,
        F,
        C,
        p,
        w,
        greedy_ordered_machine_sequences(p, w, m_val, lambda j: (p[j], -w[j], j)),
    )
    model.setObjective(gp.quicksum(w[j] * C[j] for j in J), GRB.MINIMIZE)
    start = time.monotonic()
    model.optimize()
    elapsed = time.monotonic() - start

    result = mip_result(model, elapsed)
    model.dispose()
    return result


# ================== Method 4: canonical fixed-K column generation ==================
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
# ================== Running, saving, and plotting ==================
def format_obj(value):
    if value is None or math.isinf(value):
        return "inf"
    return f"{value:.1f}"


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


def row_matches_current_machine_count(row, n):
    effective_m = effective_machine_count(m, n)
    row_m = parse_int(row.get("m"))
    row_requested_m = parse_int(row.get("requested_m"))
    if row_m is None:
        return effective_m == 3 and (row_requested_m is None or row_requested_m == 3)
    return row_m == effective_m


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
    if result.get("time_sec") is None:
        return False
    status = str(result.get("status") or "").strip().lower()
    return status not in {"", "none", "nan", "not_started"}


def load_checkpoint(filename=OUTPUT_CSV):
    results = {}
    if not Path(filename).exists():
        print(f"No checkpoint found at {filename}. Starting fresh.")
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
            if inst_id is None or method not in METHODS:
                skipped += 1
                continue
            inst = next((item for item in instances if item["id"] == inst_id), None)
            if inst is None or not row_matches_current_machine_count(row, int(inst["n"])):
                skipped += 1
                continue

            effective_m = effective_machine_count(m, int(inst["n"]))
            requested_m = parse_int(row.get("requested_m")) or effective_m
            row_m = parse_int(row.get("m")) or effective_m
            result = {
                "requested_m": requested_m,
                "m": row_m,
                "machine_count_adjusted": parse_bool(row.get("machine_count_adjusted")) or machine_count_adjusted(requested_m, row_m),
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

            results.setdefault(
                inst_id,
                {"n": inst["n"], "p": inst["p"], "w": inst["w"], "methods": {}},
            )
            results[inst_id]["methods"][method] = result
            loaded += 1

    print(f"Loaded {loaded} finished method records from {filename}; ignored {skipped} incomplete rows.")
    return results


def mark_checkpoint(result):
    result[CHECKPOINT_TIME_FIELD] = datetime.now().isoformat(timespec="seconds")
    return result


def flatten_records(results):
    rows = []
    for inst in instances:
        inst_id = inst["id"]
        if inst_id not in results:
            continue
        n = inst["n"]
        for method, result in results[inst_id]["methods"].items():
            row = {
                "instance_id": inst_id,
                "n": n,
                "requested_m": result.get("requested_m"),
                "m": result.get("m"),
                "machine_count_adjusted": result.get("machine_count_adjusted"),
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
        "requested_m",
        "m",
        "machine_count_adjusted",
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
    Ns = [inst["n"] for inst in instances]
    methods = list(METHODS)
    labels = ["Full MIP", "Case1.1 MIP", "Case1.2 MIP", "Column Generation"]
    markers = ["o", "s", "^", "D"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for method, label, marker in zip(methods, labels, markers):
        vals = []
        for inst in instances:
            obj = results.get(inst["id"], {}).get("methods", {}).get(method, {}).get("objective")
            vals.append(np.nan if obj is None or math.isinf(obj) else obj)
        ax.plot(Ns, vals, marker=marker, label=label)
    ax.set_xlabel("Number of jobs")
    ax.set_ylabel("Total Weighted Completion Time")
    ax.set_title("Objective Value Comparison")
    ax.legend()
    ax.grid(True)

    ax = axes[1]
    for method, label, marker in zip(methods, labels, markers):
        times = []
        for inst in instances:
            times.append(results.get(inst["id"], {}).get("methods", {}).get(method, {}).get("time_sec", np.nan))
        ax.plot(Ns, times, marker=marker, label=label)
    ax.set_xlabel("Number of jobs")
    ax.set_ylabel("Solving Time (seconds)")
    ax.set_title("Computation Time Comparison")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    if os.environ.get("PMCG_SHOW_PLOTS") == "1":
        plt.show()
    plt.close(fig)


def main():
    results = load_checkpoint()

    for inst in instances:
        inst_id = inst["id"]
        n = inst["n"]
        p = inst["p"]
        w = inst["w"]
        m_eff = effective_machine_count(m, n)
        m_was_adjusted = machine_count_adjusted(m, m_eff)
        results.setdefault(inst_id, {"n": n, "p": p, "w": w, "methods": {}})

        print(f"\n{'=' * 20} instance={inst_id}, n={n}, m={m_eff} {'=' * 20}")
        if m_was_adjusted:
            print(f"requested m={m} exceeds n={n}; using m={m_eff}.")
        print(f"p = {p}")
        print(f"w = {w}")

        if is_finished_result(results[inst_id]["methods"].get("FullMIP")):
            r = results[inst_id]["methods"]["FullMIP"]
            print(f"Full MIP already finished, skip. obj={format_obj(r['objective'])}, status={r['status']}")
        else:
            print("Solving Full MIP ...", flush=True)
            results[inst_id]["methods"]["FullMIP"] = mark_checkpoint(solve_full_MIP(p, w, m_eff))
            r = results[inst_id]["methods"]["FullMIP"]
            r["requested_m"] = int(m)
            r["m"] = m_eff
            r["machine_count_adjusted"] = m_was_adjusted
            print(f"  obj={format_obj(r['objective'])}, time={r['time_sec']:.2f}s, status={r['status']}")
            save_csv(results)
            gc.collect()

        if is_finished_result(results[inst_id]["methods"].get("Case11MIP")):
            r = results[inst_id]["methods"]["Case11MIP"]
            print(f"Case1.1 MIP already finished, skip. obj={format_obj(r['objective'])}, status={r['status']}")
        else:
            print("Solving Case1.1 MIP ...", flush=True)
            results[inst_id]["methods"]["Case11MIP"] = mark_checkpoint(solve_case1_1_MIP(p, w, m_eff))
            r = results[inst_id]["methods"]["Case11MIP"]
            r["requested_m"] = int(m)
            r["m"] = m_eff
            r["machine_count_adjusted"] = m_was_adjusted
            print(f"  obj={format_obj(r['objective'])}, time={r['time_sec']:.2f}s, status={r['status']}")
            save_csv(results)
            gc.collect()

        if is_finished_result(results[inst_id]["methods"].get("Case12MIP")):
            r = results[inst_id]["methods"]["Case12MIP"]
            print(f"Case1.2 MIP already finished, skip. obj={format_obj(r['objective'])}, status={r['status']}")
        else:
            print("Solving Case1.2 MIP ...", flush=True)
            results[inst_id]["methods"]["Case12MIP"] = mark_checkpoint(solve_case1_2_MIP(p, w, m_eff))
            r = results[inst_id]["methods"]["Case12MIP"]
            r["requested_m"] = int(m)
            r["m"] = m_eff
            r["machine_count_adjusted"] = m_was_adjusted
            print(f"  obj={format_obj(r['objective'])}, time={r['time_sec']:.2f}s, status={r['status']}")
            save_csv(results)
            gc.collect()

        if is_finished_result(results[inst_id]["methods"].get("CG")):
            r = results[inst_id]["methods"]["CG"]
            print(
                f"Column Generation already finished, skip. obj={format_obj(r['objective'])}, "
                f"status={r['status']}, iters={r.get('iterations')}, cols={r.get('num_columns')}"
            )
        else:
            print(f"Solving Column Generation, fixed K={K_FIXED} ...", flush=True)
            results[inst_id]["methods"]["CG"] = mark_checkpoint(
                column_generation_with_time_limit(p, w, m_eff, K_FIXED, TIME_LIMIT)
            )
            r = results[inst_id]["methods"]["CG"]
            r["requested_m"] = int(m)
            r["m"] = m_eff
            r["machine_count_adjusted"] = m_was_adjusted
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
