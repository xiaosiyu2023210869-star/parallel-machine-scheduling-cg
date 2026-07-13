#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined comparison script.

This script keeps the fixed n/p/w instances and the recorded results for the
recorded first-three-method results from the source experiment, then runs the
improved hybrid column-generation implementation.

Important parameter choices are kept from compare.html so the CG run is
comparable with the recorded MIP results:
    m = 3, c = 4, theta = 2.0, TIME_LIMIT = 600
"""

import csv
import gc
import json
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

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

import gurobipy as gp
from gurobipy import GRB


# ================== Fixed parameters from compare.html ==================
m = 3
c = 4
theta = 2.0
TIME_LIMIT = 600


# ================== Column generation parameters from Untitled7.html ==================
K_FIXED = 30
CG_MAX_ITER = 300
REDUCED_COST_TOL = 1e-6
FINAL_MASTER_RESERVE = 30.0
INCUMBENT_UPDATE_TIME_LIMIT = 1.5
INCUMBENT_UPDATE_PERIOD = 4
DP_PRICING_MAX_N = 20
DP_TOP_PER_STATE_BY_N = ((12, 4), (14, 3), (16, 1))
PRICING_MIP_TIME_LIMIT_CAP = 20.0
PRICING_POOL_SEARCH_MODE = 1
PRICING_POOL_EXTRA = 4
PRICING_MIP_CANDIDATE_LIMIT = 55
PRICING_FORBID_LIMIT = 250
INITIAL_RANDOM_SOLUTIONS = 48
HEURISTIC_PRICING_ROUNDS = 80
HEURISTIC_PRICING_RCL = 7
MAX_COLUMNS_FOR_INT_MASTER = 8000
RANDOM_SEED = 20260524

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


def remaining_seconds(deadline):
    return max(0.0, deadline - time.monotonic())


def final_master_reserve(deadline):
    remaining = remaining_seconds(deadline)
    return min(FINAL_MASTER_RESERVE, max(0.1, 0.1 * remaining))


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


# ================== Method 4: hybrid column generation from Untitled7.html ==================
@dataclass
class PricingResult:
    sequences: list
    status: str
    objective_bound: Optional[float]
    best_objective: Optional[float]
    sol_count: int


@dataclass
class IntegerMasterResult:
    objective: Optional[float]
    status: str
    bound: Optional[float]
    gap: Optional[float]
    sol_count: int
    selected_keys: list


def make_column(seq, p, w, source="generated"):
    seq = list(seq)
    _, cost = compute_column(seq, p, w)
    key = tuple(seq)
    return {
        "seq": seq,
        "cost": float(cost),
        "seq_key": key,
        "job_set": set(seq),
        "source": source,
    }


def add_column_if_new(columns, existing, seq, p, w, source="generated"):
    key = tuple(seq)
    if not key or key in existing:
        return False
    columns.append(make_column(seq, p, w, source))
    existing.add(key)
    return True


def schedule_objective(machine_seq, p, w):
    return sum(compute_column(seq, p, w)[1] for seq in machine_seq if seq)


def schedule_keys(machine_seq):
    return [tuple(seq) for seq in machine_seq if seq]


def validate_schedule(machine_seq, n):
    seen = []
    for seq in machine_seq:
        seen.extend(seq)
    return len(seen) == n and sorted(seen) == list(range(n))


def greedy_ordered_machine_sequences(p, w, order_key):
    sorted_jobs = sorted(range(len(p)), key=order_key)
    machine_seq = [[] for _ in range(m)]

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


def greedy_insert_schedule(order, p, w, m_val, improve_passes=1):
    machine_seq = [[] for _ in range(m_val)]
    machine_costs = [0.0] * m_val

    for job in order:
        best = (float("inf"), 0, 0, None, 0.0)
        for i, seq in enumerate(machine_seq):
            old_cost = machine_costs[i]
            for pos in range(len(seq) + 1):
                trial = seq[:pos] + [job] + seq[pos:]
                _, new_cost = compute_column(trial, p, w)
                delta = new_cost - old_cost
                balance = 1e-5 * len(trial) * len(trial)
                candidate = (delta + balance, i, pos, trial, new_cost)
                if candidate[0] < best[0]:
                    best = candidate
        _, i, _, trial, new_cost = best
        machine_seq[i] = trial
        machine_costs[i] = new_cost

    if improve_passes > 0:
        improve_schedule_by_reinsertion(machine_seq, p, w, passes=improve_passes)
    return machine_seq


def improve_schedule_by_reinsertion(machine_seq, p, w, passes=1):
    n = sum(len(seq) for seq in machine_seq)
    jobs = list(range(n))

    for _ in range(passes):
        improved = False
        for job in jobs:
            source_i = None
            source_pos = None
            for i, seq in enumerate(machine_seq):
                if job in seq:
                    source_i = i
                    source_pos = seq.index(job)
                    break
            if source_i is None or len(machine_seq[source_i]) <= 1:
                continue

            source_seq = machine_seq[source_i]
            source_old_cost = compute_column(source_seq, p, w)[1]
            source_without = source_seq[:source_pos] + source_seq[source_pos + 1 :]
            source_without_cost = compute_column(source_without, p, w)[1] if source_without else 0.0

            best_delta = 0.0
            best_i = source_i
            best_seq = source_seq

            for target_i, target_seq in enumerate(machine_seq):
                if target_i == source_i:
                    base_seq = source_without
                    old_total = source_old_cost
                    for pos in range(len(base_seq) + 1):
                        trial = base_seq[:pos] + [job] + base_seq[pos:]
                        if trial == source_seq:
                            continue
                        new_total = compute_column(trial, p, w)[1]
                        delta = new_total - old_total
                        if delta < best_delta - REDUCED_COST_TOL:
                            best_delta = delta
                            best_i = target_i
                            best_seq = trial
                else:
                    target_old_cost = compute_column(target_seq, p, w)[1] if target_seq else 0.0
                    old_total = source_old_cost + target_old_cost
                    for pos in range(len(target_seq) + 1):
                        trial_target = target_seq[:pos] + [job] + target_seq[pos:]
                        new_total = source_without_cost + compute_column(trial_target, p, w)[1]
                        delta = new_total - old_total
                        if delta < best_delta - REDUCED_COST_TOL:
                            best_delta = delta
                            best_i = target_i
                            best_seq = trial_target

            if best_delta < -REDUCED_COST_TOL:
                machine_seq[source_i] = source_without
                machine_seq[best_i] = best_seq
                improved = True

        if not improved:
            break


def round_robin_schedule(order, m_val):
    machine_seq = [[] for _ in range(m_val)]
    for idx, job in enumerate(order):
        machine_seq[idx % m_val].append(job)
    return machine_seq


def contiguous_split_schedule(order, m_val):
    n = len(order)
    machine_seq = []
    for i in range(m_val):
        start = round(i * n / m_val)
        end = round((i + 1) * n / m_val)
        machine_seq.append(list(order[start:end]))
    return machine_seq


def add_schedule_to_pool(columns, existing, machine_seq, p, w, source):
    if not validate_schedule(machine_seq, len(p)):
        return None, []

    keys = []
    obj = 0.0
    for seq in machine_seq:
        if not seq:
            continue
        key = tuple(seq)
        _, cost = compute_column(seq, p, w)
        obj += cost
        keys.append(key)
        if key not in existing:
            columns.append(
                {
                    "seq": list(seq),
                    "cost": float(cost),
                    "seq_key": key,
                    "job_set": set(seq),
                    "source": source,
                }
            )
            existing.add(key)
    return float(obj), keys


def build_initial_column_pool(p, w, m_val):
    n = len(p)
    jobs = list(range(n))
    rng = random.Random(RANDOM_SEED + 1009 * n + 37 * sum(p) + 17 * sum(w))
    columns = []
    existing = set()
    best_obj = float("inf")
    best_keys = []
    schedule_count = 0

    def remember(machine_seq, source):
        nonlocal best_obj, best_keys, schedule_count
        obj, keys = add_schedule_to_pool(columns, existing, machine_seq, p, w, source)
        if obj is None:
            return
        schedule_count += 1
        if obj < best_obj - REDUCED_COST_TOL:
            best_obj = obj
            best_keys = keys

    ratio_order = sorted(jobs, key=lambda j: (p[j] / max(w[j], 1e-9), p[j], -w[j], j))
    base_orders = [
        ("wspt", ratio_order),
        ("weight_desc", sorted(jobs, key=lambda j: (-w[j], p[j], j))),
        ("processing_asc", sorted(jobs, key=lambda j: (p[j], -w[j], j))),
        ("processing_desc", sorted(jobs, key=lambda j: (-p[j], -w[j], j))),
        ("smith_desc", sorted(jobs, key=lambda j: (-w[j] / max(p[j], 1e-9), p[j], j))),
        ("index", jobs),
    ]

    for name, order in base_orders:
        remember(greedy_insert_schedule(order, p, w, m_val, improve_passes=2), f"initial_{name}_insert")
        remember(round_robin_schedule(order, m_val), f"initial_{name}_roundrobin")
        remember(contiguous_split_schedule(order, m_val), f"initial_{name}_split")

    for r in range(INITIAL_RANDOM_SOLUTIONS):
        if r % 3 == 0:
            order = sorted(
                jobs,
                key=lambda j: (
                    math.log((p[j] / max(w[j], 1e-9)) + 1e-9) + rng.gauss(0.0, 0.45),
                    rng.random(),
                ),
            )
        elif r % 3 == 1:
            order = sorted(
                jobs,
                key=lambda j: (
                    -(w[j] / max(p[j], 1e-9)) + rng.gauss(0.0, 0.35),
                    rng.random(),
                ),
            )
        else:
            order = jobs[:]
            rng.shuffle(order)

        remember(
            greedy_insert_schedule(order, p, w, m_val, improve_passes=1 if r < 16 else 0),
            f"initial_random_{r}",
        )

    if not best_keys:
        fallback = greedy_ordered_machine_sequences(p, w, lambda j: (p[j] / w[j], j))
        remember(fallback, "initial_fallback")

    return columns, best_obj, best_keys, schedule_count


def build_rmp(columns, n, m_val, time_limit=None):
    model = gp.Model("RMP")
    model.setParam("OutputFlag", 0)
    model.setParam("Method", 1)
    model.setParam("Presolve", 2)
    if time_limit is not None:
        model.setParam("TimeLimit", max(1e-3, time_limit))

    col_ids = list(range(len(columns)))
    lambdas = model.addVars(
        col_ids,
        lb=0.0,
        ub=1.0,
        obj=[columns[i]["cost"] for i in col_ids],
        name="lambda",
    )

    job_to_cols = [[] for _ in range(n)]
    for i, col in enumerate(columns):
        for j in col["job_set"]:
            job_to_cols[j].append(i)

    cover = {}
    for j in range(n):
        cover[j] = model.addConstr(
            gp.quicksum(lambdas[i] for i in job_to_cols[j]) == 1,
            name=f"cover_{j}",
        )
    mach_constr = model.addConstr(
        gp.quicksum(lambdas[i] for i in col_ids) == m_val,
        name="machines",
    )
    model.setObjective(
        gp.quicksum(lambdas[i] * columns[i]["cost"] for i in col_ids),
        GRB.MINIMIZE,
    )
    return model, lambdas, cover, mach_constr


def sequence_reduced_cost(seq, p, w, pi, mu):
    _, cost = compute_column(seq, p, w)
    return cost - sum(pi[j] for j in seq) - mu


def improve_sequence_order(seq, p, w, max_passes=2):
    if len(seq) <= 2:
        return list(seq)

    best = sorted(seq, key=lambda j: (p[j] / max(w[j], 1e-9), p[j], -w[j], j))
    best_cost = compute_column(best, p, w)[1]
    current = list(seq)
    current_cost = compute_column(current, p, w)[1]
    if current_cost < best_cost:
        best = current
        best_cost = current_cost

    for _ in range(max_passes):
        improved = False
        for idx in range(len(best)):
            job = best[idx]
            without = best[:idx] + best[idx + 1 :]
            for pos in range(len(without) + 1):
                if pos == idx:
                    continue
                trial = without[:pos] + [job] + without[pos:]
                cost = compute_column(trial, p, w)[1]
                if cost < best_cost - REDUCED_COST_TOL:
                    best = trial
                    best_cost = cost
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best


def select_candidate_jobs_for_pricing(p, w, pi, limit):
    n = len(p)
    jobs = list(range(n))
    limit = min(max(1, limit), n)
    chosen = []
    seen = set()

    orderings = [
        sorted(jobs, key=lambda j: (-pi[j], p[j], -w[j], j)),
        sorted(jobs, key=lambda j: (-(pi[j] - w[j] * p[j]), p[j], j)),
        sorted(jobs, key=lambda j: (-(pi[j] / max(p[j], 1e-9)), p[j], j)),
        sorted(jobs, key=lambda j: (p[j] / max(w[j], 1e-9), -pi[j], j)),
        sorted(jobs, key=lambda j: (-w[j], p[j], -pi[j], j)),
    ]

    quota = max(1, math.ceil(limit / len(orderings)))
    for order in orderings:
        for job in order[:quota]:
            if job not in seen:
                chosen.append(job)
                seen.add(job)
            if len(chosen) >= limit:
                return chosen

    combined = sorted(
        jobs,
        key=lambda j: (-(pi[j] - 0.25 * w[j] * p[j]), p[j] / max(w[j], 1e-9), j),
    )
    for job in combined:
        if job not in seen:
            chosen.append(job)
            seen.add(job)
        if len(chosen) >= limit:
            break
    return chosen


def reconstruct_sequence(parent_mask, parent_job, mask):
    seq = []
    while mask:
        job = parent_job[mask]
        seq.append(job)
        mask = parent_mask[mask]
    seq.reverse()
    return tuple(seq)


def dp_top_per_state(n, max_columns):
    target = max(1, max_columns)
    for n_limit, cap in DP_TOP_PER_STATE_BY_N:
        if n <= n_limit:
            return min(target, cap)
    return 1


def solve_pricing_dp_single(p, w, pi, mu, max_columns, deadline, forbidden):
    n = len(p)
    size = 1 << n
    full = size - 1
    inf = float("inf")

    p_sum = [0.0] * size
    for mask in range(1, size):
        bit = mask & -mask
        job = bit.bit_length() - 1
        p_sum[mask] = p_sum[mask ^ bit] + p[job]

    dp = [inf] * size
    parent_mask = [-1] * size
    parent_job = [-1] * size
    dp[0] = 0.0

    for mask in range(size):
        base_cost = dp[mask]
        if base_cost == inf:
            continue
        if mask and mask % 4096 == 0 and remaining_seconds(deadline) <= final_master_reserve(deadline):
            return PricingResult([], "TIME_LIMIT", None, None, 0)

        length = mask.bit_count()
        if length == n:
            continue

        cur_time = 0.0 if length == 0 else p_sum[mask] + ((length - 1) // c) * theta
        setup = theta if length > 0 and length % c == 0 else 0.0
        remaining_jobs = full ^ mask

        while remaining_jobs:
            bit = remaining_jobs & -remaining_jobs
            job = bit.bit_length() - 1
            remaining_jobs ^= bit

            new_mask = mask | bit
            completion = cur_time + setup + p[job]
            new_cost = base_cost + w[job] * completion - pi[job]
            if new_cost < dp[new_mask] - REDUCED_COST_TOL:
                dp[new_mask] = new_cost
                parent_mask[new_mask] = mask
                parent_job[new_mask] = job

    best_reduced_cost = None
    candidates = []
    seen = set()
    blocked_negative = False

    for mask in range(1, size):
        if dp[mask] == inf:
            continue
        reduced_cost = dp[mask] - mu
        if best_reduced_cost is None or reduced_cost < best_reduced_cost:
            best_reduced_cost = reduced_cost
        if reduced_cost >= -REDUCED_COST_TOL:
            continue

        seq = reconstruct_sequence(parent_mask, parent_job, mask)
        if seq in forbidden:
            blocked_negative = True
            continue
        if seq in seen:
            continue
        candidates.append((reduced_cost, list(seq)))
        seen.add(seq)

    candidates.sort(key=lambda item: item[0])
    sequences = [seq for _, seq in candidates[:max_columns]]

    if sequences:
        return PricingResult(sequences, "OPTIMAL", best_reduced_cost, best_reduced_cost, len(candidates))
    if blocked_negative:
        return PricingResult([], "DP_FORBIDDEN_AMBIGUOUS", best_reduced_cost, best_reduced_cost, 0)
    return PricingResult([], "OPTIMAL", best_reduced_cost, best_reduced_cost, 0)


def solve_pricing_dp(p, w, pi, mu, max_columns, deadline, forbidden_seqs=()):
    n = len(p)
    if n > DP_PRICING_MAX_N:
        return None

    time_limit = remaining_seconds(deadline) - final_master_reserve(deadline)
    if max_columns <= 0 or time_limit <= 0:
        return PricingResult([], "TIME_LIMIT", None, None, 0)

    top_limit = dp_top_per_state(n, max_columns)
    forbidden = {tuple(seq) for seq in forbidden_seqs}
    if top_limit == 1:
        return solve_pricing_dp_single(p, w, pi, mu, max_columns, deadline, forbidden)

    size = 1 << n
    full = size - 1

    p_sum = [0.0] * size
    for mask in range(1, size):
        bit = mask & -mask
        job = bit.bit_length() - 1
        p_sum[mask] = p_sum[mask ^ bit] + p[job]

    dp = [[] for _ in range(size)]
    dp[0] = [(0.0, ())]

    for mask in range(size):
        if not dp[mask]:
            continue
        if mask and mask % 2048 == 0 and remaining_seconds(deadline) <= final_master_reserve(deadline):
            return PricingResult([], "TIME_LIMIT", None, None, 0)

        length = mask.bit_count()
        if length == n:
            continue

        cur_time = 0.0 if length == 0 else p_sum[mask] + ((length - 1) // c) * theta
        setup = theta if length > 0 and length % c == 0 else 0.0
        available = full ^ mask

        for cost, seq in dp[mask]:
            remaining_jobs = available
            while remaining_jobs:
                bit = remaining_jobs & -remaining_jobs
                job = bit.bit_length() - 1
                remaining_jobs ^= bit

                new_mask = mask | bit
                completion = cur_time + setup + p[job]
                new_cost = cost + w[job] * completion - pi[job]
                bucket = dp[new_mask]
                bucket.append((new_cost, seq + (job,)))
                if len(bucket) > top_limit * 4:
                    bucket.sort(key=lambda item: item[0])
                    del bucket[top_limit:]

    candidates = []
    seen = set()
    best_reduced_cost = None
    blocked_negative = False

    for bucket in dp[1:]:
        if not bucket:
            continue
        bucket.sort(key=lambda item: item[0])
        del bucket[top_limit:]

        for cost, seq in bucket:
            reduced_cost = cost - mu
            if best_reduced_cost is None or reduced_cost < best_reduced_cost:
                best_reduced_cost = reduced_cost
            if reduced_cost >= -REDUCED_COST_TOL:
                break
            if seq in forbidden:
                blocked_negative = True
                continue
            if seq in seen:
                continue
            candidates.append((reduced_cost, list(seq)))
            seen.add(seq)

    candidates.sort(key=lambda item: item[0])
    sequences = [seq for _, seq in candidates[:max_columns]]

    if sequences:
        status = "OPTIMAL"
    elif blocked_negative:
        status = "DP_FORBIDDEN_AMBIGUOUS"
    else:
        status = "OPTIMAL"

    return PricingResult(
        sequences,
        status,
        best_reduced_cost,
        best_reduced_cost,
        len(candidates),
    )


def solve_pricing_heuristic(p, w, pi, mu, max_columns, deadline, forbidden_seqs=()):
    n = len(p)
    rng = random.Random(RANDOM_SEED + n + int(abs(mu) * 1000) % 1000003)
    forbidden = {tuple(seq) for seq in forbidden_seqs}
    seen = set()
    candidates = []
    best_rc = None

    def try_add(seq, source):
        nonlocal best_rc
        if not seq:
            return
        key = tuple(seq)
        if key in forbidden or key in seen:
            return
        rc = sequence_reduced_cost(seq, p, w, pi, mu)
        best_rc = rc if best_rc is None else min(best_rc, rc)
        if rc < -REDUCED_COST_TOL:
            candidates.append((rc, list(seq), source))
            seen.add(key)

    jobs = list(range(n))
    orders = [
        sorted(jobs, key=lambda j: (-pi[j], p[j] / max(w[j], 1e-9), j)),
        sorted(jobs, key=lambda j: (-(pi[j] - w[j] * p[j]), p[j], j)),
        sorted(jobs, key=lambda j: (p[j] / max(w[j], 1e-9), -pi[j], j)),
        sorted(jobs, key=lambda j: (-w[j], p[j], -pi[j], j)),
    ]

    for order_no, order in enumerate(orders):
        prefix = []
        for job in order:
            prefix.append(job)
            if len(prefix) <= 3 or len(prefix) % 2 == 0:
                try_add(prefix, f"prefix_{order_no}")
                if len(prefix) >= 4:
                    try_add(improve_sequence_order(prefix, p, w, max_passes=1), f"prefix_sorted_{order_no}")
            if remaining_seconds(deadline) <= final_master_reserve(deadline):
                break

    pool_limit = min(n, max(PRICING_MIP_CANDIDATE_LIMIT + 20, max_columns))
    base_pool = select_candidate_jobs_for_pricing(p, w, pi, pool_limit)

    for round_no in range(HEURISTIC_PRICING_ROUNDS):
        if remaining_seconds(deadline) <= final_master_reserve(deadline):
            break

        pool = base_pool[:]
        if round_no % 4 != 0:
            rng.shuffle(pool)

        seq = []
        used = set()
        cur_time = 0.0
        for step in range(len(pool)):
            setup = theta if step > 0 and step % c == 0 else 0.0
            options = []
            for job in pool:
                if job in used:
                    continue
                completion = cur_time + setup + p[job]
                delta = w[job] * completion - pi[job]
                noise = rng.random() * max(1.0, abs(delta)) * 0.03
                options.append((delta + noise, delta, job, completion))
            if not options:
                break

            options.sort(key=lambda item: item[0])
            rcl = options[: min(HEURISTIC_PRICING_RCL, len(options))]
            if round_no % 5 == 0:
                _, delta, job, completion = rcl[0]
            else:
                _, delta, job, completion = rng.choice(rcl)

            seq.append(job)
            used.add(job)
            cur_time = completion

            if delta > 0 and len(seq) > max(4, n // (2 * m)):
                try_add(seq, f"greedy_{round_no}")
                try_add(improve_sequence_order(seq, p, w, max_passes=1), f"greedy_sorted_{round_no}")
                break

            if len(seq) <= 5 or len(seq) % 3 == 0:
                try_add(seq, f"greedy_{round_no}")
                if len(seq) >= 4:
                    try_add(improve_sequence_order(seq, p, w, max_passes=1), f"greedy_sorted_{round_no}")

        try_add(seq, f"greedy_final_{round_no}")
        if len(seq) >= 4:
            try_add(improve_sequence_order(seq, p, w, max_passes=2), f"greedy_final_sorted_{round_no}")

    candidates.sort(key=lambda item: item[0])
    selected = []
    selected_seen = set()
    for _, seq, _ in candidates:
        key = tuple(seq)
        if key in selected_seen:
            continue
        selected.append(seq)
        selected_seen.add(key)
        if len(selected) >= max_columns:
            break

    return PricingResult(selected, "HEURISTIC", best_rc, best_rc, len(candidates))


def add_forbidden_sequence_constraints_limited(model, x, u, forbidden_seqs, candidate_set, max_len):
    added = 0
    for seq in forbidden_seqs:
        if added >= PRICING_FORBID_LIMIT:
            break
        seq = tuple(seq)
        length = len(seq)
        if length == 0 or length > max_len:
            continue
        if any(job not in candidate_set for job in seq):
            continue

        match_expr = gp.quicksum(x[job, pos] for pos, job in enumerate(seq))
        if length < max_len:
            model.addConstr(match_expr - u[length] <= length - 1)
        else:
            model.addConstr(match_expr <= length - 1)
        added += 1


def solve_pricing_mip_restricted(p, w, pi, mu, max_columns, deadline, forbidden_seqs=()):
    time_limit = remaining_seconds(deadline) - final_master_reserve(deadline)
    if max_columns <= 0 or time_limit <= 0:
        return PricingResult([], "TIME_LIMIT", None, None, 0)

    candidate_jobs = select_candidate_jobs_for_pricing(p, w, pi, PRICING_MIP_CANDIDATE_LIMIT)
    positions = range(len(candidate_jobs))
    candidate_set = set(candidate_jobs)
    time_limit = min(time_limit, PRICING_MIP_TIME_LIMIT_CAP)
    horizon = sum(p[j] for j in candidate_jobs) + max(0, (len(candidate_jobs) - 1) // c) * theta
    pool_solutions = max(max_columns, max_columns * PRICING_POOL_EXTRA)

    model = gp.Model("PricingRestricted")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", max(1e-3, time_limit))
    model.setParam("PoolSolutions", pool_solutions)
    model.setParam("PoolSearchMode", PRICING_POOL_SEARCH_MODE)
    model.setParam("MIPFocus", 1)
    model.setParam("Heuristics", 0.35)
    model.setParam("Presolve", 2)

    x = model.addVars(candidate_jobs, positions, vtype=GRB.BINARY, name="x")
    u = model.addVars(positions, vtype=GRB.BINARY, name="u")
    comp = model.addVars(positions, lb=0.0, ub=horizon, name="comp")
    z = model.addVars(candidate_jobs, positions, lb=0.0, ub=horizon, name="z")

    for job in candidate_jobs:
        model.addConstr(gp.quicksum(x[job, pos] for pos in positions) <= 1)
    for pos in positions:
        model.addConstr(gp.quicksum(x[job, pos] for job in candidate_jobs) == u[pos])
    model.addConstr(u[0] == 1)
    for pos in range(1, len(candidate_jobs)):
        model.addConstr(u[pos] <= u[pos - 1])

    model.addConstr(comp[0] == gp.quicksum(p[job] * x[job, 0] for job in candidate_jobs))
    for pos in range(1, len(candidate_jobs)):
        setup = theta if pos % c == 0 else 0.0
        model.addConstr(
            comp[pos]
            == comp[pos - 1]
            + setup * u[pos]
            + gp.quicksum(p[job] * x[job, pos] for job in candidate_jobs)
        )

    for job in candidate_jobs:
        for pos in positions:
            model.addConstr(z[job, pos] <= horizon * x[job, pos])
            model.addConstr(z[job, pos] <= comp[pos])
            model.addConstr(z[job, pos] >= comp[pos] - horizon * (1 - x[job, pos]))

    add_forbidden_sequence_constraints_limited(
        model,
        x,
        u,
        forbidden_seqs,
        candidate_set,
        len(candidate_jobs),
    )

    model.setObjective(
        gp.quicksum(
            w[job] * z[job, pos] - pi[job] * x[job, pos]
            for job in candidate_jobs
            for pos in positions
        )
        - mu,
        GRB.MINIMIZE,
    )
    model.optimize()

    sequences = []
    seen = {tuple(seq) for seq in forbidden_seqs}
    sol_count = int(model.SolCount)
    best_obj = None
    bound = safe_model_attr(model, "ObjBound")
    bound = None if bound is None else float(bound)

    if sol_count > 0:
        best_obj = float(model.ObjVal)
        for sol_no in range(sol_count):
            model.setParam("SolutionNumber", sol_no)
            reduced_cost = float(model.PoolObjVal)
            if reduced_cost >= -REDUCED_COST_TOL:
                continue

            seq = []
            for pos in positions:
                if u[pos].Xn < 0.5:
                    break
                chosen = [job for job in candidate_jobs if x[job, pos].Xn > 0.5]
                if not chosen:
                    break
                seq.append(chosen[0])

            key = tuple(seq)
            if key and key not in seen:
                sequences.append(seq)
                seen.add(key)
            if len(sequences) >= max_columns:
                break

    status = status_name(model.Status)
    if sequences and status == "TIME_LIMIT":
        status = "TIME_LIMIT_WITH_COLUMNS"
    result = PricingResult(sequences, status, bound, best_obj, sol_count)
    model.dispose()
    return result


def solve_pricing_multiple(p, w, pi, mu, max_columns, deadline, forbidden_seqs=()):
    dp_result = solve_pricing_dp(p, w, pi, mu, max_columns, deadline, forbidden_seqs=forbidden_seqs)
    if dp_result is not None and dp_result.sequences:
        return dp_result
    if dp_result is not None and dp_result.status == "TIME_LIMIT":
        return dp_result

    heuristic = solve_pricing_heuristic(
        p,
        w,
        pi,
        mu,
        max_columns,
        deadline,
        forbidden_seqs=forbidden_seqs,
    )
    if len(heuristic.sequences) >= max_columns:
        return heuristic

    mip = solve_pricing_mip_restricted(
        p,
        w,
        pi,
        mu,
        max_columns - len(heuristic.sequences),
        deadline,
        forbidden_seqs=forbidden_seqs,
    )

    combined = []
    seen = set()
    for seq in heuristic.sequences + mip.sequences:
        key = tuple(seq)
        if key in seen:
            continue
        combined.append(seq)
        seen.add(key)
        if len(combined) >= max_columns:
            break

    best_values = [v for v in (heuristic.best_objective, mip.best_objective) if v is not None]
    best_obj = min(best_values) if best_values else None
    bound_values = [v for v in (heuristic.objective_bound, mip.objective_bound) if v is not None]
    bound = min(bound_values) if bound_values else None
    status = f"{heuristic.status}+{mip.status}"
    return PricingResult(combined, status, bound, best_obj, heuristic.sol_count + mip.sol_count)


def select_columns_for_integer_master(columns, warm_start_keys=()):
    if len(columns) <= MAX_COLUMNS_FOR_INT_MASTER:
        return list(range(len(columns)))

    warm_start_keys = set(warm_start_keys or [])
    selected = {i for i, col in enumerate(columns) if col["seq_key"] in warm_start_keys}
    n = max((max(col["job_set"]) for col in columns if col["job_set"]), default=-1) + 1
    job_to_cols = [[] for _ in range(n)]
    for i, col in enumerate(columns):
        for job in col["job_set"]:
            job_to_cols[job].append(i)

    for job_cols in job_to_cols:
        job_cols.sort(key=lambda i: columns[i]["cost"] / max(1, len(columns[i]["seq"])))
        selected.update(job_cols[:60])

    remaining = [
        i
        for i in range(len(columns))
        if i not in selected
    ]
    remaining.sort(key=lambda i: columns[i]["cost"] / max(1, len(columns[i]["seq"]) ** 0.7))
    for i in remaining:
        selected.add(i)
        if len(selected) >= MAX_COLUMNS_FOR_INT_MASTER:
            break
    return sorted(selected)


def solve_integer_master(columns, n, m_val, deadline, time_limit_cap=None, warm_start_keys=()):
    time_limit = remaining_seconds(deadline)
    if time_limit <= 0:
        return IntegerMasterResult(None, "SKIPPED_NO_TIME", None, None, 0, [])
    if time_limit_cap is not None:
        time_limit = min(time_limit, time_limit_cap)
    if time_limit <= 0:
        return IntegerMasterResult(None, "SKIPPED_NO_TIME", None, None, 0, [])

    active_ids = select_columns_for_integer_master(columns, warm_start_keys)
    model = gp.Model("IntMaster")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", max(1e-3, time_limit))
    model.setParam("MIPFocus", 1)
    model.setParam("Heuristics", 0.45)
    model.setParam("Presolve", 2)

    y = model.addVars(
        active_ids,
        vtype=GRB.BINARY,
        obj={i: columns[i]["cost"] for i in active_ids},
        name="y",
    )
    job_to_cols = [[] for _ in range(n)]
    for i in active_ids:
        for job in columns[i]["job_set"]:
            job_to_cols[job].append(i)

    for job in range(n):
        model.addConstr(gp.quicksum(y[i] for i in job_to_cols[job]) == 1)
    model.addConstr(gp.quicksum(y[i] for i in active_ids) == m_val)

    warm_start_keys = set(warm_start_keys or [])
    for i in active_ids:
        y[i].Start = 1.0 if columns[i]["seq_key"] in warm_start_keys else 0.0

    model.optimize()

    obj = float(model.ObjVal) if model.SolCount > 0 else None
    bound = safe_model_attr(model, "ObjBound")
    gap = safe_model_attr(model, "MIPGap")
    selected_keys = []
    if model.SolCount > 0:
        selected_keys = [columns[i]["seq_key"] for i in active_ids if y[i].X > 0.5]

    result = IntegerMasterResult(
        objective=obj,
        status=status_name(model.Status),
        bound=None if bound is None else float(bound),
        gap=None if gap is None else float(gap),
        sol_count=int(model.SolCount),
        selected_keys=selected_keys,
    )
    model.dispose()
    return result


def column_generation_with_time_limit(p, w, m_val, k_fixed, time_limit):
    n = len(p)
    jobs = list(range(n))
    start = time.monotonic()
    deadline = start + time_limit

    columns, initial_obj, incumbent_keys, initial_schedule_count = build_initial_column_pool(p, w, m_val)
    initial_column_count = len(columns)
    incumbent_obj = initial_obj
    incumbent_source = "initial_pool_best_schedule"
    incumbent_status = "HEURISTIC_INITIAL_POOL"
    last_integer_update_iter = 0
    existing = {col["seq_key"] for col in columns}

    iter_count = 0
    status = "NOT_STARTED"
    last_pricing_status = None
    last_rmp_obj = None
    last_best_reduced_cost = None
    last_pricing_sol_count = 0
    had_pricing_time_limit = False
    final_int_result = IntegerMasterResult(None, "NOT_RUN", None, None, 0, [])

    while iter_count < CG_MAX_ITER:
        if remaining_seconds(deadline) <= final_master_reserve(deadline):
            status = "TIME_LIMIT"
            break

        iter_count += 1
        rmp, _, cover, mach_constr = build_rmp(
            columns,
            n,
            m_val,
            time_limit=remaining_seconds(deadline) - final_master_reserve(deadline),
        )
        rmp.optimize()

        if rmp.Status != GRB.OPTIMAL:
            status = "RMP_" + status_name(rmp.Status)
            rmp.dispose()
            break

        last_rmp_obj = float(rmp.ObjVal)
        pi = {job: cover[job].Pi for job in jobs}
        mu = mach_constr.Pi
        rmp.dispose()

        pricing = solve_pricing_multiple(p, w, pi, mu, k_fixed, deadline, forbidden_seqs=existing)
        last_pricing_status = pricing.status
        last_best_reduced_cost = pricing.best_objective
        last_pricing_sol_count = pricing.sol_count
        had_pricing_time_limit = had_pricing_time_limit or "TIME_LIMIT" in pricing.status

        added = 0
        for seq in pricing.sequences:
            if add_column_if_new(columns, existing, seq, p, w, source=f"pricing_iter_{iter_count}"):
                added += 1

        if added == 0:
            if pricing.status == "OPTIMAL":
                status = "CONVERGED"
            elif "TIME_LIMIT" in pricing.status:
                status = "PRICING_TIME_LIMIT"
            else:
                status = "NO_NEW_COLUMNS_" + pricing.status
            break

        should_update = (
            iter_count == 1
            or iter_count % INCUMBENT_UPDATE_PERIOD == 0
            or added >= max(10, k_fixed // 2)
        )
        if should_update and remaining_seconds(deadline) > final_master_reserve(deadline):
            update = solve_integer_master(
                columns,
                n,
                m_val,
                deadline,
                time_limit_cap=INCUMBENT_UPDATE_TIME_LIMIT,
                warm_start_keys=incumbent_keys,
            )
            if update.objective is not None:
                incumbent_status = update.status
                last_integer_update_iter = iter_count
                if update.objective < incumbent_obj - REDUCED_COST_TOL:
                    incumbent_obj = update.objective
                    incumbent_source = f"iter_{iter_count}_integer_master"
                    incumbent_keys = update.selected_keys
            elif update.status != "SKIPPED_NO_TIME":
                incumbent_status = update.status
    else:
        status = "MAX_ITER"

    final_int_result = solve_integer_master(
        columns,
        n,
        m_val,
        deadline,
        warm_start_keys=incumbent_keys,
    )
    int_status = final_int_result.status
    if final_int_result.objective is not None:
        incumbent_status = final_int_result.status
        if final_int_result.objective < incumbent_obj - REDUCED_COST_TOL:
            incumbent_obj = final_int_result.objective
            incumbent_source = "final_integer_master"
            incumbent_keys = final_int_result.selected_keys
    else:
        int_status = int_status + "_USED_INCUMBENT"

    elapsed = time.monotonic() - start
    return {
        "objective": incumbent_obj,
        "time_sec": elapsed,
        "status": status,
        "timed_out": (
            elapsed >= time_limit - 1e-3
            or status in {"TIME_LIMIT", "PRICING_TIME_LIMIT"}
            or had_pricing_time_limit
        ),
        "iterations": iter_count,
        "pricing_status": last_pricing_status,
        "had_pricing_time_limit": had_pricing_time_limit,
        "integer_status": int_status,
        "incumbent_status": incumbent_status,
        "objective_source": incumbent_source,
        "last_integer_update_iter": last_integer_update_iter,
        "num_columns": len(columns),
        "initial_columns": initial_column_count,
        "initial_schedules": initial_schedule_count,
        "initial_objective": initial_obj,
        "last_rmp_obj": last_rmp_obj,
        "best_reduced_cost": last_best_reduced_cost,
        "pricing_sol_count": last_pricing_sol_count,
        "final_int_bound": final_int_result.bound,
        "final_int_gap": final_int_result.gap,
        "final_int_sol_count": final_int_result.sol_count,
        "K": k_fixed,
    }


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
