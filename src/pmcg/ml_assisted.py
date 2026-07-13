# -*- coding: utf-8 -*-
"""
Corrected and time-limited column generation experiment.

This file mirrors the notebook code, but fixes the pricing model and makes every
(instance, K) run respect the wall-clock time budget.
"""

import ast
import gc
import json
import math
import os
import pickle
import random
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
try:
    from scipy.stats import skew
except ImportError:
    def skew(values):
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return 0.0
        centered = arr - arr.mean()
        std = arr.std()
        if std == 0:
            return 0.0
        return float(np.mean((centered / std) ** 3))

import gurobipy as gp
from gurobipy import GRB


# -------------------- Parameters --------------------
m = 3
c = 2
theta = 4.0

TIME_LIMIT = 1800
CG_MAX_ITER = 300
K_VALUES = list(range(5, 51, 5))

# Resume controls.  The CSV result file is the checkpoint: any saved
# (instance_id, K) row is treated as already run, even if it timed out.
# Because the current manual run has reached n=22, K=30, the next fresh
# launch skips everything up to and including that pair.  Set this to None
# after you no longer need the manual cutoff.
MANUAL_RESUME_AFTER = None
RESUME_RUN_ID = "training_excel_n22_to_n62_resume_v1"
CHECKPOINT_VERSION_COLUMN = "checkpoint_version"
CHECKPOINT_TIME_COLUMN = "checkpoint_saved_at"

REDUCED_COST_TOL = 1e-6
FINAL_MASTER_RESERVE = 360.0
FINAL_K_HEURISTIC_TIME_LIMIT = 120.0
FINAL_RMP_REFRESH_TIME_LIMIT = 35.0
FINAL_GENERATED_INTEGER_TIME_LIMIT = 170.0
FINAL_INTEGER_POLISH_TIME_LIMIT = 0.0
INCUMBENT_UPDATE_TIME_LIMIT = 3.0
DP_PRICING_MAX_N = 20
DP_TOP_PER_STATE_BY_N = ((12, 4), (14, 3), (16, 1))
PRICING_MIP_TIME_LIMIT_CAP = 20.0
PRICING_POOL_SEARCH_MODE = 1
PRICING_POOL_EXTRA = 4
INCUMBENT_UPDATE_PERIOD = 4
PRICING_MIP_CANDIDATE_LIMIT = 55
PRICING_FORBID_LIMIT = 250
INITIAL_RANDOM_SOLUTIONS = 48
HEURISTIC_PRICING_ROUNDS = 80
HEURISTIC_PRICING_RCL = 7
MAX_COLUMNS_FOR_INT_MASTER = 8000
INT_MASTER_COLUMNS_PER_JOB = 60
FINAL_MAX_COLUMNS_FOR_INT_MASTER = 20000
FINAL_INT_MASTER_COLUMNS_PER_JOB = 140
LAST_RMP_POSITIVE_TOL = 1e-6
ROUNDING_MAX_SCHEDULES = 18
ROUNDING_RANDOM_ATTEMPTS = 8
ROUNDING_IMPROVE_PASSES = 1
FINAL_K_SCHEDULE_MIN = 6
FINAL_K_SCHEDULE_CAP = 42
FINAL_K_SCHEDULE_FACTOR = 1.0
RANDOM_SEED = 20260524
TRAINING_INITIAL_RANDOM_SOLUTIONS = 6

# Repository paths are configured through PMCG_TRAINING_DATA and PMCG_OUTPUT_DIR.




PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(os.environ.get("PMCG_OUTPUT_DIR", PROJECT_ROOT / "outputs" / "ml-assisted"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_DATA_FILE = Path(os.environ.get("PMCG_TRAINING_DATA", PROJECT_ROOT / "data" / "training_data.xlsx"))
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
COMPARISON_OUTPUT_FILE = OUTPUT_DIR / f"ml_assisted_results_{RUN_TIMESTAMP}.xlsx"
MODEL_FILE = OUTPUT_DIR / "k_selection_model.pkl"
ONLINE_FEEDBACK_FILE = OUTPUT_DIR / "online_feedback.xlsx"
OUTPUT_FILE = COMPARISON_OUTPUT_FILE
BASELINE_K = 30
ML_EXACT_SMALL_N_THRESHOLD = 16
ML_SMALL_N_THRESHOLD = ML_EXACT_SMALL_N_THRESHOLD
ML_OBJECTIVE_TOL = 1e-6
EXACT_OPTIMAL_TOL = 1e-6
ML_NEIGHBORS = 5
LARGE_OBJECTIVE_REL_TOL = 0.005
ONLINE_LEARNING_ENABLED = True
ONLINE_TRAINING_METHODS = {"baseline_fixed_k", "ml_selected_k"}
ONLINE_REWARD_WEIGHT = 2.0
ONLINE_PENALTY_WEIGHT = 8.0
ONLINE_ACCEPTABLE_LARGE_REL_LOSS = 0.005
COMPARISON_RUN_VERSION = "dual_scale_time_focus_all_training_data_v1"
ML_LABEL_RULE_VERSION = "all_rows_small_best_obj_fastest_large_near_best_fastest_v1"
FEATURE_COLUMNS = [f"feat_{idx}" for idx in range(14)]
ML_OBJECTIVE_COLUMN = "__ml_objective"
ML_OBJECTIVE_SOURCE_COLUMN = "__ml_objective_source"
COMPARISON_MIN_N = 5
COMPARISON_MAX_N = 36
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INSTANCES_FILE = PROJECT_ROOT / "data" / "instances_ml.json"
INSTANCES_FILE = Path(os.environ.get("PMCG_INSTANCES_FILE", DEFAULT_INSTANCES_FILE))
INLINE_COMPARISON_INSTANCES = json.loads(INSTANCES_FILE.read_text(encoding="utf-8"))
REQUIRED_RESULT_COLUMNS = {
    "instance_id",
    "K",
    "objective",
    "time_sec",
    "timed_out",
    "iterations",
    "cg_status",
    "pricing_status",
    "integer_status",
    "num_columns",
}


def parse_sequence_cell(value, field_name, instance_id):
    if isinstance(value, list):
        return [int(item) for item in value]
    if pd.isna(value):
        raise ValueError(f"Instance {instance_id} has an empty {field_name} value")
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError) as exc:
        raise ValueError(
            f"Instance {instance_id} has an invalid {field_name} value: {value!r}"
        ) from exc
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"Instance {instance_id} {field_name} value is not a list: {value!r}")
    return [int(item) for item in parsed]


def feature_signature(values):
    return tuple(round(float(value), 10) for value in values)


def feature_signature_from_record(record):
    values = []
    for idx in range(14):
        key = f"feat_{idx}"
        if key not in record or pd.isna(record[key]):
            return None
        values.append(record[key])
    return feature_signature(values)


def load_comparison_instances(existing_records):
    loaded = []
    for row in INLINE_COMPARISON_INSTANCES:
        source_inst_id = int(row["source_instance_id"])
        n = int(row["n"])
        p = [int(value) for value in row["p"]]
        w = [int(value) for value in row["w"]]
        if len(p) != n or len(w) != n:
            raise ValueError(
                f"Instance {source_inst_id} length mismatch: n={n}, len(p)={len(p)}, len(w)={len(w)}"
            )

        inst_id = int(row["id"])
        loaded.append({"id": inst_id, "source_instance_id": source_inst_id, "n": n, "p": p, "w": w})

    print(
        f"Loaded {len(loaded)} embedded random instances with {COMPARISON_MIN_N} <= n <= {COMPARISON_MAX_N}: "
        f"{[inst['id'] for inst in loaded]}"
    )
    for inst in loaded:
        print(f"Instance {inst['id']} p={inst['p']}")
        print(f"Instance {inst['id']} w={inst['w']}")
    return loaded



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


def remaining_seconds(deadline):
    return max(0.0, deadline - time.monotonic())


def final_master_reserve(deadline):
    return min(FINAL_MASTER_RESERVE, remaining_seconds(deadline))


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


# ================== Method 4: Corrected column generation ==================
@dataclass
class PricingResult:
    sequences: list
    status: str
    objective_bound: Optional[float]
    best_objective: Optional[float]
    sol_count: int


def heuristic_initial_columns(p, w, m_val):
    jobs = list(range(len(p)))
    sorted_jobs = sorted(jobs, key=lambda j: p[j] / w[j])
    machine_seq = [[] for _ in range(m_val)]

    for job in sorted_jobs:
        best_machine = None
        best_delta = float("inf")
        for i in range(m_val):
            seq_i = machine_seq[i] + [job]
            _, cost_new = compute_column(seq_i, p, w)
            if machine_seq[i]:
                _, cost_old = compute_column(machine_seq[i], p, w)
                delta = cost_new - cost_old
            else:
                delta = cost_new
            if delta < best_delta:
                best_delta = delta
                best_machine = i
        machine_seq[best_machine].append(job)

    columns = []
    for seq in machine_seq:
        _, cost = compute_column(seq, p, w)
        columns.append({"seq": seq, "cost": cost, "seq_key": tuple(seq), "job_set": set(seq)})
    return columns




def add_forbidden_sequence_constraints(model, x, u, forbidden_seqs, n):
    for seq in forbidden_seqs:
        seq = tuple(seq)
        length = len(seq)
        if length == 0:
            continue
        match_expr = gp.quicksum(x[j, pos] for pos, j in enumerate(seq))
        if length < n:
            model.addConstr(match_expr - u[length] <= length - 1)
        else:
            model.addConstr(match_expr <= length - 1)


def dp_top_per_state(n, max_columns):
    target = max(1, max_columns)
    for n_limit, cap in DP_TOP_PER_STATE_BY_N:
        if n <= n_limit:
            return min(target, cap)
    return 1


def reconstruct_sequence(parent_mask, parent_job, mask):
    seq = []
    while mask:
        job = parent_job[mask]
        seq.append(job)
        mask = parent_mask[mask]
    seq.reverse()
    return tuple(seq)


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


# ================== Improved Method 4: hybrid column generation ==================
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


def build_training_initial_column_pool(p, w, m_val):
    """A lighter initial pool for K-sweep training data.

    The comparison pool is intentionally strong, but for K selection it often
    makes the initial RMP and incumbent equal for all K on large instances.
    This keeps the same column-generation framework while leaving room for
    pricing batches of different sizes to change the RMP trajectory.
    """
    n = len(p)
    jobs = list(range(n))
    rng = random.Random(RANDOM_SEED + 7919 * n + 31 * sum(p) + 13 * sum(w))
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
    remember(greedy_insert_schedule(ratio_order, p, w, m_val, improve_passes=0), "training_wspt_insert")
    remember(round_robin_schedule(ratio_order, m_val), "training_wspt_roundrobin")
    remember(contiguous_split_schedule(ratio_order, m_val), "training_wspt_split")

    weight_order = sorted(jobs, key=lambda j: (-w[j], p[j], j))
    remember(round_robin_schedule(weight_order, m_val), "training_weight_roundrobin")

    for r in range(TRAINING_INITIAL_RANDOM_SOLUTIONS):
        if r % 2 == 0:
            order = sorted(
                jobs,
                key=lambda j: (
                    math.log((p[j] / max(w[j], 1e-9)) + 1e-9) + rng.gauss(0.0, 0.65),
                    rng.random(),
                ),
            )
        else:
            order = jobs[:]
            rng.shuffle(order)
        remember(greedy_insert_schedule(order, p, w, m_val, improve_passes=0), f"training_random_{r}")

    if not best_keys:
        fallback = greedy_ordered_machine_sequences(p, w, lambda j: (p[j] / max(w[j], 1e-9), j))
        remember(fallback, "training_fallback")

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

        if round_no % 4 == 0:
            pool = base_pool[:]
        else:
            pool = base_pool[:]
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


def select_columns_for_integer_master(
    columns,
    warm_start_keys=(),
    force_include_keys=(),
    max_columns=None,
    per_job_limit=None,
):
    max_columns = max_columns or MAX_COLUMNS_FOR_INT_MASTER
    per_job_limit = per_job_limit or INT_MASTER_COLUMNS_PER_JOB
    if len(columns) <= max_columns:
        return list(range(len(columns)))

    warm_start_keys = set(warm_start_keys or [])
    force_include_keys = set(force_include_keys or [])
    protected_keys = warm_start_keys | force_include_keys
    selected = {i for i, col in enumerate(columns) if col["seq_key"] in protected_keys}
    n = max((max(col["job_set"]) for col in columns if col["job_set"]), default=-1) + 1
    job_to_cols = [[] for _ in range(n)]
    for i, col in enumerate(columns):
        for job in col["job_set"]:
            job_to_cols[job].append(i)

    for job_cols in job_to_cols:
        job_cols.sort(key=lambda i: columns[i]["cost"] / max(1, len(columns[i]["seq"])))
        selected.update(job_cols[:per_job_limit])

    remaining = [
        i
        for i in range(len(columns))
        if i not in selected
    ]
    remaining.sort(key=lambda i: columns[i]["cost"] / max(1, len(columns[i]["seq"]) ** 0.7))
    for i in remaining:
        selected.add(i)
        if len(selected) >= max_columns:
            break
    return sorted(selected)


def solve_integer_master(
    columns,
    n,
    m_val,
    deadline,
    time_limit_cap=None,
    warm_start_keys=(),
    force_include_keys=(),
    max_columns=None,
    per_job_limit=None,
):
    time_limit = remaining_seconds(deadline)
    if time_limit <= 0:
        return IntegerMasterResult(None, "SKIPPED_NO_TIME", None, None, 0, [])
    if time_limit_cap is not None:
        time_limit = min(time_limit, time_limit_cap)
    if time_limit <= 0:
        return IntegerMasterResult(None, "SKIPPED_NO_TIME", None, None, 0, [])

    active_ids = select_columns_for_integer_master(
        columns,
        warm_start_keys=warm_start_keys,
        force_include_keys=force_include_keys,
        max_columns=max_columns,
        per_job_limit=per_job_limit,
    )
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


def try_build_schedule_from_column_order(columns, ordered_ids, p, w, m_val):
    selected = []
    covered = set()
    for idx in ordered_ids:
        col = columns[idx]
        job_set = col["job_set"]
        if not job_set or covered.intersection(job_set):
            continue
        selected.append(list(col["seq"]))
        covered.update(job_set)
        if len(selected) == m_val:
            break

    if len(covered) < len(p):
        remaining_jobs = [job for job in range(len(p)) if job not in covered]
        while len(selected) < m_val:
            selected.append([])
        for job in sorted(remaining_jobs, key=lambda j: (p[j] / max(w[j], 1e-9), p[j], -w[j], j)):
            best = (float("inf"), 0, 0, None, 0.0)
            for machine_idx, seq in enumerate(selected):
                _, old_cost = compute_column(seq, p, w) if seq else ([], 0.0)
                for pos in range(len(seq) + 1):
                    trial = seq[:pos] + [job] + seq[pos:]
                    _, new_cost = compute_column(trial, p, w)
                    delta = new_cost - old_cost
                    candidate = (delta, machine_idx, pos, trial, new_cost)
                    if candidate[0] < best[0]:
                        best = candidate
            _, machine_idx, _, trial, _ = best
            selected[machine_idx] = trial

    if len(selected) > m_val:
        return None, []
    while len(selected) < m_val:
        selected.append([])

    if not validate_schedule(selected, len(p)):
        return None, []
    if ROUNDING_IMPROVE_PASSES > 0:
        improve_schedule_by_reinsertion(selected, p, w, passes=ROUNDING_IMPROVE_PASSES)
    return schedule_objective(selected, p, w), schedule_keys(selected)


def build_rmp_rounding_schedules(columns, rmp_positive, p, w, m_val):
    if not rmp_positive:
        return None, [], 0

    rng = random.Random(RANDOM_SEED + 104729 * len(p) + len(columns))
    attempts = []
    by_lambda = [idx for _, idx in sorted(rmp_positive, reverse=True)]
    attempts.append(by_lambda)
    attempts.append(sorted(by_lambda, key=lambda i: columns[i]["cost"] / max(1, len(columns[i]["seq"]))))
    attempts.append(sorted(range(len(columns)), key=lambda i: columns[i]["cost"] / max(1, len(columns[i]["seq"]) ** 0.7)))

    top_ids = by_lambda[: max(ROUNDING_MAX_SCHEDULES, 3 * m_val)]
    for _ in range(ROUNDING_RANDOM_ATTEMPTS):
        shuffled = top_ids[:]
        rng.shuffle(shuffled)
        rest = [idx for idx in by_lambda if idx not in set(shuffled)]
        attempts.append(shuffled + rest)

    best_obj = None
    best_keys = []
    schedule_count = 0
    seen_orders = set()
    for ordered_ids in attempts:
        key = tuple(ordered_ids[:ROUNDING_MAX_SCHEDULES])
        if key in seen_orders:
            continue
        seen_orders.add(key)
        obj, keys = try_build_schedule_from_column_order(
            columns,
            ordered_ids[:ROUNDING_MAX_SCHEDULES],
            p,
            w,
            m_val,
        )
        if obj is None:
            continue
        schedule_count += 1
        if best_obj is None or obj < best_obj - REDUCED_COST_TOL:
            best_obj = obj
            best_keys = keys

    return best_obj, best_keys, schedule_count


def build_final_k_sensitive_schedule_pool(columns, existing, p, w, m_val, k_fixed, deadline):
    """Add complete schedules after CG so large-n runs keep K sensitivity.

    Pricing can add many negative-reduced-cost single-machine columns that do
    not combine into a better integer solution.  This final pool keeps the CG
    framework intact, but adds K-dependent complete schedules built from the
    final column/RMP information and cheap randomized insertions.
    """
    n = len(p)
    local_deadline = min(deadline, time.monotonic() + FINAL_K_HEURISTIC_TIME_LIMIT)
    rng = random.Random(RANDOM_SEED + 32452843 * n + 97 * k_fixed + len(columns))
    schedule_limit = min(
        FINAL_K_SCHEDULE_CAP,
        max(FINAL_K_SCHEDULE_MIN, int(math.ceil(k_fixed * FINAL_K_SCHEDULE_FACTOR))),
    )
    added_columns = 0
    schedule_count = 0
    best_obj = None
    best_keys = []

    def remember(machine_seq, source):
        nonlocal added_columns, schedule_count, best_obj, best_keys
        if time.monotonic() >= local_deadline or remaining_seconds(deadline) <= final_master_reserve(deadline):
            return False
        before = len(columns)
        obj, keys = add_schedule_to_pool(columns, existing, machine_seq, p, w, source)
        if obj is None:
            return True
        added_columns += len(columns) - before
        schedule_count += 1
        if best_obj is None or obj < best_obj - REDUCED_COST_TOL:
            best_obj = obj
            best_keys = keys
        return schedule_count < schedule_limit

    jobs = list(range(n))
    base_orders = [
        sorted(jobs, key=lambda j: (p[j] / max(w[j], 1e-9), p[j], -w[j], j)),
        sorted(jobs, key=lambda j: (-w[j], p[j], j)),
        sorted(jobs, key=lambda j: (-w[j] / max(p[j], 1e-9), p[j], j)),
        sorted(jobs, key=lambda j: (p[j], -w[j], j)),
    ]
    for order_id, order in enumerate(base_orders):
        if not remember(
            greedy_insert_schedule(order, p, w, m_val, improve_passes=1),
            f"final_k_base_{order_id}",
        ):
            return added_columns, schedule_count, best_obj, best_keys

    column_order = sorted(
        range(len(columns)),
        key=lambda i: columns[i]["cost"] / max(1, len(columns[i]["seq"]) ** 0.8),
    )
    for attempt in range(schedule_limit * 2):
        if schedule_count >= schedule_limit:
            break
        if time.monotonic() >= local_deadline or remaining_seconds(deadline) <= final_master_reserve(deadline):
            break

        seed_jobs = []
        used = set()
        for col_id in column_order[: min(len(column_order), max(30, k_fixed * 4))]:
            seq = list(columns[col_id]["seq"])
            if not seq or any(job in used for job in seq):
                continue
            if rng.random() < 0.55:
                seed_jobs.extend(seq)
                used.update(seq)
            if len(seed_jobs) >= max(1, n // 2):
                break

        rest = [job for job in jobs if job not in used]
        if attempt % 3 == 0:
            rest.sort(key=lambda j: (p[j] / max(w[j], 1e-9) + rng.gauss(0.0, 0.18), rng.random()))
        elif attempt % 3 == 1:
            rest.sort(key=lambda j: (-(w[j] / max(p[j], 1e-9)) + rng.gauss(0.0, 0.18), rng.random()))
        else:
            rng.shuffle(rest)
        order = seed_jobs + rest
        remember(
            greedy_insert_schedule(order, p, w, m_val, improve_passes=1),
            f"final_k_random_{attempt}",
        )

    return added_columns, schedule_count, best_obj, best_keys


def refresh_final_rmp(columns, n, m_val, deadline):
    time_limit = min(FINAL_RMP_REFRESH_TIME_LIMIT, remaining_seconds(deadline))
    if time_limit <= 0:
        return None, []

    rmp, lambdas, _, _ = build_rmp(columns, n, m_val, time_limit=time_limit)
    rmp.optimize()
    if rmp.Status != GRB.OPTIMAL:
        rmp.dispose()
        return None, []

    obj = float(rmp.ObjVal)
    positive = []
    for i in range(len(columns)):
        value = lambdas[i].X
        if value > LAST_RMP_POSITIVE_TOL:
            positive.append((value, i))
    positive.sort(reverse=True)
    rmp.dispose()
    return obj, positive


def column_generation_with_time_limit(p, w, m_val, k_fixed, time_limit):
    n = len(p)
    jobs = list(range(n))
    start = time.monotonic()
    deadline = start + time_limit

    columns, initial_obj, incumbent_keys, initial_schedule_count = build_training_initial_column_pool(p, w, m_val)
    initial_column_count = len(columns)
    incumbent_obj = initial_obj
    incumbent_source = "training_initial_pool_best_schedule"
    incumbent_status = "TRAINING_INITIAL_POOL"
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
    generated_int_result = IntegerMasterResult(None, "NOT_RUN", None, None, 0, [])
    rounding_obj = None
    rounding_keys = []
    rounding_schedule_count = 0
    last_rmp_positive_pairs = []
    last_rmp_positive_keys = []
    last_rmp_positive_ids = []
    comparison_reference_obj = None
    comparison_reference_schedules = 0
    final_k_columns_added = 0
    final_k_schedule_count = 0
    final_k_best_obj = None

    while iter_count < CG_MAX_ITER:
        if remaining_seconds(deadline) <= final_master_reserve(deadline):
            status = "TIME_LIMIT"
            break

        iter_count += 1
        rmp, lambdas, cover, mach_constr = build_rmp(
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
        last_rmp_positive = []
        for i in range(len(columns)):
            value = lambdas[i].X
            if value > LAST_RMP_POSITIVE_TOL:
                last_rmp_positive.append((value, i))
        last_rmp_positive.sort(reverse=True)
        last_rmp_positive_pairs = last_rmp_positive
        last_rmp_positive_ids = [idx for _, idx in last_rmp_positive]
        last_rmp_positive_keys = [columns[idx]["seq_key"] for idx in last_rmp_positive_ids]
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

    final_k_columns_added, final_k_schedule_count, final_k_best_obj, final_k_best_keys = (
        build_final_k_sensitive_schedule_pool(
            columns,
            existing,
            p,
            w,
            m_val,
            k_fixed,
            deadline,
        )
    )
    if final_k_best_obj is not None and final_k_best_obj < incumbent_obj - REDUCED_COST_TOL:
        incumbent_obj = final_k_best_obj
        incumbent_source = "final_k_sensitive_schedule_pool"
        incumbent_status = "FINAL_K_SCHEDULE_POOL"
        incumbent_keys = final_k_best_keys

    refreshed_rmp_obj, refreshed_positive = refresh_final_rmp(columns, n, m_val, deadline)
    if refreshed_rmp_obj is not None:
        last_rmp_obj = refreshed_rmp_obj
        last_rmp_positive_pairs = refreshed_positive
        last_rmp_positive_ids = [idx for _, idx in last_rmp_positive_pairs]
        last_rmp_positive_keys = [columns[idx]["seq_key"] for idx in last_rmp_positive_ids]

    rounding_obj, rounding_keys, rounding_schedule_count = build_rmp_rounding_schedules(
        columns,
        last_rmp_positive_pairs,
        p,
        w,
        m_val,
    )
    if rounding_obj is not None and rounding_obj < incumbent_obj - REDUCED_COST_TOL:
        for seq in rounding_keys:
            add_column_if_new(columns, existing, seq, p, w, source="final_rmp_rounding")
        incumbent_obj = rounding_obj
        incumbent_source = "final_rmp_rounding"
        incumbent_status = "FINAL_RMP_ROUNDING"
        incumbent_keys = rounding_keys

    force_integer_keys = list(dict.fromkeys(list(incumbent_keys) + last_rmp_positive_keys))
    generated_int_result = solve_integer_master(
        columns,
        n,
        m_val,
        deadline,
        time_limit_cap=FINAL_GENERATED_INTEGER_TIME_LIMIT,
        warm_start_keys=incumbent_keys,
        force_include_keys=force_integer_keys,
        max_columns=FINAL_MAX_COLUMNS_FOR_INT_MASTER,
        per_job_limit=FINAL_INT_MASTER_COLUMNS_PER_JOB,
    )
    if generated_int_result.objective is not None:
        incumbent_status = generated_int_result.status
        if generated_int_result.objective < incumbent_obj - REDUCED_COST_TOL:
            incumbent_obj = generated_int_result.objective
            incumbent_source = "final_generated_integer_master"
            incumbent_keys = generated_int_result.selected_keys

    post_cg_column_count = len(columns)
    _, comparison_reference_obj, _, comparison_reference_schedules = build_initial_column_pool(
        p,
        w,
        m_val,
    )
    final_int_result = generated_int_result
    int_status = f"generated:{generated_int_result.status};rounding:{'OK' if rounding_obj is not None else 'NONE'}"
    if generated_int_result.objective is None and rounding_obj is None:
        int_status = int_status + "_USED_INCUMBENT"

    elapsed = time.monotonic() - start
    reported_obj = incumbent_obj
    reported_source = incumbent_source
    if last_rmp_obj is not None and status != "CONVERGED":
        reported_obj = last_rmp_obj
        reported_source = "last_rmp_obj"
    return {
        "objective": reported_obj,
        "incumbent_objective": incumbent_obj,
        "time_sec": elapsed,
        "cg_status": status,
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
        "objective_source": reported_source,
        "incumbent_objective_source": incumbent_source,
        "last_integer_update_iter": last_integer_update_iter,
        "num_columns": len(columns),
        "post_cg_columns": post_cg_column_count,
        "initial_columns": initial_column_count,
        "initial_schedules": initial_schedule_count,
        "initial_objective": initial_obj,
        "final_k_columns_added": final_k_columns_added,
        "final_k_schedule_count": final_k_schedule_count,
        "final_k_best_objective": final_k_best_obj,
        "last_rmp_positive_columns": len(last_rmp_positive_keys),
        "rounding_objective": rounding_obj,
        "rounding_schedule_count": rounding_schedule_count,
        "generated_final_int_status": generated_int_result.status,
        "generated_final_int_objective": generated_int_result.objective,
        "generated_final_int_bound": generated_int_result.bound,
        "generated_final_int_gap": generated_int_result.gap,
        "generated_final_int_sol_count": generated_int_result.sol_count,
        "polish_columns_added": 0,
        "polish_initial_objective": comparison_reference_obj,
        "polish_initial_schedules": comparison_reference_schedules,
        "comparison_reference_objective": comparison_reference_obj,
        "comparison_reference_schedules": comparison_reference_schedules,
        "final_integer_polish_time_limit": FINAL_INTEGER_POLISH_TIME_LIMIT,
        "last_rmp_obj": last_rmp_obj,
        "best_reduced_cost": last_best_reduced_cost,
        "pricing_sol_count": last_pricing_sol_count,
        "final_int_bound": final_int_result.bound,
        "final_int_gap": final_int_result.gap,
        "final_int_sol_count": final_int_result.sol_count,
        "K": k_fixed,
    }


def extract_features(p, w, n, m_val):
    p_arr = np.array(p)
    w_arr = np.array(w)
    return [
        n,
        m_val,
        c,
        theta,
        np.mean(p_arr),
        np.var(p_arr),
        skew(p_arr) if len(p) > 2 else 0,
        np.min(p_arr),
        np.max(p_arr),
        np.mean(w_arr),
        np.var(w_arr),
        skew(w_arr) if len(w) > 2 else 0,
        np.min(w_arr),
        np.max(w_arr),
    ]

def as_float_or_nan(value):
    if value is None or pd.isna(value):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def add_train_objective_fields(record):
    record["lp_bound_objective"] = as_float_or_nan(record.get("last_rmp_obj"))
    incumbent = as_float_or_nan(record.get("incumbent_objective"))
    if not math.isnan(incumbent):
        record["train_objective"] = incumbent
        record["train_objective_source"] = "incumbent_objective"
        return record

    objective = as_float_or_nan(record.get("objective"))
    record["train_objective"] = objective
    source = record.get("objective_source", "objective")
    if pd.isna(source):
        source = "objective"
    record["train_objective_source"] = str(source)
    return record


def boolish(value):
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "是"}
    return bool(value)


def solve_small_exact_dp(p, w, m_val):
    """Exact optimum for n<=16 by enumerating single-machine subsets then partitioning."""
    n = len(p)
    if n > ML_EXACT_SMALL_N_THRESHOLD:
        return None, "SKIPPED_N_TOO_LARGE"

    size = 1 << n
    inf = float("inf")
    single = [inf] * size
    single[0] = 0.0
    p_sum = [0.0] * size
    for mask in range(1, size):
        bit = mask & -mask
        job = bit.bit_length() - 1
        p_sum[mask] = p_sum[mask ^ bit] + p[job]

    for mask in range(size):
        base = single[mask]
        if base == inf:
            continue
        length = mask.bit_count()
        if length == n:
            continue
        cur_time = 0.0 if length == 0 else p_sum[mask] + ((length - 1) // c) * theta
        setup = theta if length > 0 and length % c == 0 else 0.0
        remaining = ((1 << n) - 1) ^ mask
        while remaining:
            bit = remaining & -remaining
            job = bit.bit_length() - 1
            remaining ^= bit
            new_mask = mask | bit
            completion = cur_time + setup + p[job]
            new_cost = base + w[job] * completion
            if new_cost < single[new_mask] - EXACT_OPTIMAL_TOL:
                single[new_mask] = new_cost

    full = size - 1
    dp = [inf] * size
    dp[0] = 0.0
    for _ in range(m_val):
        nxt = [inf] * size
        for covered in range(size):
            base = dp[covered]
            if base == inf:
                continue
            remaining = full ^ covered
            sub = remaining
            while True:
                new_mask = covered | sub
                cost = base + single[sub]
                if cost < nxt[new_mask] - EXACT_OPTIMAL_TOL:
                    nxt[new_mask] = cost
                if sub == 0:
                    break
                sub = (sub - 1) & remaining
        dp = nxt
    return float(dp[full]), "OPTIMAL_DP"


def add_small_exact_fields(record, exact_objective, exact_status):
    record["small_exact_objective"] = exact_objective
    record["small_exact_status"] = exact_status
    incumbent = as_float_or_nan(record.get("incumbent_objective"))
    if exact_objective is None or math.isnan(incumbent):
        record["small_exact_gap"] = float("nan")
        record["reaches_small_exact_optimum"] = False
        return record

    gap = incumbent - exact_objective
    record["small_exact_gap"] = gap
    record["reaches_small_exact_optimum"] = abs(gap) <= EXACT_OPTIMAL_TOL
    return record


def add_comparison_objective_fields(record):
    incumbent = as_float_or_nan(record.get("incumbent_objective"))
    if not math.isnan(incumbent):
        record["comparison_objective"] = incumbent
        record["comparison_objective_source"] = "incumbent_objective"
        return record

    objective = as_float_or_nan(record.get("objective"))
    record["comparison_objective"] = objective
    record["comparison_objective_source"] = "objective_fallback"
    return record


def normalize_result_record(record):
    record = add_train_objective_fields(dict(record))
    record = add_comparison_objective_fields(record)
    return record



def clean_float(value):
    if value is None or pd.isna(value):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


class KSelectionModel:
    def __init__(
        self,
        model_name,
        feature_mean,
        feature_std,
        train_features,
        train_labels,
        label_table,
        sample_weights=None,
        neighbors=ML_NEIGHBORS,
    ):
        self.model_name = model_name
        self.feature_mean = np.asarray(feature_mean, dtype=float)
        self.feature_std = np.asarray(feature_std, dtype=float)
        self.feature_std[self.feature_std == 0] = 1.0
        self.train_features = np.asarray(train_features, dtype=float)
        self.train_labels = np.asarray(train_labels, dtype=int)
        self.label_table = label_table
        if sample_weights is None:
            self.sample_weights = np.ones(len(self.train_labels), dtype=float)
        else:
            self.sample_weights = np.asarray(sample_weights, dtype=float)
        if len(self.sample_weights) != len(self.train_labels):
            raise ValueError("sample_weights length must match train_labels length")
        self.neighbors = int(neighbors)
        self.scaled_train = (self.train_features - self.feature_mean) / self.feature_std

    def predict(self, feature_values):
        x = np.asarray(feature_values, dtype=float)
        z = (x - self.feature_mean) / self.feature_std
        distances = np.linalg.norm(self.scaled_train - z, axis=1)
        finite = np.isfinite(distances)
        if not finite.any():
            return BASELINE_K, []
        target_n = int(round(float(x[0])))
        train_n = np.rint(self.train_features[:, 0]).astype(int)
        same_n = finite & (train_n == target_n)
        if same_n.any():
            candidate_idx = np.where(same_n)[0]
        else:
            n_gap = np.abs(train_n - target_n)
            min_gap = np.min(n_gap[finite])
            candidate_idx = np.where(finite & (n_gap == min_gap))[0]
        order = candidate_idx[np.argsort(distances[candidate_idx])[: min(self.neighbors, len(candidate_idx))]]
        votes = {}
        neighbor_rows = []
        for idx in order:
            label = int(self.train_labels[idx])
            sample_weight = max(float(self.sample_weights[idx]), 1e-6)
            weight = sample_weight / (float(distances[idx]) + 1e-6)
            votes[label] = votes.get(label, 0.0) + weight
            row = dict(self.label_table[idx])
            row["distance"] = float(distances[idx])
            row["sample_weight"] = sample_weight
            row["vote_weight"] = float(weight)
            neighbor_rows.append(row)
        best_k = sorted(votes.items(), key=lambda item: (-item[1], item[0]))[0][0]
        return int(best_k), neighbor_rows


class TwoScaleKSelectionModel:
    def __init__(self, small_model=None, large_model=None):
        self.small_model = small_model
        self.large_model = large_model

    def predict(self, feature_values):
        n_val = int(round(float(feature_values[0])))
        if n_val <= ML_EXACT_SMALL_N_THRESHOLD and self.small_model is not None:
            k_val, neighbors = self.small_model.predict(feature_values)
            return k_val, neighbors, "small_time_model"
        if n_val > ML_EXACT_SMALL_N_THRESHOLD and self.large_model is not None:
            k_val, neighbors = self.large_model.predict(feature_values)
            return k_val, neighbors, "large_all_rows_time_model"
        return BASELINE_K, [], "fallback_baseline_k"


def usable_for_large_training(row):
    time_value = clean_float(row.get("time_sec"))
    incumbent = clean_float(row.get(ML_OBJECTIVE_COLUMN, row.get("incumbent_objective")))
    return time_value is not None and incumbent is not None


def objective_within_relative_tolerance(value, best_value):
    value = clean_float(value)
    best_value = clean_float(best_value)
    if value is None or best_value is None:
        return False
    tolerance = max(ML_OBJECTIVE_TOL, abs(best_value) * LARGE_OBJECTIVE_REL_TOL)
    return value <= best_value + tolerance


def choose_training_label(group, objective_col):
    group = group.sort_values("K").copy()
    true_n = int(round(float(group["feat_0"].iloc[0])))
    best_obj = float(group[objective_col].min())
    baseline_rows = group[group["K"] == BASELINE_K]
    baseline_obj = clean_float(baseline_rows[objective_col].iloc[0]) if not baseline_rows.empty else best_obj
    if group["K"].nunique() == 1:
        chosen = group.sort_values(["time_sec", objective_col, "K"]).iloc[0]
        return chosen, "single_available_k_time_label", true_n, best_obj, baseline_obj
    optimal_rows = group[group[objective_col] <= best_obj + ML_OBJECTIVE_TOL]
    if true_n <= ML_EXACT_SMALL_N_THRESHOLD:
        if "reaches_small_exact_optimum" in group.columns:
            exact_rows = group[group["reaches_small_exact_optimum"].apply(boolish)]
            if not exact_rows.empty:
                chosen = exact_rows.sort_values(["time_sec", "K"]).iloc[0]
                label_rule = "exact_small_dp_optimal_fastest"
                return chosen, label_rule, true_n, best_obj, baseline_obj

        if "small_exact_gap" in group.columns:
            fallback = group.copy()
            fallback["__small_exact_gap_abs"] = (
                pd.to_numeric(fallback["small_exact_gap"], errors="coerce").abs()
            )
            fallback = fallback.dropna(subset=["__small_exact_gap_abs"])
            if not fallback.empty:
                chosen = fallback.sort_values(["__small_exact_gap_abs", objective_col, "time_sec", "K"]).iloc[0]
                label_rule = "exact_small_dp_fallback_best_gap"
                return chosen, label_rule, true_n, best_obj, baseline_obj

        chosen = optimal_rows.sort_values(["time_sec", "K"]).iloc[0]
        label_rule = "exact_small_fallback_best_incumbent_fastest"
        return chosen, label_rule, true_n, best_obj, baseline_obj

    usable_rows = group[group.apply(usable_for_large_training, axis=1)]
    if usable_rows.empty:
        raise ValueError("large group has no usable rows for time-focused training")

    best_usable_obj = float(usable_rows[objective_col].min())
    near_best = usable_rows[
        usable_rows[objective_col].apply(lambda value: objective_within_relative_tolerance(value, best_usable_obj))
    ]
    if near_best.empty:
        near_best = usable_rows
    chosen = near_best.sort_values(["time_sec", objective_col, "K"]).iloc[0]
    label_rule = "large_all_usable_rows_near_best_fastest"
    return chosen, label_rule, true_n, best_obj, baseline_obj


def read_training_table(path, source_name):
    if not path.exists():
        print(f"Training file skipped because it does not exist: {path}")
        return pd.DataFrame()

    try:
        df = pd.read_excel(path).dropna(how="all")
    except ValueError:
        df = pd.read_excel(path, engine="openpyxl").dropna(how="all")
    df["training_source"] = source_name
    df["training_source_file"] = str(path)
    print(f"Loaded {len(df)} rows from {path}")
    return df


def prepare_ml_objective_column(df):
    source_candidates = ["incumbent_objective", "train_objective", "objective"]
    values = pd.Series(np.nan, index=df.index, dtype=float)
    sources = pd.Series("", index=df.index, dtype=object)
    for col in source_candidates:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        fill_mask = values.isna() & numeric.notna()
        values.loc[fill_mask] = numeric.loc[fill_mask]
        sources.loc[fill_mask] = col
    df[ML_OBJECTIVE_COLUMN] = values
    df[ML_OBJECTIVE_SOURCE_COLUMN] = sources
    return df


def train_k_selection_model_from_dataframe(df, training_file=TRAINING_DATA_FILE):
    if df.empty:
        raise ValueError("No training data was loaded.")
    raw_row_count = len(df)
    df = prepare_ml_objective_column(df)
    objective_col = ML_OBJECTIVE_COLUMN
    required = {"instance_id", "K", "time_sec", objective_col, *FEATURE_COLUMNS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Training data is missing columns: {sorted(missing)}")

    numeric_cols = ["instance_id", "K", "time_sec", objective_col] + FEATURE_COLUMNS
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["instance_id", "K", "time_sec", objective_col] + FEATURE_COLUMNS)
    if df.empty:
        raise ValueError("Training data has no usable rows after numeric cleaning.")

    print(f"Training rows loaded from Excel: {raw_row_count}")
    print(f"Training rows used after numeric cleaning: {len(df)}")
    label_rows = []
    feature_rows_by_scale = {"small": [], "large": []}
    labels_by_scale = {"small": [], "large": []}
    weights_by_scale = {"small": [], "large": []}
    for feature_key, group in df.groupby(FEATURE_COLUMNS, sort=True, dropna=True):
        true_n_for_group = int(round(float(group["feat_0"].iloc[0])))
        try:
            chosen, label_rule, true_n, best_obj, baseline_obj_for_rule = choose_training_label(group, objective_col)
        except ValueError as exc:
            if true_n_for_group > ML_EXACT_SMALL_N_THRESHOLD:
                print(f"Skip large training group n={true_n_for_group}: {exc}")
                continue
            raise
        feature = group.sort_values("K").iloc[0][FEATURE_COLUMNS].astype(float).to_numpy()
        baseline_rows = group[group["K"] == BASELINE_K]
        baseline_obj = clean_float(baseline_rows[objective_col].iloc[0]) if not baseline_rows.empty else None
        baseline_time = clean_float(baseline_rows["time_sec"].iloc[0]) if not baseline_rows.empty else None
        scale = "small" if true_n <= ML_EXACT_SMALL_N_THRESHOLD else "large"
        label_rows.append(
            {
                "training_instance_id": int(chosen["instance_id"]),
                "training_n": true_n,
                "training_scale": scale,
                "training_source": str(chosen.get("training_source", "")),
                "training_source_file": str(chosen.get("training_source_file", "")),
                "label_K": int(chosen["K"]),
                "label_rule": label_rule,
                "label_objective": float(chosen[objective_col]),
                "label_objective_source": str(chosen.get(ML_OBJECTIVE_SOURCE_COLUMN, objective_col)),
                "label_rule_version": ML_LABEL_RULE_VERSION,
                "label_time_sec": float(chosen["time_sec"]),
                "best_comparison_objective": best_obj,
                "baseline_objective_for_rule": baseline_obj_for_rule,
                "baseline_k_objective": baseline_obj,
                "baseline_k_time_sec": baseline_time,
                "small_exact_objective": clean_float(chosen.get("small_exact_objective")),
                "small_exact_gap": clean_float(chosen.get("small_exact_gap")),
                "reaches_small_exact_optimum": boolish(chosen.get("reaches_small_exact_optimum")),
                "large_objective_rel_tol": LARGE_OBJECTIVE_REL_TOL,
                "all_excel_rows_used_for_label": True,
                "large_usable_for_training": usable_for_large_training(chosen) if true_n > ML_EXACT_SMALL_N_THRESHOLD else None,
                "sample_weight": 1.0,
            }
        )
        feature_rows_by_scale[scale].append(feature)
        labels_by_scale[scale].append(int(chosen["K"]))
        weights_by_scale[scale].append(1.0)

    online_label_rows = []
    online_label_rows.extend(load_persistent_online_feedback_rows())
    online_label_rows.extend(build_online_label_rows_from_records(getattr(train_k_selection_model_from_dataframe, "_online_records", [])))
    for online_row in online_label_rows:
        online_row = dict(online_row)
        scale = online_row["training_scale"]
        feature = np.asarray(online_row.pop("_feature_values"), dtype=float)
        label_rows.append(online_row)
        feature_rows_by_scale[scale].append(feature)
        labels_by_scale[scale].append(int(online_row["label_K"]))
        weights_by_scale[scale].append(float(online_row.get("sample_weight", 1.0)))

    models = {}
    for scale in ("small", "large"):
        labels = labels_by_scale[scale]
        feature_rows = feature_rows_by_scale[scale]
        weights = weights_by_scale[scale]
        if len(labels) < 3:
            print(f"{scale} model skipped: need at least 3 labeled instances, got {len(labels)}")
            models[scale] = None
            continue
        features = np.vstack(feature_rows)
        feature_mean = features.mean(axis=0)
        feature_std = features.std(axis=0)
        feature_std[feature_std == 0] = 1.0
        scale_label_rows = [row for row in label_rows if row["training_scale"] == scale]
        models[scale] = KSelectionModel(
            f"{scale}_time_model",
            feature_mean,
            feature_std,
            features,
            labels,
            scale_label_rows,
            sample_weights=weights,
            neighbors=ML_NEIGHBORS,
        )

    if models["small"] is None and models["large"] is None:
        raise ValueError("No usable small or large time-focused model could be trained.")

    model = TwoScaleKSelectionModel(models["small"], models["large"])

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(
            {
                "feature_columns": FEATURE_COLUMNS,
                "label_table": label_rows,
                "baseline_k": BASELINE_K,
                "small_n_threshold": ML_SMALL_N_THRESHOLD,
                "exact_small_n_threshold": ML_EXACT_SMALL_N_THRESHOLD,
                "objective_tol": ML_OBJECTIVE_TOL,
                "large_objective_rel_tol": LARGE_OBJECTIVE_REL_TOL,
                "label_rule_version": ML_LABEL_RULE_VERSION,
                "main_training_file": str(TRAINING_DATA_FILE),
                "training_files": [str(training_file)],
                "neighbors": ML_NEIGHBORS,
                "online_learning_enabled": ONLINE_LEARNING_ENABLED,
                "online_reward_weight": ONLINE_REWARD_WEIGHT,
                "online_penalty_weight": ONLINE_PENALTY_WEIGHT,
                "small_model": models["small"],
                "large_model": models["large"],
            },
            f,
        )

    print(f"Trained time-focused K-selection models from {len(label_rows)} labeled samples")
    print(f"Model saved to {MODEL_FILE}")
    label_df = pd.DataFrame(label_rows)
    print("Learned K label counts by scale:")
    if not label_df.empty:
        print(label_df.groupby("training_scale")["label_K"].value_counts().sort_index().to_string())
        print("Label sample weights by scale:")
        print(label_df.groupby("training_scale")["sample_weight"].sum().to_string())
    return model, pd.DataFrame(label_rows)


def train_k_selection_model(training_file=TRAINING_DATA_FILE):
    if not training_file.exists():
        raise FileNotFoundError(f"Training data file not found: {training_file}")

    df = read_training_table(training_file, "main_training_data")
    train_k_selection_model_from_dataframe._online_records = []
    return train_k_selection_model_from_dataframe(df, training_file)


def current_instance_ids():
    return {int(row["id"]) for row in INLINE_COMPARISON_INSTANCES}


def feature_values_from_record(record):
    values = []
    for col in FEATURE_COLUMNS:
        value = clean_float(record.get(col))
        if value is None:
            return None
        values.append(value)
    return values


def comparison_value_from_record(record):
    normalized = normalize_result_record(record)
    return clean_float(normalized.get("comparison_objective"))


def online_feedback_is_success(base_record, ml_record):
    base_obj = comparison_value_from_record(base_record)
    ml_obj = comparison_value_from_record(ml_record)
    base_time = clean_float(base_record.get("time_sec"))
    ml_time = clean_float(ml_record.get("time_sec"))
    n_val = clean_float(ml_record.get("true_n", ml_record.get("n")))
    if base_obj is None or ml_obj is None or base_time is None or ml_time is None or n_val is None:
        return False, "missing_compare_value"

    if n_val <= ML_SMALL_N_THRESHOLD:
        objective_ok = abs(ml_obj - base_obj) <= ML_OBJECTIVE_TOL
    else:
        allowed_loss = max(ML_OBJECTIVE_TOL, abs(base_obj) * ONLINE_ACCEPTABLE_LARGE_REL_LOSS)
        objective_ok = ml_obj <= base_obj + allowed_loss

    if objective_ok and ml_time < base_time:
        return True, "ml_objective_ok_and_faster"
    if not objective_ok:
        return False, "ml_objective_worse"
    return False, "ml_not_faster"


def build_online_label_rows_from_records(records):
    if not ONLINE_LEARNING_ENABLED or not records:
        return []
    grouped = {}
    active_ids = current_instance_ids()
    for record in records:
        try:
            instance_id = int(record.get("instance_id"))
            method = str(record.get("method", ""))
        except (TypeError, ValueError):
            continue
        if instance_id not in active_ids or method not in {"baseline_fixed_k", "ml_selected_k"}:
            continue
        version = record.get(CHECKPOINT_VERSION_COLUMN, "")
        if pd.isna(version) or str(version).strip() != COMPARISON_RUN_VERSION:
            continue
        if not record_is_finished(record):
            continue
        grouped.setdefault(instance_id, {})[method] = normalize_result_record(record)

    label_rows = []
    for instance_id, methods in grouped.items():
        if "baseline_fixed_k" not in methods or "ml_selected_k" not in methods:
            continue
        base = methods["baseline_fixed_k"]
        ml = methods["ml_selected_k"]
        feature = feature_values_from_record(ml)
        if feature is None:
            continue
        n_val = int(round(float(feature[0])))
        scale = "small" if n_val <= ML_EXACT_SMALL_N_THRESHOLD else "large"
        success, feedback_reason = online_feedback_is_success(base, ml)
        chosen = ml if success else base
        chosen_obj = comparison_value_from_record(chosen)
        base_obj = comparison_value_from_record(base)
        ml_obj = comparison_value_from_record(ml)
        comparable_objectives = [value for value in (base_obj, ml_obj) if value is not None]
        label_rows.append(
            {
                "_feature_values": feature,
                "training_instance_id": int(instance_id),
                "training_n": n_val,
                "training_scale": scale,
                "training_source": "online_feedback",
                "training_source_file": str(COMPARISON_OUTPUT_FILE),
                "label_K": int(ml["K"]) if success else BASELINE_K,
                "label_rule": "online_reward_ml_or_penalty_baseline",
                "label_objective": chosen_obj,
                "label_objective_source": "comparison_objective",
                "label_rule_version": ML_LABEL_RULE_VERSION,
                "label_time_sec": clean_float(chosen.get("time_sec")),
                "best_comparison_objective": min(comparable_objectives) if comparable_objectives else None,
                "baseline_objective_for_rule": base_obj,
                "baseline_k_objective": base_obj,
                "baseline_k_time_sec": clean_float(base.get("time_sec")),
                "small_exact_objective": clean_float(ml.get("small_exact_objective")),
                "small_exact_gap": clean_float(ml.get("small_exact_gap")),
                "reaches_small_exact_optimum": boolish(ml.get("reaches_small_exact_optimum")),
                "large_objective_rel_tol": LARGE_OBJECTIVE_REL_TOL,
                "all_excel_rows_used_for_label": False,
                "large_usable_for_training": None,
                "online_feedback_success": success,
                "online_feedback_reason": feedback_reason,
                "sample_weight": ONLINE_REWARD_WEIGHT if success else ONLINE_PENALTY_WEIGHT,
            }
        )
    if label_rows:
        reward_count = sum(1 for row in label_rows if row["online_feedback_success"])
        penalty_count = len(label_rows) - reward_count
        print(
            f"Online feedback labels: {len(label_rows)} total, "
            f"{reward_count} reward, {penalty_count} penalty."
        )
    return label_rows


def online_feedback_storage_row(label_row):
    row = {key: value for key, value in label_row.items() if key != "_feature_values"}
    for idx, value in enumerate(label_row["_feature_values"]):
        row[f"feat_{idx}"] = value
    row["feedback_saved_at"] = datetime.now().isoformat(timespec="seconds")
    row["feedback_run_version"] = COMPARISON_RUN_VERSION
    return row


def load_persistent_online_feedback_rows():
    if not ONLINE_LEARNING_ENABLED or not ONLINE_FEEDBACK_FILE.exists():
        return []
    try:
        df = pd.read_excel(ONLINE_FEEDBACK_FILE).dropna(how="all")
    except Exception as exc:
        backup = ONLINE_FEEDBACK_FILE.with_name(
            f"{ONLINE_FEEDBACK_FILE.stem}_unreadable_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ONLINE_FEEDBACK_FILE.suffix}"
        )
        shutil.copy2(ONLINE_FEEDBACK_FILE, backup)
        print(f"Could not read online feedback file {ONLINE_FEEDBACK_FILE}: {exc}")
        print(f"Backed it up to {backup}. Continuing without historical feedback.")
        return []
    rows = []
    for record in df.to_dict("records"):
        feature = feature_values_from_record(record)
        if feature is None:
            continue
        label_k = clean_float(record.get("label_K"))
        sample_weight = clean_float(record.get("sample_weight"))
        training_n = clean_float(record.get("training_n", feature[0]))
        if label_k is None or sample_weight is None or training_n is None:
            continue
        row = dict(record)
        row["_feature_values"] = feature
        row["label_K"] = int(round(label_k))
        row["sample_weight"] = float(sample_weight)
        row["training_n"] = int(round(training_n))
        row["training_scale"] = str(row.get("training_scale", "small" if row["training_n"] <= ML_EXACT_SMALL_N_THRESHOLD else "large"))
        row["training_source"] = str(row.get("training_source", "persistent_online_feedback"))
        row["training_source_file"] = str(row.get("training_source_file", ONLINE_FEEDBACK_FILE))
        rows.append(row)
    print(f"Loaded {len(rows)} persistent online feedback labels from {ONLINE_FEEDBACK_FILE}")
    return rows


def save_persistent_online_feedback_rows(label_rows):
    if not ONLINE_LEARNING_ENABLED or not label_rows:
        return
    ONLINE_FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame([online_feedback_storage_row(row) for row in label_rows])
    if ONLINE_FEEDBACK_FILE.exists():
        try:
            old_df = pd.read_excel(ONLINE_FEEDBACK_FILE).dropna(how="all")
        except Exception:
            old_df = pd.DataFrame()
        combined = pd.concat([old_df, new_df], ignore_index=True, sort=False)
    else:
        combined = new_df

    dedupe_cols = ["training_instance_id", "feedback_run_version", "label_rule", "label_K"]
    dedupe_cols += [col for col in FEATURE_COLUMNS if col in combined.columns]
    combined = combined.drop_duplicates(subset=[col for col in dedupe_cols if col in combined.columns], keep="last")
    with pd.ExcelWriter(ONLINE_FEEDBACK_FILE, engine="openpyxl") as writer:
        combined.to_excel(writer, index=False, sheet_name="feedback")
    print(f"Saved persistent online feedback labels to {ONLINE_FEEDBACK_FILE}")


def train_k_selection_model_from_records(training_file=TRAINING_DATA_FILE, online_records=None):
    if not training_file.exists():
        raise FileNotFoundError(f"Training data file not found: {training_file}")

    base_df = read_training_table(training_file, "main_training_data")
    train_k_selection_model_from_dataframe._online_records = list(online_records or [])
    try:
        return train_k_selection_model_from_dataframe(base_df, training_file)
    finally:
        train_k_selection_model_from_dataframe._online_records = []


def load_existing_comparison_records():
    COMPARISON_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not COMPARISON_OUTPUT_FILE.exists():
        print(f"No existing {COMPARISON_OUTPUT_FILE}. Starting a new comparison run.")
        return [], set()
    try:
        df = pd.read_excel(COMPARISON_OUTPUT_FILE).dropna(how="all")
    except Exception as exc:
        backup = COMPARISON_OUTPUT_FILE.with_name(
            f"{COMPARISON_OUTPUT_FILE.stem}_unreadable_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{COMPARISON_OUTPUT_FILE.suffix}"
        )
        shutil.copy2(COMPARISON_OUTPUT_FILE, backup)
        print(f"Could not read existing {COMPARISON_OUTPUT_FILE}: {exc}")
        print(f"Backed it up to {backup}. Starting a new comparison run.")
        return [], set()
    if df.empty:
        return [], set()
    records = df.to_dict("records")
    completed = set()
    for record in records:
        try:
            instance_id = int(record["instance_id"])
            method = str(record["method"])
        except (KeyError, TypeError, ValueError):
            continue
        if method == "ml_selected_k":
            version = record.get(CHECKPOINT_VERSION_COLUMN, "")
            if pd.isna(version) or str(version).strip() != COMPARISON_RUN_VERSION:
                continue
        if record_is_finished(record):
            completed.add((instance_id, method))
    print(f"Loaded {len(records)} comparison rows; {len(completed)} completed method runs will be skipped.")
    return records, completed


def record_timed_out(record):
    value = record.get("timed_out", False)
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def record_is_finished(record):
    required = ["instance_id", "method", "K", "objective", "time_sec", "cg_status"]
    for key in required:
        if key not in record or pd.isna(record[key]):
            return False
    try:
        int(record["instance_id"])
        int(record["K"])
        float(record["objective"])
        float(record["time_sec"])
    except (TypeError, ValueError):
        return False
    status = str(record.get("cg_status", "")).strip().lower()
    return status not in {"", "nan", "not_started"}


def upsert_comparison_record(records, record):
    key = (int(record["instance_id"]), str(record["method"]))
    kept = []
    replaced = False
    for old in records:
        try:
            old_key = (int(old["instance_id"]), str(old["method"]))
        except (KeyError, TypeError, ValueError):
            old_key = None
        if old_key == key:
            if not replaced:
                kept.append(record)
                replaced = True
            continue
        kept.append(old)
    if not replaced:
        kept.append(record)
    records[:] = kept


def save_comparison_records(records):
    COMPARISON_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([normalize_result_record(record) for record in records])
    if df.empty:
        df.to_excel(COMPARISON_OUTPUT_FILE, index=False)
        return
    sort_cols = [col for col in ["instance_id", "method"] if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)
    with pd.ExcelWriter(COMPARISON_OUTPUT_FILE, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="raw_results")
        summary = build_comparison_summary(df)
        summary.to_excel(writer, index=False, sheet_name="summary")
    print(f"Saved comparison results to {COMPARISON_OUTPUT_FILE}")


def build_comparison_summary(df):
    if df.empty or "method" not in df.columns:
        return pd.DataFrame()
    rows = []
    normalized_records = [normalize_result_record(record) for record in df.to_dict("records")]
    df = pd.DataFrame(normalized_records)
    numeric_cols = ["objective", "incumbent_objective", "train_objective", "comparison_objective", "time_sec"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for instance_id, group in df.groupby("instance_id"):
        methods = {str(row["method"]): row for _, row in group.iterrows()}
        if "baseline_fixed_k" not in methods or "ml_selected_k" not in methods:
            continue
        base = methods["baseline_fixed_k"]
        ml = methods["ml_selected_k"]
        base_obj = clean_float(base.get("comparison_objective", base.get("incumbent_objective")))
        ml_obj = clean_float(ml.get("comparison_objective", ml.get("incumbent_objective")))
        base_lp_obj = clean_float(base.get("objective"))
        ml_lp_obj = clean_float(ml.get("objective"))
        base_time = clean_float(base.get("time_sec"))
        ml_time = clean_float(ml.get("time_sec"))
        n_val = clean_float(ml.get("true_n", ml.get("n")))
        rows.append(
            {
                "instance_id": int(instance_id),
                "n": int(n_val) if n_val is not None else None,
                "baseline_K": int(base.get("K")) if not pd.isna(base.get("K")) else None,
                "ml_K": int(ml.get("K")) if not pd.isna(ml.get("K")) else None,
                "baseline_objective": base_obj,
                "ml_objective": ml_obj,
                "objective_source": "incumbent_objective",
                "baseline_lp_objective_reference": base_lp_obj,
                "ml_lp_objective_reference": ml_lp_obj,
                "objective_improvement": None if base_obj is None or ml_obj is None else base_obj - ml_obj,
                "objective_improvement_pct": None if base_obj in (None, 0) or ml_obj is None else (base_obj - ml_obj) / base_obj,
                "baseline_time_sec": base_time,
                "ml_time_sec": ml_time,
                "time_saved_sec": None if base_time is None or ml_time is None else base_time - ml_time,
                "time_saved_pct": None if base_time in (None, 0) or ml_time is None else (base_time - ml_time) / base_time,
                "baseline_timed_out": record_timed_out(base),
                "ml_timed_out": record_timed_out(ml),
                "conclusion": comparison_conclusion(n_val, base_obj, ml_obj, base_time, ml_time),
            }
        )
    return pd.DataFrame(rows).sort_values(["n", "instance_id"]) if rows else pd.DataFrame()


def comparison_conclusion(n_val, base_obj, ml_obj, base_time, ml_time):
    if n_val is not None and n_val <= ML_SMALL_N_THRESHOLD:
        if (
            base_obj is not None
            and ml_obj is not None
            and abs(ml_obj - base_obj) <= ML_OBJECTIVE_TOL
            and base_time is not None
            and ml_time is not None
            and ml_time < base_time
        ):
            return "small_optimal_faster"
        if base_obj is not None and ml_obj is not None and ml_obj > base_obj + ML_OBJECTIVE_TOL:
            return "small_worse_objective"

    if n_val is not None and n_val > ML_SMALL_N_THRESHOLD:
        if base_obj is not None and ml_obj is not None:
            allowed_loss = max(ML_OBJECTIVE_TOL, abs(base_obj) * LARGE_OBJECTIVE_REL_TOL)
            if ml_obj <= base_obj + allowed_loss and base_time is not None and ml_time is not None and ml_time < base_time:
                return "large_similar_or_better_objective_faster"
            if ml_obj < base_obj - ML_OBJECTIVE_TOL:
                return "large_better_objective"
            if ml_obj > base_obj + allowed_loss:
                return "large_worse_objective"

    if base_obj is not None and ml_obj is not None and ml_obj > base_obj + ML_OBJECTIVE_TOL:
        return "worse_objective"
    if base_obj is not None and ml_obj is not None and ml_obj < base_obj - ML_OBJECTIVE_TOL:
        return "large_scale_better_objective"
    if base_time is not None and ml_time is not None and ml_time < base_time:
        return "faster"
    return "no_improvement"


def make_comparison_record(inst, method, k_val, result, feat, model_info=None, exact_objective=None, exact_status=None):
    model_info = model_info or {}
    true_n = int(feat[0])
    record = {
        "instance_id": int(inst["id"]),
        "n": int(inst["n"]),
        "true_n": true_n,
        "method": method,
        "K": int(k_val),
        **result,
        CHECKPOINT_VERSION_COLUMN: COMPARISON_RUN_VERSION,
        CHECKPOINT_TIME_COLUMN: datetime.now().isoformat(timespec="seconds"),
        "source_instance_id": inst.get("source_instance_id", inst["id"]),
        **{f"feat_{i}": val for i, val in enumerate(feat)},
        **model_info,
    }
    if true_n <= ML_EXACT_SMALL_N_THRESHOLD:
        add_small_exact_fields(record, exact_objective, exact_status or "UNKNOWN")
    return normalize_result_record(record)


def neighbor_info_fields(neighbors):
    if not neighbors:
        return {
            "ml_neighbor_ids": "",
            "ml_neighbor_ns": "",
            "ml_neighbor_label_Ks": "",
            "ml_neighbor_distances": "",
        }
    top = neighbors[:ML_NEIGHBORS]
    return {
        "ml_neighbor_ids": ";".join(str(row.get("training_instance_id", "")) for row in top),
        "ml_neighbor_ns": ";".join(str(row.get("training_n", "")) for row in top),
        "ml_neighbor_label_Ks": ";".join(str(row.get("label_K", "")) for row in top),
        "ml_neighbor_distances": ";".join(f"{row.get('distance', 0.0):.6g}" for row in top),
    }


def main():
    records, completed = load_existing_comparison_records()
    model, label_df = train_k_selection_model_from_records(TRAINING_DATA_FILE, records)
    instances = load_comparison_instances(records)

    for inst in instances:
        inst_id = int(inst["id"])
        p = inst["p"]
        w = inst["w"]
        feat = extract_features(p, w, len(p), m)
        exact_objective, exact_status = (None, "SKIPPED_N_TOO_LARGE")
        if len(p) <= ML_EXACT_SMALL_N_THRESHOLD:
            exact_objective, exact_status = solve_small_exact_dp(p, w, m)
        predicted_k, neighbors, selected_model_name = model.predict(feat)
        predicted_k = int(min(K_VALUES, key=lambda value: abs(value - predicted_k)))
        methods = [
            ("baseline_fixed_k", BASELINE_K, {}),
            (
                "ml_selected_k",
                predicted_k,
                {
                    "ml_predicted_K": predicted_k,
                    "ml_selected_model": selected_model_name,
                    "ml_model_neighbors": ML_NEIGHBORS,
                    "ml_exact_small_n_threshold": ML_EXACT_SMALL_N_THRESHOLD,
                    "ml_small_n_threshold": ML_SMALL_N_THRESHOLD,
                    "ml_objective_tol": ML_OBJECTIVE_TOL,
                    "ml_large_objective_rel_tol": LARGE_OBJECTIVE_REL_TOL,
                    "ml_label_rule_version": ML_LABEL_RULE_VERSION,
                    **neighbor_info_fields(neighbors),
                },
            ),
        ]
        exact_msg = ""
        if len(p) <= ML_EXACT_SMALL_N_THRESHOLD:
            exact_msg = f", exact_dp={exact_objective:.2f}({exact_status})"
        print(
            f"\nInstance {inst_id}: recorded_n={inst['n']}, true_n={len(p)}, "
            f"ML predicted K={predicted_k} via {selected_model_name}{exact_msg}"
        )

        for method, k_val, model_info in methods:
            key = (inst_id, method)
            if key in completed:
                print(f"  {method} already saved, skip.")
                continue
            print(f"  Running {method} with K={k_val}...", end="", flush=True)
            result = column_generation_with_time_limit(p, w, m, k_val, TIME_LIMIT)
            record = make_comparison_record(
                inst,
                method,
                k_val,
                result,
                feat,
                model_info=model_info,
                exact_objective=exact_objective,
                exact_status=exact_status,
            )
            upsert_comparison_record(records, record)
            if record_is_finished(record):
                completed.add(key)
            save_comparison_records(records)
            print(
                f" finished, comparison_obj={record['comparison_objective']:.2f}, "
                f"lp_obj={record['objective']:.2f}, time={record['time_sec']:.1f}s, "
                f"iters={record['iterations']}, cols={record['num_columns']}, status={record['cg_status']}"
            )
            gc.collect()

        if ONLINE_LEARNING_ENABLED:
            new_feedback = build_online_label_rows_from_records(records)
            if new_feedback:
                save_persistent_online_feedback_rows(new_feedback)
            model, label_df = train_k_selection_model_from_records(TRAINING_DATA_FILE, records)
            print("  Online learning model refreshed with persistent feedback labels.")

    save_comparison_records(records)
    summary = build_comparison_summary(pd.DataFrame(records))
    if not summary.empty:
        print("\nComparison summary:")
        print(summary.to_string(index=False))
    print(f"\nAll comparison data saved to {COMPARISON_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
