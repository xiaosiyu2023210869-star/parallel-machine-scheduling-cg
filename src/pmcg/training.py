# -*- coding: utf-8 -*-
"""
Training-data workflow for the canonical, time-limited fixed-K column generation.
"""

import ast
import gc
import json
import math
import os
import pickle
import shutil
from datetime import datetime
from pathlib import Path

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

# -------------------- Parameters --------------------
m = 3
c = 2
theta = 4.0

TIME_LIMIT = 1800
K_VALUES = list(range(5, 51, 5))

# Resume controls.  The CSV result file is the checkpoint: any saved
# (instance_id, K) row is treated as already run, even if it timed out.
# Because the current manual run has reached n=22, K=30, the next fresh
# launch skips everything up to and including that pair.  Set this to None
# after you no longer need the manual cutoff.
MANUAL_RESUME_AFTER = None
RESUME_RUN_ID = "training_excel_n40_to_n80_rerun_v1"
CHECKPOINT_VERSION_COLUMN = "checkpoint_version"
CHECKPOINT_TIME_COLUMN = "checkpoint_saved_at"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(os.environ.get("PMCG_OUTPUT_DIR", PROJECT_ROOT / "outputs" / "training"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = OUTPUT_DIR / f"training_data_{timestamp}.xlsx"
LEGACY_PARTIAL_FILE = OUTPUT_DIR / "training_data_partial.csv"
MODEL_FILE = OUTPUT_DIR / "k_selection_model.pkl"
COMPARISON_MIN_N = 6
COMPARISON_MAX_N = 36
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INSTANCES_FILE = PROJECT_ROOT / "data" / "instances_training.json"
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



def column_generation_with_time_limit(p, w, m_val, k_fixed, time_limit):
    """Delegate training runs to the ML workflow's canonical fixed-K CG."""
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
    objective = as_float_or_nan(record.get("objective"))
    last_rmp_obj = as_float_or_nan(record.get("last_rmp_obj"))
    if not math.isnan(last_rmp_obj) and (math.isnan(objective) or last_rmp_obj < objective):
        record["train_objective"] = last_rmp_obj
        record["train_objective_source"] = "last_rmp_obj"
    else:
        record["train_objective"] = objective
        source = record.get("objective_source", "objective")
        if pd.isna(source):
            source = "objective"
        record["train_objective_source"] = str(source)
    return record


def load_existing_records():
    if OUTPUT_FILE.exists():
        backup = OUTPUT_FILE.with_name(
            f"{OUTPUT_FILE.stem}_backup_before_{RESUME_RUN_ID}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{OUTPUT_FILE.suffix}"
        )
        shutil.copy2(OUTPUT_FILE, backup)
        print(f"Backed up existing {OUTPUT_FILE} to {backup}. Starting a full rerun.")
    else:
        print(f"No existing {OUTPUT_FILE}. Starting a full rerun.")
    return [], set()


def record_timed_out(record):
    value = record.get("timed_out", False)
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def record_is_finished(record):
    """A saved row is a checkpoint, including rows stopped by time limit."""
    required = ["instance_id", "K", "objective", "time_sec", "cg_status"]
    for key in required:
        if key not in record or pd.isna(record[key]):
            return False
    try:
        float(record["objective"])
        float(record["time_sec"])
        int(record["instance_id"])
        int(record["K"])
    except (TypeError, ValueError):
        return False
    status = str(record.get("cg_status", "")).strip().lower()
    return status not in {"", "nan", "not_started"}


def should_skip_by_manual_resume(n, k_val):
    if MANUAL_RESUME_AFTER is None:
        return False
    resume_n, resume_k = MANUAL_RESUME_AFTER
    return (int(n), int(k_val)) <= (int(resume_n), int(resume_k))


def record_has_active_checkpoint(record):
    value = record.get(CHECKPOINT_VERSION_COLUMN, "")
    if pd.isna(value):
        return False
    return str(value).strip() == RESUME_RUN_ID


def record_should_be_skipped(record):
    if not record_is_finished(record):
        return False
    try:
        n = int(record.get("n", -1))
    except (TypeError, ValueError):
        n = -1
    if n < COMPARISON_MIN_N:
        return True
    if not record_has_active_checkpoint(record):
        return False
    k_val = int(record["K"])
    if should_skip_by_manual_resume(n, k_val):
        return True
    return True


def upsert_record(records, record):
    key = (int(record["instance_id"]), int(record["K"]))
    replaced = False
    kept = []
    for old_record in records:
        old_key = (int(old_record["instance_id"]), int(old_record["K"]))
        if old_key == key:
            if not replaced:
                kept.append(record)
                replaced = True
            continue
        kept.append(old_record)
    if not replaced:
        kept.append(record)
    records[:] = kept


def save_records(records):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    normalized = [add_train_objective_fields(dict(record)) for record in records]
    df = pd.DataFrame(normalized)
    if df.empty:
        df.to_excel(OUTPUT_FILE, index=False)
        return
    df.sort_values(["instance_id", "K"]).to_excel(OUTPUT_FILE, index=False)


def train_model_if_ready(records):
    df = pd.DataFrame(records)
    if df.empty:
        print("No records available; skip model training.")
        return

    unique_instances = df["instance_id"].nunique()
    if unique_instances < 5:
        print(f"Only {unique_instances} instances available; skip model training for now.")
        return

    try:
        import lightgbm as lgb
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        print(f"Skip model training because a dependency is missing: {exc}")
        return

    if "train_objective" not in df.columns:
        df = pd.DataFrame([add_train_objective_fields(dict(record)) for record in records])

    feat_cols = [col for col in df.columns if col.startswith("feat_")]
    best_df = (
        df.sort_values(["timed_out", "train_objective", "time_sec"])
        .groupby("instance_id", as_index=False)
        .first()[["instance_id", "n", "K", "train_objective", "time_sec", "iterations"]]
    )
    unique_feat = df.drop_duplicates(subset="instance_id")[["instance_id"] + feat_cols].set_index("instance_id")
    X = unique_feat.loc[best_df["instance_id"]].values
    y = best_df["K"].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = lgb.LGBMRegressor(n_estimators=100, max_depth=6, verbosity=-1, random_state=42)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(10)],
    )

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_FILE}")


def main():
    records, completed = load_existing_records()
    instances = load_comparison_instances(records)

    for inst in instances:
        inst_id = inst["id"]
        n = inst["n"]
        p = inst["p"]
        w = inst["w"]
        feat = extract_features(p, w, n, m)
        print(f"\nInstance {inst_id}: n={n}")

        for k_val in K_VALUES:
            key = (inst_id, k_val)
            if should_skip_by_manual_resume(n, k_val):
                print(f"  K={k_val} skipped by manual resume point {MANUAL_RESUME_AFTER}.")
                continue
            if key in completed:
                print(f"  K={k_val} already saved in checkpoint, skip.")
                continue

            print(f"  Running K={k_val}...", end="", flush=True)
            result = column_generation_with_time_limit(p, w, m, k_val, TIME_LIMIT)
            record = {
                "instance_id": inst_id,
                "n": n,
                "K": k_val,
                **result,
                CHECKPOINT_VERSION_COLUMN: RESUME_RUN_ID,
                CHECKPOINT_TIME_COLUMN: datetime.now().isoformat(timespec="seconds"),
                "source_instance_id": inst.get("source_instance_id", inst_id),
                **{f"feat_{i}": val for i, val in enumerate(feat)},
            }
            add_train_objective_fields(record)
            upsert_record(records, record)
            if record_is_finished(record):
                completed.add((inst_id, k_val))
            save_records(records)

            print(
                f" finished, obj={record['objective']:.2f}, "
                f"inc={record['incumbent_objective']:.2f}, "
                f"train_obj={record['train_objective']:.2f}({record['train_objective_source']}), "
                f"time={record['time_sec']:.1f}s, "
                f"iters={record['iterations']}, cols={record['num_columns']}, "
                f"status={record['cg_status']}, timed_out={record['timed_out']}"
            )
            gc.collect()

    save_records(records)
    print(f"\nAll data saved to {OUTPUT_FILE}")
    print("Model training skipped for this standalone Excel export.")


if __name__ == "__main__":
    main()
