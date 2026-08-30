# Reproducibility

## Default scheduling parameters

| Workflow | Machines | Tool-change interval | Tool-change time | Time limit | K |
|---|---:|---:|---:|---:|---:|
| Small comparison | 3 | 4 | 2.0 | 1800 s | 30 |
| Medium comparison | 3 | 2 | 4.0 | 1800 s | 30 |
| Large comparison | 3 | 2 | 4.0 | 1800 s | 30 |
| Training sweep | 3 | 2 | 4.0 | 1800 s | 5, 10, ..., 50 |
| ML-assisted comparison | 3 | 2 | 4.0 | 1800 s | fixed 30 vs selected K |
| WSPT baseline | same as saved CG row | same as instance scale | same as instance scale | solver-free | none |
| Theta/c sensitivity | 3 | 2 or 4 | `rho * mean(p)` | 1800 s | 30 |
| Machine-count sensitivity | 3, 6, 9 | same as instance scale | same as instance scale | scale-specific | 30 |

The comparison, training, and ML workflows use random seed `20260524`. The theta/c experiments use seed `20260819`. Each experiment writes checkpoints after completed runs so a long experiment can resume safely.

All workflows enforce a valid machine count at the instance level. If a requested value satisfies `requested_m > n`, the run uses `m = n` for that instance and records both `requested_m` and the actual `m` in the output row. This avoids infeasible set-partitioning masters caused by requesting more nonempty machine schedules than jobs.

## Canonical column-generation implementation

All fixed-`K` runs use `pmcg.ml_assisted.column_generation_with_time_limit`. The small/medium/large comparison and training workflows call it through `run_fixed_k_column_generation`, which temporarily applies the workflow's tool-change interval and tool-change time. This keeps pricing, RMP updates, final `K`-sensitive schedule generation, RMP rounding, and the final integer master identical across experiments.

## WSPT-BR baseline

The WSPT baseline is computed by `scripts/run_wspt_baseline.py`. It reads a saved comparison workbook or CSV, filters the existing `CG` rows, and adds one `WSPT-BR` row for each matching instance id. It does not call Gurobi or rerun column generation.

`WSPT-BR` first sorts jobs by the WSPT ratio, assigns them using a least-current-processing-load rule, splits each machine sequence into tool-life blocks of length `c`, sorts jobs inside each block by WSPT, and orders the blocks by the batch-level index `(P_B + theta) / W_B`, where `P_B` and `W_B` are the block processing-time and weight totals.

## Theta/c sensitivity

`scripts/run_theta_c_sensitivity.py` reproduces the 75-case primary theta scan and the 36-case fixed-`c` interaction experiment. The primary scan uses 25 job counts, `rho` in `{0.5, 1.0, 2.0}`, and `c=4` for `n <= 36` or `c=2` for `n > 36`. The interaction experiment uses six representative instances at both `c=2` and `c=4`.

`scripts/run_high_rho_extension.py` reproduces the 12 additional cases at `rho` in `{3.0, 4.0}` for `n=6,16,40,110,150,200`, using the same size-dependent `c` policy. Processing times and weights are generated independently as discrete uniform integers on `{1,...,36}`. Use `--dry-run` on either entry point to validate the complete case plan without calling Gurobi.

## Machine-count sensitivity

Use `scripts/run_m_sensitivity.py` to evaluate requested values such as `m=3,6,9` without changing source-code constants:

```bash
python scripts/run_m_sensitivity.py --scale small --machines 3 6 9 --time-limit 1800
python scripts/run_m_sensitivity.py --scale medium --machines 3 6 9 --time-limit 1800
python scripts/run_m_sensitivity.py --scale large --machines 3 6 9 --time-limit 1800
```

The script runs the canonical fixed-`K` CG engine, keeps checkpoint rows in `outputs/m-sensitivity/`, and reports `requested_m`, actual `m`, and whether the value was adjusted.

Section 6.5 uses the 18 records in `data/instances_m_sensitivity.json`. All 18 are paired at `m=6,9`; instance ids `860018`, `860022`, and `860030` also form the complete `m=3,6,9` subset. The JSON file preserves the 600-second budgets for `n <= 30` in the first group and 1800-second budgets for the eight larger fixed instances. `run_m_sensitivity.py` reads these record-level budgets unless `--time-limit` is supplied.

## Solver reporting

Record the Gurobi version, license type, CPU, RAM, operating system, Python version, wall-clock limit, optimality gap, and random seed with published results. A time-limited incumbent is not a proof of global optimality.

## Data policy

The repository includes the instance definitions and the training workbook used by the experiment scripts. Final artifacts used in the manuscript are published under `results/`, including the comparison, theta/c, high-rho, machine-count, and ML workbooks, the online feedback records, and the trained K-selection model. Locally generated checkpoints and plots belong in `outputs/` and remain ignored by Git.
