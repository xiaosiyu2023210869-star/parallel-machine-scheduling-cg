# Reproducibility

## Default scheduling parameters

| Workflow | Machines | Tool-change interval | Tool-change time | Time limit | K |
|---|---:|---:|---:|---:|---:|
| Small comparison | 3 | 4 | 2.0 | 600 s | 30 |
| Medium comparison | 3 | 2 | 4.0 | 1800 s | 30 |
| Large comparison | 3 | 2 | 4.0 | 1800 s | 30 |
| Training sweep | 3 | 2 | 4.0 | 1800 s | 5, 10, ..., 50 |
| ML-assisted comparison | 3 | 2 | 4.0 | 1800 s | fixed 30 vs selected K |

The original random seed is `20260524`. Each experiment writes checkpoints after completed runs so a long experiment can resume safely.

## Solver reporting

Record the Gurobi version, license type, CPU, RAM, operating system, Python version, wall-clock limit, optimality gap, and random seed with published results. A time-limited incumbent is not a proof of global optimality.

## Data policy

The repository includes the instance definitions and the training workbook used by the recovered scripts. Generated models, feedback workbooks, checkpoints, and plots belong in `outputs/` and are ignored by Git.
