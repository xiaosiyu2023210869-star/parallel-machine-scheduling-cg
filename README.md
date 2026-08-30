# Parallel-Machine Scheduling with Column Generation

Reproducibility code for parallel-machine weighted-completion-time scheduling with periodic tool changes, heuristic column generation, a WSPT-based baseline, and machine-learning-assisted selection of the column-insertion parameter `K`.

## Workflows

1. **Four-method comparison**: Full MIP, two restricted benchmark MIPs, and the same fixed-`K` column-generation engine used by the ML workflow on small, medium, or large instances.
2. **WSPT baseline**: read saved comparison results and append a solver-free WSPT-BR heuristic row for the same instance ids.
3. **Reset-duration sensitivity**: compare CG with WSPT-BR over the paper's 111 primary/interaction cases and 12 high-`rho` extension cases.
4. **Machine-count sensitivity**: rerun fixed-`K` column generation for requested values such as `m=3,6,9`.
5. **ML-assisted column generation**: generate `K`-sweep training records, train/select `K`, and compare fixed-`K` with ML-selected-`K` runs.

The small benchmark retains the recorded FullMIP/Case11MIP/Case12MIP values from the source experiment and recomputes CG. Medium and large benchmarks solve all four methods.

The fixed-`K` implementation in `src/pmcg/ml_assisted.py` is canonical. The comparison and training modules delegate to that implementation while supplying their workflow-specific machine, tool-change, `K`, and time-limit settings.

## Repository layout

```text
data/                 Instance JSON files and the training workbook
src/pmcg/             Comparison, training, and ML-assisted engines
scripts/              Command-line entry points
tests/                Fast structural and data-integrity checks
docs/                 Reproducibility and source-mapping notes
outputs/              Generated checkpoints, workbooks, models, and figures
```

## Requirements

- Python 3.10+
- Gurobi 12 or 13 and a valid local license
- Packages listed in `requirements.txt`

Gurobi is commercial software. The Python package can be installed from PyPI, but optimization runs still require a valid Gurobi license.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[ml]"
```

## Run the comparison experiment

```bash
python scripts/run_comparison.py --scale small
python scripts/run_comparison.py --scale medium --time-limit 1800
python scripts/run_comparison.py --scale large --k 30
python scripts/run_comparison.py --scale medium --machines 6
```

Checkpoints and plots are written under `outputs/comparison-<scale>/`. Re-running the same command resumes from completed CSV records.

When a requested machine count is larger than the number of jobs in an instance, the scripts use `m = min(requested_m, n)`. Output rows include `requested_m`, the actual `m`, and `machine_count_adjusted`.

## Run machine-count sensitivity

```bash
python scripts/run_m_sensitivity.py --scale small --machines 3 6 9 --time-limit 1800
python scripts/run_m_sensitivity.py --scale medium --machines 3 6 9 --time-limit 1800
```

The sensitivity script runs fixed-`K` column generation only and writes `outputs/m-sensitivity/cg_m_sensitivity_<scale>.csv`.

To reproduce the paired machine-count dataset used in the paper, first run all 18 fixed instances at `m=6,9`, then add the three-instance `m=3` subset to the same checkpoint:

```bash
python scripts/run_m_sensitivity.py --scale medium --instances data/instances_m_sensitivity.json --machines 6 9
python scripts/run_m_sensitivity.py --scale medium --instances data/instances_m_sensitivity.json --machines 3 --instance-ids 860018 860022 860030
```

The data file stores the paper's per-instance time budgets: 600 seconds for the smaller cases and 1800 seconds for the larger cases. Passing `--time-limit` overrides them.

## Add the WSPT baseline to saved CG results

```bash
python scripts/run_wspt_baseline.py --saved-comparison path/to/saved_comparison.xlsx
python scripts/run_wspt_baseline.py --saved-comparison path/to/saved_comparison.xlsx --machines 6
```

The script does not run column generation. It uses the saved `CG` rows as the comparison reference, loads the corresponding fixed instances from `data/`, and writes `WSPT-BR` rows under `outputs/wspt-baseline/`.

## Run reset-duration sensitivity

Validate the experiment plans without starting Gurobi:

```bash
python scripts/run_theta_c_sensitivity.py --dry-run
python scripts/run_high_rho_extension.py --dry-run
```

Run the 111 primary and fixed-`c` interaction cases, followed by the 12 high-`rho` cases reported in the paper:

```bash
python scripts/run_theta_c_sensitivity.py --cg-time-limit 1800
python scripts/run_high_rho_extension.py --cg-time-limit 1800
```

Both scripts use seed `20260819`, generate processing times and weights independently from the discrete uniform distribution on `1,...,36`, and write checkpoint workbooks under `outputs/theta-c-sensitivity/`. The high-`rho` extension uses `n=6,16,40,110,150,200`, `rho=3,4`, and the paper's size-dependent `c` policy.

## Run the ML workflow

```bash
python scripts/generate_training_data.py --time-limit 1800
python scripts/generate_training_data.py --time-limit 1800 --machines 6
python scripts/run_ml_assisted.py --training-data data/training_data.xlsx
python scripts/run_ml_assisted.py --training-data data/training_data.xlsx --machines 6
```

The ML experiment writes its model, online-feedback workbook, and timestamped comparison results under `outputs/ml-assisted/`. These generated artifacts are ignored by Git.

## Quick validation

```bash
python -m compileall -q src scripts
python -m unittest discover -s tests -v
```

## Reproducibility notes

The experiment random seed and default parameters are preserved. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for parameter mappings and solver-reporting notes.
