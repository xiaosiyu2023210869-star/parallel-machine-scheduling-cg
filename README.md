# Parallel-Machine Scheduling with Column Generation

Reproducibility code for parallel-machine weighted-completion-time scheduling with batching constraints, periodic tool changes, hybrid column generation, and machine-learning-assisted selection of the column-insertion parameter `K`.

## Workflows

1. **Four-method comparison**: Full MIP, two special-case strengthened MIPs, and the same fixed-`K` column-generation engine used by the ML workflow on small, medium, or large instances.
2. **ML-assisted column generation**: generate `K`-sweep training records, train/select `K`, and compare fixed-`K` with ML-selected-`K` runs.

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
```

Checkpoints and plots are written under `outputs/comparison-<scale>/`. Re-running the same command resumes from completed CSV records.

## Run the ML workflow

```bash
python scripts/generate_training_data.py --time-limit 1800
python scripts/run_ml_assisted.py --training-data data/training_data.xlsx
```

The ML experiment writes its model, online-feedback workbook, and timestamped comparison results under `outputs/ml-assisted/`. These generated artifacts are ignored by Git.

## Quick validation

```bash
python -m compileall -q src scripts
python -m unittest discover -s tests -v
```

## Reproducibility notes

The experiment random seed and default parameters are preserved. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for parameter mappings and solver-reporting notes.
