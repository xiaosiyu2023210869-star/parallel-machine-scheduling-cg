# Source mapping

The repository was recovered and organized from the following local exports:

| Original export | Repository destination |
|---|---|
| `小规模对比.html` | `src/pmcg/small_comparison.py`, `data/instances_small.json` |
| `中规模对比.html` | `src/pmcg/comparison.py`, `data/instances_medium.json` |
| `大规模对比.html` | shared `src/pmcg/comparison.py`, `data/instances_large.json` |
| `数据生成代码.html` | `src/pmcg/training.py`, `data/instances_training.json` |
| `列生成_时间优先机器学习对比实验代码_每次新Excel_跨运行反馈版_副本.html` | `src/pmcg/ml_assisted.py`, `data/instances_ml.json` |

Repeated top-level function definitions that were overridden by later definitions in the notebook exports were removed. User-specific Desktop paths were replaced with repository-relative paths and command-line arguments. Raw result workbooks, online feedback, and serialized models are intentionally excluded from version control.
