# Published experiment artifacts

These files are the final tabular artifacts used for the manuscript and its supplementary materials. They are versioned separately from `outputs/`, which remains reserved for locally generated checkpoints and scratch runs.

| Directory | Artifact | Paper coverage |
| --- | --- | --- |
| `comparison/` | `comparison_23_instances.xlsx` | Main comparison: 23 instances, FullMIP/Case11MIP/Case12MIP/CG/WSPT-BR records and solver-status fields. |
| `theta/` | `theta_c_sensitivity_111_cases.xlsx` | Primary and interaction theta/c sensitivity: 111 cases. |
| `theta/` | `theta_high_rho_12_cases.xlsx` | Matched high-rho extension: 12 cases at ratios 3 and 4. |
| `machine_count/` | `machine_count_sensitivity_18_pairs.xlsx` | Fixed-instance machine-count sensitivity: 18 matched m=6/m=9 pairs. |
| `machine_count/` | `machine_count_sensitivity_analysis.json` | The three-instance nested m=3/m=6/m=9 subset and the paired m=6/m=9 analysis used for Fig. 9. |
| `ml/` | `ml_comparison_231_pairs.xlsx` | Learning-assisted comparison: 231 paired fixed-K/ML-selected-K cases. |
| `ml/` | `online_feedback_202_records.xlsx` | Online feedback records used by the learning controller. |
| `ml/` | `training_data_source_2890_records.xlsx` | Source training workbook corresponding to the published training data. |
| `ml/` | `k_selection_model.pkl` | Trained K-selection model used by the ML experiment. |

The workbook dimensions and case counts are recorded in `manifest.json`. All files are published as-is from the final local experiment outputs; no solver rerun is implied by this release.
