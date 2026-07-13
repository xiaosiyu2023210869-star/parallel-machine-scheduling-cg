# Data

- `instances_small.json`: six small instances and recorded baseline values.
- `instances_medium.json`: eight medium-scale instances.
- `instances_large.json`: nine large-scale instances.
- `instances_training.json`: instances used for K-sweep data generation.
- `instances_ml.json`: instances used for fixed-K versus ML-selected-K comparison.
- `training_data.xlsx`: recovered training workbook used by the ML selector.

Every JSON record must satisfy `n == len(p) == len(w)` and contain strictly positive processing times and weights.
