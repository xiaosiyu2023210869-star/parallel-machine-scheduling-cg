# Contributing

1. Create a focused branch.
2. Keep instance data in `data/*.json`; do not embed large arrays in Python modules.
3. Preserve deterministic seeds when changing algorithms.
4. Run `python -m compileall -q src scripts` and `python -m unittest discover -s tests -v`.
5. Do not commit solver licenses, generated models, checkpoints, or private experiment outputs.
