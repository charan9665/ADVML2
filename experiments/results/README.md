# Experiment artifacts

Timestamped files are the authoritative outputs from executed experiments. The epsilon replay run is `epsilon_experiments_20261117T101452915722Z.csv` and its matching learning-curve CSV/PNG. It replays actual MovieLens ratings for a fixed, seeded set of users and compares epsilon-greedy policies. This is historical offline replay, not live user feedback or a counterfactual online experiment.

The evaluation artifacts are `evaluation_20261117T101937521826Z.*` (JSON, CSV, PNG). They use the chronological MovieLens split documented in `docs/EVALUATION.md`.

The older un-timestamped `epsilon_experiments.csv` and `epsilon_learning_curve.*` files were produced by an earlier neutral reward-pool prototype in which each epsilon policy saw the same reward sequence. They are retained as project history, but should not be used as the final epsilon comparison. Use the timestamped policy replay artifacts instead.
