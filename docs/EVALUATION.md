# MovieMind Actual Evaluation Results

## Data and split

These figures were produced by running `python experiments/evaluate.py` over the official MovieLens 1M files in this workspace. The global timestamp split placed 800,168 ratings in training and 200,041 in test, with a training cutoff of **2000-12-02 14:52:18 UTC**. There were 1,783 test users. Only earlier interactions trained the models and formed user states. Candidate lists excluded each user's training-seen movies. A test rating of 4 or 5 is relevant.

The fitted-Q model used 50,000 sampled training transitions and a 64-movie next-action pool chosen from training-window popularity. The popularity baseline ranked from training rows only. Epsilon replay results use 1,000 seeded users and 20 selected historical ratings per user; they are not counterfactual live-user outcomes.

The `/api/rl/train` endpoint was also executed against the training partition. It processed 800,168 historical interactions and saved a separate 238-state, 130,403-entry artifact at `models/offline_q_table_20261117T102437343286Z.json`. The endpoint reported that the live interactive policy was unchanged.

## Chronological holdout metrics

| Model | K | Hit Rate | Precision | Recall | Coverage | Mean observed reward / user |
|---|---:|---:|---:|---:|---:|---:|
| Tabular Q-learning | 5 | 0.449804 | 0.155244 | 0.016262 | 0.109503 | 1.214807 |
| Popularity baseline | 5 | 0.457656 | 0.172855 | 0.017557 | 0.011196 | 1.450365 |
| Fitted Q-learning | 5 | 0.122266 | 0.030286 | 0.002754 | 0.451393 | 0.190690 |
| Tabular Q-learning | 10 | 0.597869 | 0.152159 | 0.029979 | 0.171218 | 2.360628 |
| Popularity baseline | 10 | 0.578239 | 0.150084 | 0.030056 | 0.017750 | 2.472799 |
| Fitted Q-learning | 10 | 0.175547 | 0.025070 | 0.004337 | 0.652649 | 0.312956 |

Tabular Q-learning had the strongest Hit Rate@10 and slightly higher Precision@10 than popularity. Popularity had higher Hit Rate@5, Precision/Recall at both K values, and observed reward. Tabular and fitted Q had much higher catalog coverage than popularity. Fitted Q had the weakest hit/precision/recall/reward here, while reaching the broadest catalog coverage. These results do not support a claim that Q-learning is uniformly better.

Mean observed reward sums mapped rewards for recommended items with an observed held-out rating and divides by test users. Unrated recommendations do not have counterfactual outcomes. Coverage is measured against the training catalog.

## Epsilon replay

| Strategy | Episodes | Mean reward | Exploration | Exploitation | Final epsilon |
|---|---:|---:|---:|---:|---:|
| Fixed 1.0 | 20,000 | 0.7068 | 1.0000 | 0.0000 | 1.0000 |
| Fixed 0.5 | 20,000 | 0.8999 | 0.4981 | 0.5020 | 0.5000 |
| Fixed 0.1 | 20,000 | 0.9941 | 0.0998 | 0.9002 | 0.1000 |
| Decay 1.0 × 0.99995, min 0.03 | 20,000 | 0.8613 | 0.6285 | 0.3715 | 0.3679 |

The simulation gives each sampled user 20 selections from items they actually rated in the historical file. The policy observes a selected item's recorded rating and updates Q. This replay shows how epsilon affected reward over this historical action pool; it does not predict ratings for unseen movies.

## Artifacts

The run saved timestamped `evaluation_*.json`, `evaluation_*.csv`, `evaluation_*.png`, `epsilon_experiments_*.csv`, `epsilon_learning_curve_*.csv`, and `epsilon_learning_curve_*.png` files under `experiments/results/`. The API evaluation and experiment routes read the latest timestamped runs. Re-run the scripts to produce a new measured artifact set.
