# MovieMind

### A personalized movie recommendation agent powered by Q-learning

MovieMind is a full-stack reinforcement learning mini-project for **Unit III: Reinforcement Learning**. It turns a movie rating into an observable learning step: the user rates a recommendation, the agent receives a reward, updates its Q-table and preference state, then uses the updated policy for later recommendations.

The interactive learner is a genuine tabular Q-learning agent. Separate offline experiments use historical MovieLens ratings for epsilon comparisons, fitted-Q training, and evaluation against popularity. Historical ratings are not presented as live user feedback or as counterfactual responses.

## At a glance

- **Dataset:** Original GroupLens MovieLens 1M — 6,040 users, 3,883 movies, 1,000,209 ratings.
- **Algorithm:** Persistent, per-profile tabular Q-learning with epsilon-greedy exploration.
- **Advanced component:** Offline fitted Q-learning with an ExtraTrees regressor.
- **Evaluation:** Chronological holdout comparison against a popularity baseline.
- **Interface:** React, Vite, FastAPI, SQLite, and Recharts.
- **Poster artwork:** Wikipedia image lookup when available, with a generated fallback; an internet connection is needed to look up external images.

## Contents

- [Features](#features)
- [Quick start](#quick-start)
- [MovieLens dataset](#movielens-dataset)
- [Reinforcement learning design](#reinforcement-learning-design)
- [Experiments and evaluation](#experiments-and-evaluation)
- [Project structure](#project-structure)
- [Limitations](#limitations)
- [Documentation](#documentation)

## Features

- Browse, search, and filter the MovieLens catalog.
- Explore popular, trending, and personalized recommendations.
- Switch between valid MovieLens user profiles; profiles and learned Q-state/action values stay isolated by user ID.
- Rate a movie from 1 to 5 stars and inspect the reward, state transition, and Q-value update.
- Review rating and recommendation history, genre preferences, and learning statistics.
- Compare tabular Q-learning and fitted Q-learning with a popularity baseline.
- Inspect the Q-learning and evaluation results in the dashboard and evaluation pages.

## Quick start

### Prerequisites

- Python 3.10 or later.
- Node.js `^20.19.0` or `>=22.12.0` (required by the installed Vite 8 release).
- The MovieLens 1M dataset from [GroupLens](https://grouplens.org/datasets/movielens/1m/).

### 1. Install the dataset

Download `ml-1m.zip` from the official GroupLens page and extract it so these files are present:

```text
data/ml-1m/movies.dat
data/ml-1m/ratings.dat
data/ml-1m/users.dat
```

On Windows, from the repository root:

```powershell
New-Item -ItemType Directory -Force .\data | Out-Null
# Save the downloaded ml-1m.zip to .\data\ first.
Expand-Archive -LiteralPath .\data\ml-1m.zip -DestinationPath .\data
Get-ChildItem .\data\ml-1m\movies.dat, .\data\ml-1m\ratings.dat, .\data\ml-1m\users.dat
```

The dataset is not committed to this repository. The loader uses the original `::`-delimited files and does not substitute sample data. If the files are missing, the API reports the missing paths and data-dependent requests return HTTP 503.

### 2. Start the API

In a terminal at the repository root:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

The API is at `http://127.0.0.1:8000`; interactive API documentation is at `http://127.0.0.1:8000/docs`.

### 3. Start the web app

In a second terminal at the repository root:

```powershell
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

Open `http://127.0.0.1:5173`. Vite proxies `/api` requests to the FastAPI backend. Keep both terminals running while using the app.

## MovieLens dataset

The checked dataset contains:

| Table | Rows |
|---|---:|
| Users | 6,040 |
| Movies | 3,883 |
| Ratings | 1,000,209 |

The loader converts rating timestamps to UTC, parses movie genres and years, and reports missing cells and duplicate rows. Inspection of the current data found zero missing cells and zero duplicate rows. The API's `/api/dataset/summary` endpoint reports the loaded dataset statistics.

## Reinforcement learning design

| Element | MovieMind design |
|---|---|
| **Agent** | A tabular Q-learning policy, with state/action keys namespaced by profile ID. |
| **Environment** | The movie catalog and the active user's explicit rating feedback. |
| **State** | Five normalized genre affinities: Action, Comedy, Romance, Drama, and Sci-Fi. Each is discretized into low, medium, or high, giving at most `3^5 = 243` states. |
| **Action** | Recommend a movie from up to 600 popular eligible titles the user has not rated. |
| **Reward** | `1★ → -2`, `2★ → -1`, `3★ → 0`, `4★ → +1`, `5★ → +2`. |
| **Next state** | Adjust the rated movie's represented genre affinities by a bounded amount proportional to reward, then discretize again. |
| **Policy** | Epsilon-greedy: explore with a random eligible movie, otherwise choose a highest-Q action. Preference affinity breaks Q-value ties. Epsilon decays after feedback to a minimum. |
| **Terminal case** | If no eligible next action remains, the future-value term is zero. |

The learner uses `α = 0.15`, `γ = 0.85`, initial `ε = 0.25`, decay `0.995`, and minimum `ε = 0.03`. The update is:

```text
Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') − Q(s, a)]
```

The live cycle is:

```text
User profile → state → recommended movie → user rating → reward
→ next state → Q-table update → persisted policy → next recommendation
```

The Q-table is persisted in `models/q_table.json`; profiles, submitted feedback, and recommendation history are stored in SQLite. Historical MovieLens ratings initialize a profile's genre affinities and help exclude movies the user has already rated.

### Offline fitted Q-learning

`backend/rl/fitted_q_learning.py` implements a simplified fitted-Q component using an ExtraTrees regressor over genre-affinity features and a normalized movie-ID action feature. It bootstraps over a fixed pool of 64 popular movies from the training window. This is a separate offline model, not the interactive Q-table or live feedback.

## Experiments and evaluation

Run these commands from the repository root after installing the dataset and backend dependencies:

```powershell
.\backend\.venv\Scripts\python.exe experiments\epsilon_experiments.py
.\backend\.venv\Scripts\python.exe experiments\evaluate.py
```

Both scripts save timestamped result files and charts under `experiments/results/`; the evaluation script also writes a JSON summary.

### Epsilon comparison

The fixed-seed replay compares `ε = 1.0`, `0.5`, and `0.1` with a decaying strategy. It selects from sampled users' historically rated movies (1,000 users, up to 20 steps each), so it compares behavior on known outcomes; it does not estimate how a user would rate an unseen recommendation.

| Strategy | Mean reward | Exploration rate | Final epsilon |
|---|---:|---:|---:|
| Fixed `1.0` | 0.7068 | 1.0000 | 1.0000 |
| Fixed `0.5` | 0.8999 | 0.4981 | 0.5000 |
| Fixed `0.1` | 0.9941 | 0.0998 | 0.1000 |
| Decay from `1.0` | 0.8613 | 0.6285 | 0.3679 |

### Chronological holdout results

The recorded evaluation uses 800,168 earlier ratings for training and 200,041 later ratings for testing. Relevance is a held-out rating of 4 or 5. The table shows the measured `K = 10` results:

| Model | Hit Rate@10 | Precision@10 | Recall@10 | Coverage | Observed reward / user |
|---|---:|---:|---:|---:|---:|
| Tabular Q-learning | 0.5979 | 0.1522 | 0.0300 | 0.1712 | 2.3606 |
| Popularity baseline | 0.5782 | 0.1501 | 0.0301 | 0.0177 | 2.4728 |
| Fitted Q-learning | 0.1755 | 0.0251 | 0.0043 | 0.6526 | 0.3130 |

Q-learning is not better on every metric in this split; popularity has higher observed reward and the fitted-Q model has weak ranking metrics. These offline results are descriptive, not evidence of counterfactual live-user responses. See [docs/EVALUATION.md](docs/EVALUATION.md) for the full metrics, methodology, and caveats.

## Project structure

```text
backend/
  main.py                 FastAPI endpoints and recommendation flow
  data_loader.py          MovieLens 1M loader
  database.py             SQLite persistence
  rl/                     Q-learning, state, reward, fitted-Q modules
  recommender/            Popularity ranking and evaluation metrics
frontend/
  src/                    React application and styles
experiments/
  evaluate.py             Chronological policy comparison
  epsilon_experiments.py  Epsilon exploration replay
  results/                Timestamped experiment outputs
tests/                    RL, dataset, and API integration tests
docs/                     Algorithm, architecture, demo, and evaluation guides
```

## Limitations

- MovieLens 1M is historical data; it does not contain recommendation exposure or live responses.
- The interactive state models only five genres; other genres do not contribute to its discrete state.
- The recommendation candidate set is capped for responsiveness.
- The fitted-Q component is a teaching-scale approximation with a restricted action pool.
- Movie poster lookup depends on internet access and image availability; generated artwork is used when no match is found.

## Documentation

- [Installation and runbook](docs/INSTALLATION.md)
- [Learning algorithm](docs/ALGORITHM.md)
- [System architecture](docs/ARCHITECTURE.md)
- [Demo guide](docs/DEMO_GUIDE.md)
- [Measured evaluation results](docs/EVALUATION.md)
- [Project report notes](docs/PROJECT_REPORT.md)

## Credits

MovieLens 1M is provided by [GroupLens Research](https://grouplens.org/datasets/movielens/1m/). Movie poster thumbnails are looked up from Wikipedia when available. This project is an educational demonstration of reinforcement learning, not a production recommendation service.
