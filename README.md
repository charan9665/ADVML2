# MovieMind

MovieMind is a college mini-project demonstrating an interactive, personalized movie recommender with a genuine tabular Q-learning agent. This workspace now contains the verified original MovieLens 1M files. Historical offline interactions and feedback submitted in the web app are kept conceptually separate.

## Dataset setup

Download the original [MovieLens 1M dataset from GroupLens](https://grouplens.org/datasets/movielens/1m/) and extract `ml-1m.zip` so these files exist:

```text
data/ml-1m/movies.dat
data/ml-1m/ratings.dat
data/ml-1m/users.dat
```

The loader parses the original `::` delimited files with Pandas, converts timestamps to UTC datetimes, counts missing cells and duplicates, and extracts years and genres. The loader does not silently substitute sample data. If the files are absent, the API starts and reports the exact missing paths while data-dependent endpoints return HTTP 503.

Verified workspace inspection: **6,040 users, 3,883 movies, 1,000,209 ratings**; zero duplicate rows and zero missing cells. Rating counts: 1★ 56,174; 2★ 107,557; 3★ 261,197; 4★ 348,971; 5★ 226,310. The 3,883 count is the number of actual rows in the downloaded file.

### Windows manual download and extraction

1. Open the official [GroupLens MovieLens 1M page](https://grouplens.org/datasets/movielens/1m/) and download `ml-1m.zip`.
2. Move the downloaded ZIP into this repository's `data` directory.
3. From the repository root in the VS Code terminal, run:

```powershell
Expand-Archive -LiteralPath .\data\ml-1m.zip -DestinationPath .\data
Get-ChildItem .\data\ml-1m\movies.dat, .\data\ml-1m\ratings.dat, .\data\ml-1m\users.dat
```

The extraction must produce `data/ml-1m/movies.dat`, `data/ml-1m/ratings.dat`, and `data/ml-1m/users.dat`. If Explorer extracts into an extra nested folder, move those three files directly into `data/ml-1m/`. Restart the API afterward. The local HTTPS download attempt in this environment fails certificate-chain validation, so do not disable TLS verification to work around it.

## Reinforcement learning formulation

- **Agent:** a user's persistent movie recommendation Q-learner.
- **Environment:** catalog plus the user's explicit rating feedback.
- **State:** five genre affinities (Action, Comedy, Romance, Drama, Sci-Fi), each quantized into low / medium / high. This gives at most 243 states and abstracts away the long rating history.
- **Action:** recommend one movie from the most popular 600 eligible unseen titles. Movies rated through the app are excluded for that user.
- **Reward:** 1★ → −2, 2★ → −1, 3★ → 0, 4★ → +1, 5★ → +2. The centered, bounded scale preserves ordering, treats a neutral rating as no preference evidence, and distinguishes positive from negative feedback.
- **Next state:** update each represented genre affinity by a bounded increment proportional to reward, then quantize again.
- **Policy:** epsilon-greedy. The action is selected uniformly at random for exploration; otherwise choose the highest Q value. Tied values use current genre affinity as a preference-aware tie-break, then random selection. Epsilon decays after updates to a minimum.
- **Terminal state:** future value is zero when there are no eligible next actions.

The tabular update is implemented directly in `backend/rl/q_learning.py`:

`Q(s,a) ← Q(s,a) + α [r + γ maxₐ′ Q(s′,a′) − Q(s,a)]`

The table persists in `models/q_table.json`; live feedback and profiles persist in SQLite. Interactive Q-state keys are namespaced by user ID, so users share the persisted model artifact without sharing one another's learned state/action values. Recommendations are logged separately from feedback.

## Switching MovieLens profiles

The header profile dropdown defaults to MovieLens User 1 and accepts any valid dataset user ID from 1 to 6040. The API verifies that the ID exists before the UI changes the active profile. The selected ID is stored in browser local storage and restored on refresh. Home recommendations, For You, movie feedback, profile/rating/recommendation history, and RL dashboard statistics follow the selected user. Historical ratings and saved feedback/preferences are kept distinct by user ID; invalid IDs show an inline error without switching the active profile. The legacy `demo` API profile remains supported for compatibility.

## Offline fitted Q-learning

`backend/rl/fitted_q_learning.py` provides a small ExtraTrees regression approximator and fitted-Q iteration. It is a distinct batch/offline component, not the interactive Q table and not live feedback. It fits explicit state/action/reward/next-state transitions sampled from the chronological training partition and bootstraps with the maximum prediction over a fixed pool of 64 popular training movies. This is a small action-space approximation; offline interactions are not described as online feedback.

## Evaluation and experiments

`experiments/evaluate.py` creates a global timestamp-ordered split (earlier ratings for train, later ratings for test), trains on the earlier partition only, filters each user's train-seen titles, and calculates Hit Rate@5/@10, Precision@5/@10, Recall@5/@10, catalog Coverage, and observed Average Reward. Relevance is rating ≥ 4. It saves a timestamped JSON, CSV, and comparison plot. `experiments/epsilon_experiments.py` runs a fixed-seed Q-learning replay over real user-rated MovieLens items for epsilon 1.0, 0.5, 0.1, and decaying epsilon; it saves summary/learning-curve CSVs and a plot. The replay is not a counterfactual live-user experiment. Both scripts use actual historical ratings and never synthesize MovieLens rows.

Definitions: Hit Rate@K is the share of evaluated users with at least one relevant recommendation in the top K. Precision@K is relevant retrieved items divided by K. Recall@K is relevant retrieved items divided by all held-out relevant items for a user. Coverage is recommended catalog items divided by catalog size. Average Reward maps observed ratings to −2…+2.

Actual measured comparisons and the epsilon replay results are documented in [docs/EVALUATION.md](docs/EVALUATION.md). The evaluation page reads the latest timestamped JSON result; the dashboard reads the latest epsilon experiment CSV. Q-learning did not outperform popularity on every metric in the measured split.

## Architecture

```text
React + Vite + React Router + Recharts
              │ /api (Vite proxy)
              ▼
FastAPI ── Pandas MovieLens loader
   ├── Q-learning policy ── models/q_table.json
   └── SQLite profiles, feedback, recommendation history
```

## Run locally

Backend (Python 3.10+):

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload
```

Frontend (Node.js):

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL (usually `http://localhost:5173`). API docs are at `http://localhost:8000/docs`.

## Pages

Home, Discover, Personalized For You, Movie Details with rating, Profile/history, RL Dashboard, Model Evaluation, and About. The poster art uses generated CSS fallbacks, so no TMDB key is required.

Architecture, algorithm, installation, and demonstration instructions are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/ALGORITHM.md](docs/ALGORITHM.md), [docs/INSTALLATION.md](docs/INSTALLATION.md), and [docs/DEMO_GUIDE.md](docs/DEMO_GUIDE.md).

## Project report notes

**Problem:** static popularity recommendations do not adapt to a person's changing explicit preferences. **Objective:** demonstrate how repeated user rewards update a recommendation policy. **Expected outcome:** a working feedback loop with persisted tabular values and inspectable reward/state transitions.

**Algorithm outline:** initialize Q values to zero; derive the current discrete state; select an eligible movie via epsilon-greedy; receive a 1–5 rating; map it to reward; update genre state; apply the Q-learning equation; persist both model and interaction; repeat with lower epsilon.

**Flow:** `Profile state → epsilon-greedy movie action → user rating → reward → next preference state → Q update → saved policy → next recommendation`.

**Presentation outline:** problem and objective; MovieLens dataset; state/action/reward design; Q-learning equation and exploration; application demo; dashboard; chronological evaluation; limitations and future work. Future work includes calibrated contextual states, user-controlled TMDB posters, richer terminal treatment, fitted-Q offline training pipeline and independently validated ranking experiments.

## Limitations and results

MovieLens 1M is a static historical dataset, and offline ratings do not represent live responses. Interactive preference updates use the demo account's direct ratings. Evaluation and experiments must be run locally after dataset installation; report the printed values with the split and seed. Screenshots can be added here after launching the UI.
