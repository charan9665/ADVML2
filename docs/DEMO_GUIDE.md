# MovieMind Demo Guide

1. Start FastAPI and Vite using the commands in `INSTALLATION.md`.
2. Open `http://127.0.0.1:5173` and check that the API reports MovieLens loaded.
3. Use the profile dropdown in the top-right header. Enter `1` or `100` and choose **Apply**. The user's profile, recommendation list, and history refresh together.
4. Refresh the page to demonstrate that the selected ID is restored from browser local storage.
5. Open **For you**; recommendation cards show whether selection was exploration or exploitation and exclude movies already rated by that user.
6. Open a recommended movie, select a 1–5 star rating, and submit. The feedback is saved against the active MovieLens user ID.
7. Verify the confirmation shows reward, old/new state, and old/new Q value. The page shows refreshed recommendations and excludes the rated title.
8. Open **Profile** and **RL dashboard** to inspect this user's history, preferences, interaction count, reward history, policy counts, and Q-table size.
9. Open **Evaluation** for the measured chronological comparison after running the evaluation script.
10. Restart FastAPI and inspect `/api/rl/q-table?user_id=1` to demonstrate that this user's interactive Q values persist.

The first visit defaults to MovieLens User 1. The selector accepts any existing ID from 1 through 6040; invalid IDs leave the current profile selected. Historical ratings initialize each user's genre preferences. Interactive feedback, recommendation history, saved preferences, and Q-state keys are associated with that user's ID. The legacy `demo` API profile remains available for compatibility.
