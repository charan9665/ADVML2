# MovieMind Demo Guide

1. Start FastAPI and Vite using the commands in `INSTALLATION.md`.
2. Open `http://127.0.0.1:5173` and check that the API reports MovieLens loaded.
3. Open **For you** with profile ID `demo`; recommendation cards show whether selection was exploration or exploitation.
4. Open a movie card, select a 1–5 star rating, and submit.
5. Verify the confirmation shows reward, old/new state, and old/new Q value. The page should show refreshed recommendations and exclude the rated title.
6. Open **Profile** and **RL dashboard** to inspect the saved history, preferences, interaction count, reward history, and policy counts.
7. Open **Evaluation** for the measured chronological comparison after running the evaluation script.
8. Restart FastAPI and inspect `/api/rl/q-table` to demonstrate that interactive Q values persist.

Use the `demo` profile for live ratings. Numeric MovieLens IDs display historical profiles and histories; their previously rated catalog movies are excluded from recommendations.
