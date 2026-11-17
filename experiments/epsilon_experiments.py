"""Seeded Q-learning replay experiment over actual MovieLens rating outcomes.

This is an offline replay simulator: it only chooses historical items with an
observed rating for that user. It is not a counterfactual live-user study.
"""
import csv
import random
import sys
from pathlib import Path
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"backend"))
from data_loader import load_dataset
from rl.q_learning import QLearner
from rl.reward import rating_reward
from rl.state_manager import GENRES, state_from_preferences, update_preferences

SEED=17
USER_SAMPLE=1000
REPLAY_STEPS_PER_USER=20
CURVE_STEP=1000

def run_policy(ratings, movie_genres, selected_users, name, epsilon, decay, minimum):
    agent=QLearner(epsilon=epsilon,epsilon_decay=decay,min_epsilon=minimum,seed=SEED)
    total=explorations=exploitation=episodes=0
    curve=[]; reward_bucket=0; bucket_n=0
    sample=ratings[ratings.UserID.isin(selected_users)].sort_values(["Timestamp","UserID"])
    for user, group in sample.groupby("UserID",sort=True):
        preferences={g:.5 for g in GENRES}
        available=[]; observed={}
        for row in group.itertuples(index=False):
            action=str(int(row.MovieID))
            if action not in observed:
                available.append(action)
                observed[action]=(int(row.Rating),str(movie_genres.get(int(row.MovieID),"" )).split("|"))
        # Replay each user-rated movie at most once; selection changes with the learned Q table.
        user_steps=0
        while available and user_steps<REPLAY_STEPS_PER_USER:
            state=state_from_preferences(preferences)
            action,mode=agent.choose(state,available)
            idx=available.index(action)
            available[idx]=available[-1]
            available.pop()
            rating,genres=observed.pop(action)
            reward=rating_reward(rating)
            preferences=update_preferences(preferences,genres,reward)
            next_state=state_from_preferences(preferences)
            agent.update(state,action,reward,next_state,None,terminal=not available)
            episodes+=1; total+=reward; reward_bucket+=reward; bucket_n+=1
            user_steps+=1
            if mode=="exploration": explorations+=1
            else: exploitation+=1
            if bucket_n==CURVE_STEP:
                curve.append({"strategy":name,"episode":episodes,"average_reward":reward_bucket/bucket_n})
                reward_bucket=bucket_n=0
    if bucket_n:
        curve.append({"strategy":name,"episode":episodes,"average_reward":reward_bucket/bucket_n})
    return {"strategy":name,"initial_epsilon":epsilon,"epsilon_decay":decay,"minimum_epsilon":minimum,"steps_per_user":REPLAY_STEPS_PER_USER,"episodes":episodes,"average_reward":total/episodes if episodes else 0.0,"exploration_rate":explorations/episodes if episodes else 0.0,"exploitation_rate":exploitation/episodes if episodes else 0.0,"final_epsilon":agent.epsilon,"seed":SEED,"users":len(selected_users)},curve

def main():
    movies,ratings,_=load_dataset()
    rng=random.Random(SEED)
    user_ids=sorted(int(x) for x in ratings.UserID.unique())
    selected_users=set(rng.sample(user_ids,min(USER_SAMPLE,len(user_ids))))
    movie_genres=movies.set_index("MovieID").Genres.to_dict()
    configs=[("epsilon_1.0",1.0,1.0,1.0),("epsilon_0.5",.5,1.0,.5),("epsilon_0.1",.1,1.0,.1),("epsilon_decay",1.0,.99995,.03)]
    results=[]; curves=[]
    for name,eps,decay,minimum in configs:
        result,curve=run_policy(ratings,movie_genres,selected_users,name,eps,decay,minimum)
        results.append(result);curves.extend(curve)
        print(result)
    out=ROOT/"experiments"/"results";out.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    summary_path=out/f"epsilon_experiments_{stamp}.csv"
    curve_path=out/f"epsilon_learning_curve_{stamp}.csv"
    plot_path=out/f"epsilon_learning_curve_{stamp}.png"
    with summary_path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=results[0].keys());writer.writeheader();writer.writerows(results)
    with curve_path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=curves[0].keys());writer.writeheader();writer.writerows(curves)
    plt.figure(figsize=(10,5))
    for name,_,_,_ in configs:
        points=[r for r in curves if r["strategy"]==name]
        plt.plot([r["episode"] for r in points],[r["average_reward"] for r in points],label=name)
    plt.xlabel("Replay episode");plt.ylabel(f"Mean reward per {CURVE_STEP}-episode window")
    plt.title("MovieLens offline Q-learning replay by epsilon strategy");plt.legend();plt.grid(alpha=.25);plt.tight_layout()
    plt.savefig(plot_path,dpi=160);plt.close()
    print("Saved measured results:",summary_path,curve_path,plot_path)
    print("Interpret results as historical-rating replay only; selection bias means this is not counterfactual recommendation quality.")

if __name__=="__main__":
    try: main()
    except FileNotFoundError as exc: raise SystemExit(f"Cannot run epsilon experiment: {exc}")
