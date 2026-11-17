"""Train on an earlier timestamp window and report measured ranking metrics."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"backend"))
from data_loader import load_dataset
from recommender.evaluation import chronological_split, ranking_metrics
from recommender.popularity import ranked_popular
from rl.q_learning import QLearner
from rl.reward import rating_reward
from rl.state_manager import GENRES, state_from_preferences, update_preferences
from rl.fitted_q_learning import FittedQ
import numpy as np
import random
import csv
import json
from datetime import datetime, timezone
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def prefs_vector(p): return [float(p.get(g,.5)) for g in GENRES]

def main():
    movies, ratings, _ = load_dataset()
    train,test=chronological_split(ratings)
    if test.empty or train.empty: raise SystemExit("Chronological split produced an empty partition")
    popular=[int(r["MovieID"]) for r in ranked_popular(movies,train,300)]
    seen=train.groupby("UserID").MovieID.apply(set).to_dict()
    users_train={}; q=QLearner(epsilon=0,seed=17); transitions=[]
    actions=[str(int(x)) for x in movies.MovieID]
    sample_rng=random.Random(17); transition_count=0; transition_limit=50000
    movie_genres=movies.set_index("MovieID").Genres.to_dict()
    # Replay only the train window in timestamp order to build genuine historical transitions.
    for user, group in train.sort_values("Timestamp").groupby("UserID",sort=False):
        p={g:.5 for g in GENRES}
        for row in group.itertuples(index=False):
            s=state_from_preferences(p); sv=prefs_vector(p); action=str(int(row.MovieID))
            reward=rating_reward(int(row.Rating))
            gs=str(movie_genres.get(row.MovieID,"" )).split("|")
            nxt=update_preferences(p,gs,reward); ns=state_from_preferences(nxt)
            transition_count+=1
            transition=(sv,int(row.MovieID),reward,prefs_vector(nxt))
            if len(transitions)<transition_limit: transitions.append(transition)
            else:
                slot=sample_rng.randrange(transition_count)
                if slot<transition_limit: transitions[slot]=transition
            q.update(s,action,reward,ns,None)
            p=nxt
        users_train[int(user)]=p
    # Current state after each user's observed train history; recommend unseen titles only.
    qrecs={}; fitted=FittedQ()
    for u in test.UserID.unique():
        state=state_from_preferences(users_train.get(int(u),{g:.5 for g in GENRES}))
        unseen=[mid for mid in actions if int(mid) not in seen.get(u,set())]
        qrecs[int(u)]=[int(a) for a in sorted(unseen,key=lambda a:(q.value(state,a),-int(a)),reverse=True)[:10]]
    baseline={int(u):[int(x) for x in popular if x not in seen.get(u,set())][:10] for u in test.UserID.unique()}
    # Feature approximator trained on explicit (s,a,r,s') transitions from train only.
    xstates=np.array([t[0] for t in transitions]); ids=np.array([t[1] for t in transitions],dtype=float)
    scale=max(1,int(movies.MovieID.max())); actions_norm=ids/scale
    rewards=np.array([t[2] for t in transitions]); nexts=np.array([t[3] for t in transitions])
    fit_candidates=np.array([mid/scale for mid in popular[:64]],dtype=float)
    fitted.fit(xstates,actions_norm,rewards,nexts,fit_candidates)
    f_recs={}
    for u in test.UserID.unique():
        s=np.array(prefs_vector(users_train.get(int(u),{g:.5 for g in GENRES})))
        candidates=[int(a) for a in actions if str(a) not in seen.get(u,set())]
        if fitted.ready:
            feat=np.column_stack([np.tile(s,(len(candidates),1)),np.array(candidates)/scale])
            vals=fitted.model.predict(feat)
            candidates=[int(mid) for _,mid in sorted(zip(vals,candidates),reverse=True)[:10]]
        else:candidates=[]
        f_recs[int(u)]=candidates
    models=(("Tabular Q-learning",qrecs),("Popularity baseline",baseline),("Fitted Q-learning",f_recs))
    measured=[]
    for name,recs in models:
        for k in (5,10):
            result=ranking_metrics(train,test,recs,k=k);row={"model":name,**result};measured.append(row);print(row)
    summary={"dataset":"MovieLens 1M","train_rows":len(train),"test_rows":len(test),"train_cutoff_unix":int(train.Timestamp.max()),"train_cutoff_utc":datetime.fromtimestamp(int(train.Timestamp.max()),timezone.utc).isoformat(),"q_train_transitions":transition_count,"fitted_q_transition_sample":len(transitions),"fitted_q_candidate_actions":len(fit_candidates),"relevance_threshold":4,"seed":17,"results":measured}
    out=Path(__file__).resolve().parent/"results";out.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    json_path=out/f"evaluation_{stamp}.json";csv_path=out/f"evaluation_{stamp}.csv";plot_path=out/f"evaluation_{stamp}.png"
    json_path.write_text(json.dumps(summary,indent=2),encoding="utf-8")
    with csv_path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=measured[0].keys());writer.writeheader();writer.writerows(measured)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,k in zip(axes,(5,10)):
        rows=[r for r in measured if r["k"]==k];x=np.arange(len(rows));w=.2
        for j,metric in enumerate(("hit_rate","precision","recall","coverage")):
            ax.bar(x+(j-1.5)*w,[r[metric] for r in rows],w,label=metric.replace("_"," "))
        ax.set_xticks(x,[r["model"].replace(" ","\n") for r in rows]);ax.set_title(f"Ranking metrics @ {k}");ax.grid(axis="y",alpha=.2)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(plot_path,dpi=160);plt.close(fig)
    print("Saved measured results:",json_path,csv_path,plot_path)
    print("Relevance threshold: rating >= 4; train-only historical transitions; global timestamp split; Q random seed 17.")

if __name__=="__main__":
    try: main()
    except FileNotFoundError as exc: raise SystemExit(f"Cannot run evaluation: {exc}")
