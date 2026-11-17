"""Leakage-conscious chronological holdout evaluation."""
import pandas as pd

def ranking_metrics(train, test, recs_by_user, k=10, threshold=4.0):
    relevant = {u:set(g.loc[g.Rating >= threshold,"MovieID"]) for u,g in test.groupby("UserID")}
    users = list(relevant)
    if not users: return {"users":0,"k":k,"hit_rate":0,"precision":0,"recall":0,"coverage":0,"average_reward":0}
    catalog=set(train.MovieID.unique()); hits=prec=rec=reward=0; shown=set(); count=0
    for u in users:
        recs=list(recs_by_user.get(int(u), []))[:k]; shown.update(recs)
        hits_u=len(set(recs)&relevant[u]); hits += int(hits_u>0)
        prec += hits_u/k; rec += hits_u/max(1,len(relevant[u])); count += len(recs)
        reward += sum((2 if r>=5 else 1 if r>=4 else 0 if r>=3 else -1 if r>=2 else -2) for r in test[(test.UserID==u)&(test.MovieID.isin(recs))].Rating)
    return {"users":len(users),"k":k,"hit_rate":hits/len(users),"precision":prec/len(users),"recall":rec/len(users),"coverage":len(shown&catalog)/max(1,len(catalog)),"average_reward":reward/len(users)}

def chronological_split(ratings, test_fraction=.2):
    ordered=ratings.sort_values(["Timestamp","UserID"])
    cutoff=ordered.Timestamp.quantile(1-test_fraction)
    return ordered[ordered.Timestamp<=cutoff].copy(), ordered[ordered.Timestamp>cutoff].copy()
