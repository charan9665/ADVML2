"""MovieMind API: original historical data plus persistent interactive Q-learning."""
from contextlib import asynccontextmanager
from pathlib import Path
import csv, json, os, sqlite3
from datetime import datetime, timezone
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from data_loader import load_dataset, DATA
from database import initialize, connect, feedback_count, watched_ids, profile, save_profile
from rl.q_learning import QLearner
from rl.reward import rating_reward
from rl.state_manager import GENRES, state_from_preferences, update_preferences
from recommender.popularity import ranked_popular

MODEL=Path(os.getenv("MOVIEMIND_MODEL_PATH",str(Path(__file__).resolve().parents[1]/"models"/"q_table.json")))
EXPERIMENTS=Path(__file__).resolve().parents[1]/"experiments"/"results"
agent=QLearner(); agent.load(MODEL)
movies=pd.DataFrame(); ratings=pd.DataFrame(); users=pd.DataFrame(); catalog=pd.DataFrame(); dataset_error=None

@asynccontextmanager
async def lifespan(app):
    global movies,ratings,users,catalog,dataset_error
    initialize()
    try:
        movies,ratings,users=load_dataset()
        catalog=pd.DataFrame()
        catalog_rows()
    except Exception as e: dataset_error=str(e)
    yield

app=FastAPI(title="MovieMind API",version="1.0.0",lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=["http://localhost:5173"],allow_methods=["*"],allow_headers=["*"])

class Feedback(BaseModel):
    user_id:str=Field(min_length=1,max_length=80)
    movie_id:int=Field(gt=0)
    rating:int=Field(ge=1,le=5)

def ready():
    if movies.empty: raise HTTPException(503,detail=f"MovieLens data is unavailable: {dataset_error or 'no rows loaded'}")

def user_preferences(user):
    default={g:.5 for g in GENRES}
    if user.isdigit() and not movies.empty:
        history=ratings[ratings.UserID==int(user)]
        if not history.empty:
            for genre in GENRES:
                ids=movies[movies.Genres.str.contains(genre,regex=False)].MovieID
                values=history[history.MovieID.isin(ids)].Rating
                if len(values):default[genre]=float((values-1).mean()/4)
    return profile(user,default)

def watched_for_user(user):
    seen=watched_ids(user)
    if user.isdigit() and not ratings.empty:
        seen.update(int(mid) for mid in ratings.loc[ratings.UserID==int(user),"MovieID"])
    return seen

def movie_record(row):
    if hasattr(row,"_asdict"):
        row=row._asdict()
    if isinstance(row,(dict,pd.Series)):
        get=row.get
        avg=get("mean",0); count=get("count",0)
        return {"movie_id":int(get("MovieID")),"title":get("Title"),"year":int(get("Year")) if pd.notna(get("Year")) else None,"genres":get("GenreList",[]),"average_rating":round(float(avg),2) if pd.notna(avg) else None,"rating_count":int(count) if pd.notna(count) else 0}
    return {"movie_id":int(row.MovieID),"title":row.Title,"year":int(row.Year) if pd.notna(row.Year) else None,"genres":row.GenreList if isinstance(row.GenreList,list) else [],"average_rating":round(float(getattr(row,"mean",0)),2) if pd.notna(getattr(row,"mean",0)) else None,"rating_count":int(getattr(row,"count",0))}

def movie_by_id(mid):
    r=catalog_rows()
    r=r[r.MovieID==mid]
    if r.empty: raise HTTPException(404,detail="Movie not found")
    return movie_record(r.iloc[0])

def catalog_rows():
    global catalog
    if catalog.empty and not movies.empty:
        stats=ratings.groupby("MovieID").Rating.agg(["count","mean"]).reset_index()
        catalog=movies.merge(stats,on="MovieID",how="left").fillna({"count":0,"mean":0})
    return catalog

def recommend(user,limit=12,record=True):
    ready(); pref=user_preferences(user); state=state_from_preferences(pref); seen=watched_for_user(user)
    rows=catalog_rows(); rows=rows[~rows.MovieID.isin(seen)].copy()
    if rows.empty:return []
    # Cap each request's action set to the 600 most popular unseen titles to keep it tractable.
    rows=rows.sort_values(["count","mean"],ascending=False).head(600)
    actions=[str(int(x)) for x in rows.MovieID]
    action_scores={str(int(row.MovieID)):sum(pref.get(g,.5) for g in row.GenreList)/max(1,len(row.GenreList)) for row in rows.itertuples(index=False)}
    chosen=[]; out=[]
    for _ in range(min(limit,len(actions))):
        available=[a for a in actions if a not in chosen]
        if not available:break
        action,mode=agent.choose(state,available,action_scores=action_scores)
        chosen.append(action); mid=int(action); row=rows[rows.MovieID==mid].iloc[0]
        m=movie_record(row); m.update({"mode":mode,"explanation":f"{mode.title()} choice for preference state {state}"})
        out.append(m)
        if record:
            with connect() as c:c.execute("INSERT INTO recommendations(user_id,movie_id,mode,state) VALUES(?,?,?,?)",(user,mid,mode,state))
    return out

@app.get("/api/health")
def health():return {"status":"ok","dataset_loaded":not movies.empty,"dataset_error":dataset_error}

@app.get("/api/dataset/summary")
def dataset_summary():
    ready()
    genre_counts={}
    for values in movies.GenreList:
        for genre in values:genre_counts[genre]=genre_counts.get(genre,0)+1
    means=ratings.groupby("UserID").Rating.mean()
    popularity=ratings.groupby("MovieID").size()
    return {"source":"MovieLens 1M","users":int(users.UserID.nunique()),"movies":int(movies.MovieID.nunique()),"ratings":int(len(ratings)),"rating_mean":float(ratings.Rating.mean()),"rating_min":int(ratings.Rating.min()),"rating_max":int(ratings.Rating.max()),"average_ratings_per_user":float(means.mean()),"most_rated_movie_count":int(popularity.max()),"genre_movie_counts":genre_counts,"inspection":movies.attrs.get("inspection",{})}

@app.get("/api/movies")
def get_movies(search:str="",genre:str="",year:int|None=None,sort:str="popularity",limit:int=Query(30,ge=1,le=100),offset:int=0):
    ready(); df=catalog_rows()
    if search:df=df[df.Title.str.contains(search,case=False,regex=False)]
    if genre:df=df[df.Genres.str.contains(genre,case=False,regex=False)]
    if year:df=df[df.Title.str.endswith(f"({year})")]
    col="mean" if sort=="rating" else "count"; df=df.sort_values([col,"count"],ascending=False).iloc[offset:offset+limit]
    return [movie_record(r) for r in df.itertuples(index=False)]

@app.get("/api/movies/popular")
def popular(limit:int=Query(20,ge=1,le=100)):
    ready(); return [movie_record(r) for r in ranked_popular(movies,ratings,limit)]

@app.get("/api/movies/trending")
def trending(limit:int=Query(20,ge=1,le=100),window_days:int=Query(90,ge=7,le=3650)):
    ready()
    cutoff=int(ratings["Timestamp"].max())-window_days*24*60*60
    recent=ratings[ratings["Timestamp"]>=cutoff]
    if recent.empty:return []
    stats=recent.groupby("MovieID").Rating.agg(["count","mean"]).reset_index()
    rows=movies.merge(stats,on="MovieID",how="inner").sort_values(["count","mean"],ascending=False).head(limit)
    return [movie_record(r) for r in rows.to_dict("records")]

@app.get("/api/movies/{movie_id}")
def get_movie(movie_id:int):ready();return movie_by_id(movie_id)

@app.get("/api/users")
def get_users(limit:int=Query(30,ge=1,le=100)):
    ready(); return [{"user_id":int(r.UserID),"gender":r.Gender,"age":int(r.Age),"occupation":int(r.Occupation)} for r in users.head(limit).itertuples(index=False)]

@app.get("/api/users/{user_id}/profile")
def get_profile(user_id:str):
    prefs=user_preferences(user_id); count=feedback_count(user_id)
    rating_values=[]
    if not movies.empty and user_id.isdigit():
        ur=ratings[ratings.UserID==int(user_id)]; count+=len(ur); rating_values.extend(ur.Rating.astype(float).tolist())
    with connect() as c: rating_values.extend(float(r[0]) for r in c.execute("SELECT rating FROM feedback WHERE user_id=?",(user_id,)))
    return {"user_id":user_id,"preferences":prefs,"state":state_from_preferences(prefs),"rated_count":count,"average_rating":sum(rating_values)/len(rating_values) if rating_values else None,"favorite_genres":sorted(prefs,key=prefs.get,reverse=True)[:3]}

@app.get("/api/users/{user_id}/history")
def history(user_id:str,limit:int=Query(40,ge=1,le=100)):
    ready()
    if not user_id.isdigit():
        with connect() as c: rows=c.execute("SELECT movie_id,rating,reward,created_at FROM feedback WHERE user_id=? ORDER BY id DESC LIMIT ?",(user_id,limit)).fetchall()
        return [{"movie":movie_by_id(r[0]),"rating":r[1],"reward":r[2],"created_at":r[3]} for r in rows]
    h=ratings[ratings.UserID==int(user_id)].sort_values("Timestamp",ascending=False).head(limit)
    return [{"movie":movie_by_id(int(r.MovieID)),"rating":int(r.Rating),"timestamp":int(r.Timestamp)} for r in h.itertuples(index=False)]

@app.get("/api/recommendations/{user_id}")
def recommendations(user_id:str,limit:int=Query(12,ge=1,le=50)):return recommend(user_id,limit)

@app.post("/api/feedback")
def feedback(body:Feedback):
    ready()
    if movies[movies.MovieID==body.movie_id].empty:raise HTTPException(404,detail="Movie not found")
    with connect() as c:
        if c.execute("SELECT 1 FROM feedback WHERE user_id=? AND movie_id=?",(body.user_id,body.movie_id)).fetchone():raise HTTPException(409,detail="This user has already rated this movie")
    pref=user_preferences(body.user_id); state=state_from_preferences(pref); reward=rating_reward(body.rating)
    genres=movie_by_id(body.movie_id)["genres"]; next_pref=update_preferences(pref,genres,reward); next_state=state_from_preferences(next_pref)
    available=[str(int(x)) for x in catalog_rows().MovieID if int(x) not in watched_for_user(body.user_id) and int(x)!=body.movie_id]
    old,new=agent.update(state,str(body.movie_id),reward,next_state,available,terminal=not available)
    save_profile(body.user_id,next_pref); agent.save(MODEL)
    try:
        with connect() as c:c.execute("INSERT INTO feedback(user_id,movie_id,rating,reward,state,next_state,old_q,new_q) VALUES(?,?,?,?,?,?,?,?)",(body.user_id,body.movie_id,body.rating,reward,state,next_state,old,new))
    except sqlite3.IntegrityError:raise HTTPException(409,detail="Feedback already exists")
    return {"reward":reward,"state":state,"next_state":next_state,"old_q":old,"new_q":new,"preferences":next_pref,"message":"Feedback learned and Q-table saved"}

@app.get("/api/rl/status")
def rl_status(user_id:str="demo"):
    prefs=user_preferences(user_id)
    with connect() as c:
        interactions=c.execute("SELECT COUNT(*) n FROM feedback").fetchone()["n"]
        recs=c.execute("SELECT COUNT(*) n FROM recommendations").fetchone()["n"]
        rewards=c.execute("SELECT AVG(reward) a FROM feedback").fetchone()["a"]
        modes=c.execute("SELECT mode,COUNT(*) n FROM recommendations GROUP BY mode").fetchall()
        history=c.execute("SELECT reward,created_at,old_q,new_q FROM feedback ORDER BY id").fetchall()
    return {"alpha":agent.alpha,"gamma":agent.gamma,"epsilon":agent.epsilon,"interactions":interactions,"recommendations":recs,"average_reward":rewards or 0,"q_states":len(agent.q),"q_entries":sum(map(len,agent.q.values())),"state":state_from_preferences(prefs),"preferences":prefs,"exploration":next((r["n"] for r in modes if r["mode"]=="exploration"),0),"exploitation":next((r["n"] for r in modes if r["mode"]=="exploitation"),0),"reward_history":[{"episode":i+1,"reward":r["reward"],"date":r["created_at"]} for i,r in enumerate(history)],"q_history":[{"episode":i+1,"old_q":r["old_q"],"new_q":r["new_q"]} for i,r in enumerate(history)]}

@app.get("/api/rl/q-table")
def q_table(state:str|None=None,limit:int=Query(100,ge=1,le=500)):
    data=agent.q if state is None else {state:agent.q.get(state,{})}
    return {"states":{s:dict(sorted(row.items(),key=lambda x:x[1],reverse=True)[:limit]) for s,row in list(data.items())[:limit]},"total_states":len(agent.q)}

@app.get("/api/rl/experiments")
def experiments():
    files=sorted(EXPERIMENTS.glob("epsilon_experiments_*.csv"),key=lambda p:p.stat().st_mtime,reverse=True)
    if not files:return {"status":"not_computed","seed":17,"message":"Run experiments/epsilon_experiments.py with MovieLens installed."}
    with files[0].open(encoding="utf-8",newline="") as f: rows=list(csv.DictReader(f))
    for row in rows:
        for key in ("initial_epsilon","epsilon_decay","minimum_epsilon","average_reward","exploration_rate","exploitation_rate","final_epsilon"):
            row[key]=float(row[key])
        for key in ("steps_per_user","episodes","seed","users"):row[key]=int(row[key])
    return {"status":"complete","results":rows,"method":"offline replay of observed MovieLens user-item ratings; not counterfactual live response"}

@app.get("/api/evaluation")
def evaluation():
    ready()
    files=sorted(EXPERIMENTS.glob("evaluation_*.json"),key=lambda p:p.stat().st_mtime,reverse=True)
    if not files:return {"status":"not_computed","message":"Run python experiments/evaluate.py to calculate chronological holdout metrics from this dataset."}
    return json.loads(files[0].read_text(encoding="utf-8"))

@app.get("/api/evaluation/comparison")
def comparison():return evaluation()

@app.post("/api/rl/train")
def train():
    ready()
    from recommender.evaluation import chronological_split
    from rl.reward import rating_reward
    from rl.state_manager import update_preferences
    training,holdout=chronological_split(ratings)
    offline=QLearner(alpha=agent.alpha,gamma=agent.gamma,epsilon=0.0,epsilon_decay=1.0,min_epsilon=0.0,seed=17)
    genres=movies.set_index("MovieID").Genres.to_dict(); transitions=0
    for _,group in training.sort_values("Timestamp").groupby("UserID",sort=False):
        preferences={g:.5 for g in GENRES}
        for row in group.itertuples(index=False):
            state=state_from_preferences(preferences); reward=rating_reward(int(row.Rating))
            next_preferences=update_preferences(preferences,str(genres.get(row.MovieID,"" )).split("|"),reward)
            next_state=state_from_preferences(next_preferences)
            offline.update(state,str(int(row.MovieID)),reward,next_state,None)
            preferences=next_preferences;transitions+=1
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    artifact=Path(__file__).resolve().parents[1]/"models"/f"offline_q_table_{stamp}.json"
    offline.save(artifact)
    metadata={"train_interactions":transitions,"holdout_interactions":len(holdout),"cutoff_unix":int(training.Timestamp.max()),"q_states":len(offline.q),"q_entries":sum(map(len,offline.q.values())),"seed":17,"artifact":str(artifact.relative_to(Path(__file__).resolve().parents[1]))}
    artifact.with_suffix(".metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    return {"status":"complete","mode":"offline_historical_training","live_agent_unchanged":True,**metadata}
