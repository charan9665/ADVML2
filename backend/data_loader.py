"""Load the original MovieLens 1M .dat files (never synthesize training data)."""
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"ml-1m"

def load_dataset(data_dir=DATA):
    p=Path(data_dir)
    required=[p/"movies.dat",p/"ratings.dat",p/"users.dat"]
    missing=[f.name for f in required if not f.exists()]
    if missing: raise FileNotFoundError(f"MovieLens 1M files missing in {p}: {', '.join(missing)}. Download and extract ml-1m.zip from GroupLens.")
    movies=pd.read_csv(required[0],sep="::",engine="python",encoding="latin-1",names=["MovieID","Title","Genres"])
    ratings=pd.read_csv(required[1],sep="::",engine="python",names=["UserID","MovieID","Rating","Timestamp"])
    users=pd.read_csv(required[2],sep="::",engine="python",names=["UserID","Gender","Age","Occupation","Zip-code"],encoding="latin-1")
    inspection={}
    for name,frame in (("movies",movies),("ratings",ratings),("users",users)):
        inspection[name]={"rows_before_deduplication":len(frame),"duplicate_rows":int(frame.duplicated().sum()),"missing_cells":int(frame.isna().sum().sum())}
        frame.drop_duplicates(inplace=True)
    ratings["TimestampDate"]=pd.to_datetime(ratings.Timestamp,unit="s",utc=True)
    movies["GenreList"]=movies.Genres.fillna("").map(lambda x:[g for g in x.split("|") if g and g!="(no genres listed)"])
    movies["Year"]=movies.Title.str.extract(r"\((\d{4})\)\s*$")[0]
    movies.attrs["inspection"]=inspection
    return movies,ratings,users
