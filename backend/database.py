"""Small SQLite persistence layer, using only the Python standard library."""
import sqlite3
import os
from contextlib import contextmanager
from pathlib import Path
DB_PATH=Path(os.getenv("MOVIEMIND_DB_PATH",str(Path(__file__).resolve().parents[1]/"moviemind.sqlite3")))

@contextmanager
def connect():
    con=sqlite3.connect(DB_PATH); con.row_factory=sqlite3.Row
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

def initialize():
    with connect() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS profiles(user_id TEXT PRIMARY KEY, preferences TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS feedback(id INTEGER PRIMARY KEY, user_id TEXT, movie_id INTEGER, rating INTEGER, reward REAL, state TEXT, next_state TEXT, old_q REAL, new_q REAL, created_at TEXT DEFAULT CURRENT_TIMESTAMP, UNIQUE(user_id,movie_id));
        CREATE TABLE IF NOT EXISTS recommendations(id INTEGER PRIMARY KEY, user_id TEXT, movie_id INTEGER, mode TEXT, state TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
        """)

def feedback_count(user):
    with connect() as c: return c.execute("SELECT COUNT(*) n FROM feedback WHERE user_id=?",(user,)).fetchone()["n"]

def watched_ids(user):
    with connect() as c: return {r[0] for r in c.execute("SELECT movie_id FROM feedback WHERE user_id=?",(user,))}

def profile(user, default):
    import json
    with connect() as c: row=c.execute("SELECT preferences FROM profiles WHERE user_id=?",(user,)).fetchone()
    return json.loads(row[0]) if row else default

def save_profile(user, prefs):
    import json
    with connect() as c: c.execute("INSERT INTO profiles VALUES(?,?) ON CONFLICT(user_id) DO UPDATE SET preferences=excluded.preferences",(user,json.dumps(prefs)))
