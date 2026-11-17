"""Real-catalog feedback integration test with isolated persistence artifacts."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"backend"))
import database,main
from data_loader import DATA
from fastapi.testclient import TestClient
from rl.q_learning import QLearner

@unittest.skipUnless(all((DATA/name).exists() for name in ("movies.dat","ratings.dat","users.dat")),"Install official MovieLens 1M files to run API integration")
class FeedbackIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="moviemind-api-test-")
        self.old_db,self.old_model,self.old_agent,self.old_catalog=database.DB_PATH,main.MODEL,main.agent,main.catalog
        database.DB_PATH=Path(self.temp.name)/"test.sqlite3"
        main.MODEL=Path(self.temp.name)/"q_table.json"
        main.agent=QLearner(seed=17)
        main.catalog=main.pd.DataFrame()
        self.client=TestClient(main.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None,None,None)
        database.DB_PATH=self.old_db
        main.MODEL=self.old_model;main.agent=self.old_agent;main.catalog=self.old_catalog
        self.temp.cleanup()

    def test_feedback_updates_persists_and_excludes_movie(self):
        user="demo"
        picks=self.client.get(f"/api/recommendations/{user}?limit=10").json()
        self.assertTrue(picks)
        selected=picks[0]
        response=self.client.post("/api/feedback",json={"user_id":user,"movie_id":selected["movie_id"],"rating":5})
        self.assertEqual(response.status_code,200,response.text)
        event=response.json()
        self.assertEqual(event["reward"],2.0)
        self.assertGreater(event["new_q"],event["old_q"])
        self.assertTrue(main.MODEL.exists())
        restored=QLearner();restored.load(main.MODEL)
        self.assertEqual(restored.value(main.q_state_key(user,event["state"]),str(selected["movie_id"])),event["new_q"])
        next_picks=self.client.get(f"/api/recommendations/{user}?limit=20").json()
        self.assertNotIn(selected["movie_id"],[m["movie_id"] for m in next_picks])
        history=self.client.get(f"/api/users/{user}/history").json()
        self.assertEqual(len(history),1)
        self.assertEqual(history[0]["rating"],5)

    def test_movie_lens_profiles_are_isolated_and_validated(self):
        main.agent.q["legacy-state"]={"77":0.42}
        main.agent.max_q["legacy-state"]=0.42
        profile_one=self.client.get("/api/users/1/profile")
        profile_hundred=self.client.get("/api/users/100/profile")
        self.assertEqual(profile_one.status_code,200)
        self.assertEqual(profile_hundred.status_code,200)
        self.assertNotEqual(profile_one.json()["rated_count"],profile_hundred.json()["rated_count"])
        self.assertNotEqual(profile_one.json()["state"],profile_hundred.json()["state"])

        history_one=self.client.get("/api/users/1/history?limit=100").json()
        history_hundred=self.client.get("/api/users/100/history?limit=100").json()
        ids_one={row["movie"]["movie_id"] for row in history_one}
        ids_hundred={row["movie"]["movie_id"] for row in history_hundred}
        self.assertNotEqual(ids_one,ids_hundred)

        before_q=dict(main.agent.q)
        picks_one=self.client.get("/api/recommendations/1?limit=10").json()
        self.assertTrue(picks_one)
        self.assertFalse(ids_one.intersection(row["movie_id"] for row in picks_one))
        picked=picks_one[0]["movie_id"]
        learned=self.client.post("/api/feedback",json={"user_id":"1","movie_id":picked,"rating":5})
        self.assertEqual(learned.status_code,200,learned.text)
        self.assertTrue(any(key.startswith("user:1:") for key in main.agent.q))
        self.assertEqual(main.agent.q.get("legacy-state",{}),before_q.get("legacy-state",{}))
        user_q=self.client.get("/api/rl/q-table?user_id=1").json()["states"]
        other_q=self.client.get("/api/rl/q-table?user_id=100").json()["states"]
        self.assertTrue(user_q)
        self.assertFalse(other_q)

        own_history=self.client.get("/api/users/1/history?limit=100").json()
        other_history=self.client.get("/api/users/100/history?limit=100").json()
        self.assertIn(picked,{row["movie"]["movie_id"] for row in own_history})
        self.assertNotIn(picked,{row["movie"]["movie_id"] for row in other_history})
        self.assertEqual(self.client.get("/api/users/1/recommendation-history").status_code,200)

        picks_hundred=self.client.get("/api/recommendations/100?limit=10").json()
        self.assertTrue(picks_hundred)
        self.assertNotEqual([x["movie_id"] for x in picks_one],[x["movie_id"] for x in picks_hundred])
        self.assertFalse(ids_hundred.intersection(row["movie_id"] for row in picks_hundred))
        self.assertEqual(self.client.get("/api/rl/status?user_id=1").json()["interactions"],1)
        self.assertEqual(self.client.get("/api/rl/status?user_id=100").json()["interactions"],0)

        for invalid in ("0","6041","99999","abc"):
            self.assertIn(self.client.get(f"/api/users/{invalid}/profile").status_code,(404,422))
            self.assertIn(self.client.get(f"/api/recommendations/{invalid}").status_code,(404,422))

if __name__=="__main__":unittest.main()
