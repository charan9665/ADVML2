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
        user="integration-demo"
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
        self.assertEqual(restored.value(event["state"],str(selected["movie_id"])),event["new_q"])
        next_picks=self.client.get(f"/api/recommendations/{user}?limit=20").json()
        self.assertNotIn(selected["movie_id"],[m["movie_id"] for m in next_picks])
        history=self.client.get(f"/api/users/{user}/history").json()
        self.assertEqual(len(history),1)
        self.assertEqual(history[0]["rating"],5)

if __name__=="__main__":unittest.main()
