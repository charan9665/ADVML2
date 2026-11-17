import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"backend"))
from rl.q_learning import QLearner
from rl.reward import rating_reward
from rl.state_manager import GENRES, state_from_preferences, update_preferences
from rl.fitted_q_learning import FittedQ

class ReinforcementLearningTests(unittest.TestCase):
    def test_reward_scale(self):
        self.assertEqual([rating_reward(i) for i in range(1,6)],[-2,-1,0,1,2])

    def test_state_is_compact_and_preferences_change(self):
        self.assertEqual(state_from_preferences({}),"11111")
        before={g:.5 for g in GENRES}
        after=update_preferences(before,["Action"],2)
        self.assertEqual(after["Action"],.58)
        self.assertEqual(after["Comedy"],.5)

    def test_q_update_and_terminal(self):
        q=QLearner(alpha=.5,gamma=.9,epsilon=0)
        self.assertEqual(q.update("s","a",2,"next",[]),(0,1.0))
        # Seed a future value, then check the Bellman target includes discounting.
        q.q["next"]={"b":4}
        self.assertEqual(q.update("s","a",2,"next",["b"]),(1.0,3.3))
        self.assertEqual(q.update("terminal","x",-2,"ignored",["b"],terminal=True),(0,-1.0))

    def test_policy_reproducible(self):
        a=QLearner(seed=17); b=QLearner(seed=17)
        self.assertEqual([a.choose("s",["1","2"])[0] for _ in range(20)],[b.choose("s",["1","2"])[0] for _ in range(20)])

    def test_q_table_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"q.json"; q=QLearner(); q.update("s","m",2,"t",[],terminal=True); q.save(path)
            loaded=QLearner(); loaded.load(path)
            self.assertEqual(loaded.value("s","m"),q.value("s","m"))
            self.assertEqual(loaded.updates,1)

    def test_full_catalog_max_cache_updates_and_reloads(self):
        q=QLearner(alpha=1,gamma=.85,epsilon=0)
        q.update("next","good",4,"terminal",[],terminal=True)
        self.assertEqual(q.max_q["next"],4)
        self.assertEqual(q.update("s","a",0,"next",None),(0,3.4))
        q.update("next","good",-2,"terminal",[],terminal=True)
        self.assertEqual(q.max_q["next"],0)

    def test_fitted_q_regressor_accepts_bounded_action_pool(self):
        # Tiny numeric function-approximator fixture; it is not MovieLens data.
        import numpy as np
        states=np.full((20,5),.5); actions=np.linspace(0,1,20)
        rewards=np.tile([-1.,0.,1.,2.],5); next_states=states.copy()
        model=FittedQ()
        self.assertTrue(model.fit(states,actions,rewards,next_states,[.1,.5,.9],iterations=1))
        predictions=model.model.predict(np.column_stack([states[:2],actions[:2]]))
        self.assertEqual(predictions.shape,(2,))

if __name__=="__main__":unittest.main()
