"""Lightweight offline fitted Q iteration using historical MovieLens transitions.

This is a separate batch model, never presented as live feedback or the tabular Q table.
"""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

class FittedQ:
    def __init__(self): self.model = ExtraTreesRegressor(n_estimators=40, min_samples_leaf=3, random_state=17, n_jobs=1); self.ready=False
    def fit(self, states, actions, rewards, next_states, candidate_actions, iterations=3, gamma=.85):
        if len(rewards) < 20: return False
        x = np.column_stack([states, actions]).astype(float)
        candidates=np.asarray(candidate_actions,dtype=float).reshape(-1)
        if not len(candidates): return False
        # Approximate max_a' with a fixed, train-only candidate pool rather than
        # evaluating just the observed action (which would be SARSA-like).
        expanded_states=np.repeat(np.asarray(next_states,dtype=float),len(candidates),axis=0)
        expanded_actions=np.tile(candidates,len(next_states))
        nx=np.column_stack([expanded_states,expanded_actions])
        y = np.asarray(rewards, dtype=float)
        targets = y.copy()
        for _ in range(iterations):
            self.model.fit(x, targets)
            max_next=self.model.predict(nx).reshape(len(next_states),len(candidates)).max(axis=1)
            targets = y + gamma * max_next
        self.model.fit(x, targets); self.ready=True
        return True
