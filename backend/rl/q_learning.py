"""Small from-scratch tabular Q learner; actions are movie IDs."""
import json, random
from pathlib import Path

class QLearner:
    def __init__(self, alpha=.15, gamma=.85, epsilon=.25, epsilon_decay=.995, min_epsilon=.03, seed=17):
        self.alpha, self.gamma = alpha, gamma
        self.epsilon, self.epsilon_decay, self.min_epsilon = epsilon, epsilon_decay, min_epsilon
        self.rng = random.Random(seed)
        self.q: dict[str, dict[str, float]] = {}
        self.max_q: dict[str, float] = {}
        self.updates = 0

    def value(self, state: str, action: str) -> float:
        return self.q.get(state, {}).get(action, 0.0)

    def choose(self, state: str, actions: list[str], explore=True, action_scores: dict[str, float] | None = None):
        if not actions: return None, "none"
        is_exploration = explore and self.rng.random() < self.epsilon
        if is_exploration:
            return self.rng.choice(actions), "exploration"
        best = max(self.value(state, a) for a in actions)
        choices = [a for a in actions if self.value(state, a) == best]
        # Use the current preference state as a deterministic tie-breaker for unseen actions.
        if action_scores:
            highest = max(action_scores.get(a, 0.0) for a in choices)
            choices = [a for a in choices if action_scores.get(a, 0.0) == highest]
        return self.rng.choice(choices), "exploitation"

    def update(self, state: str, action: str, reward: float, next_state: str, next_actions: list[str] | set[str] | None, terminal=False):
        row = self.q.setdefault(state, {})
        old = row.get(action, 0.0)
        if terminal or next_actions == [] or next_actions == set():
            future = 0.0
        elif next_actions is None:
            # `None` denotes the full catalog action set, with unseen actions at zero.
            future = self.max_q.get(next_state, 0.0)
        else:
            eligible = next_actions if isinstance(next_actions, set) else set(next_actions)
            next_row = self.q.get(next_state, {})
            known = [v for a, v in next_row.items() if a in eligible]
            # Uninitialized actions have value zero; include that value without
            # scanning the full candidate catalog on every update.
            future = max(known + ([0.0] if len(known) < len(eligible) else []))
        target = reward + self.gamma * future
        row[action] = old + self.alpha * (target - old)
        current_max = self.max_q.get(state, 0.0)
        if row[action] >= current_max:
            self.max_q[state] = row[action]
        elif old >= current_max and row[action] < old:
            self.max_q[state] = max([0.0, *row.values()])
        self.updates += 1
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return old, row[action]

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"q": self.q, "epsilon": self.epsilon, "updates": self.updates}, indent=2), encoding="utf-8")

    def load(self, path: Path):
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self.q = data.get("q", {}); self.epsilon = data.get("epsilon", self.epsilon); self.updates = data.get("updates", 0)
            self.max_q = {state:max([0.0,*row.values()]) for state,row in self.q.items()}
