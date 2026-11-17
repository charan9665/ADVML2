# MovieMind Learning Algorithm

## Interactive tabular Q-learning

The profile state is five genre-affinity values (Action, Comedy, Romance, Drama, Sci-Fi), each quantized into 0/1/2. The maximum state count is 243. An action is a movie ID chosen from at most 600 popular, unseen candidates. Exploration chooses uniformly at epsilon; exploitation selects a maximum-Q action and uses affinity to break ties.

Rating rewards are 1→−2, 2→−1, 3→0, 4→+1, 5→+2. For each genre present in the movie, affinity changes by `0.08 × reward / 2`, clipped to [0,1]. The next state is the new ternary encoding. The update is:

```text
Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]
```

When no eligible next action exists, the future term is zero. Uninitialized action values are zero. The interactive agent updates only when the feedback endpoint accepts a user's rating. Its table, epsilon, and update count are persisted to JSON.

## Historical transitions

For each user's earlier chronological ratings, the offline replay begins at neutral affinity and applies each observed rating in timestamp order. Every record becomes `(state, movie_id, mapped_reward, next_state)`. The global split uses a timestamp quantile; no held-out rows enter training. Offline training writes a separate artifact so it cannot silently replace the interactive policy.

## Fitted Q iteration

The fitted component approximates `Q(s,a)` with an ExtraTrees regressor over five affinity features and a normalized movie-ID action feature. At each fitted iteration, targets use rewards plus the maximum predicted next value across a fixed set of 64 movies popular in the training window. It is an approximation to the full next-action maximization, not a production-scale fitted-Q implementation. Its training data are historical offline rows, not demo-user ratings.

## Exploration replay caveat

The epsilon experiment provides each sampled user a pool of titles they actually rated and uses each selected title's recorded rating as reward. It compares 20 policy choices per sampled user. The simulator can compare policy behavior on known outcomes, but cannot estimate how a user would respond to an unseen recommendation.
