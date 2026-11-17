"""Reward mapping used for interactive and offline learning."""
RATING_REWARDS = {1: -2.0, 2: -1.0, 3: 0.0, 4: 1.0, 5: 2.0}

def rating_reward(rating: int) -> float:
    if rating not in RATING_REWARDS:
        raise ValueError("rating must be an integer from 1 to 5")
    return RATING_REWARDS[rating]
