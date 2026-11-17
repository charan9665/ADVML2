"""Compact discrete state from genre feedback preferences."""
GENRES = ("Action", "Comedy", "Romance", "Drama", "Sci-Fi")

def state_from_preferences(preferences: dict[str, float]) -> str:
    # Five ternary bins give 3**5=243 possible states; values are normalized 0..1.
    levels = []
    for genre in GENRES:
        value = float(preferences.get(genre, 0.5))
        levels.append("0" if value < 0.34 else "1" if value < 0.67 else "2")
    return "".join(levels)

def update_preferences(preferences: dict[str, float], genres: list[str], reward: float) -> dict[str, float]:
    updated = {g: float(preferences.get(g, 0.5)) for g in GENRES}
    # Bounded incremental update; positive feedback raises affinity, negative lowers it.
    delta = 0.08 * (reward / 2.0)
    for genre in genres:
        if genre in updated:
            updated[genre] = min(1.0, max(0.0, updated[genre] + delta))
    return updated
