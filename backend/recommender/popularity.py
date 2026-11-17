def ranked_popular(movies, ratings, limit=20):
    if ratings.empty: return []
    stats = ratings.groupby("MovieID").Rating.agg(["count", "mean"]).reset_index()
    # Bayesian shrinkage reduces small-sample popularity spikes.
    global_mean = float(ratings.Rating.mean()); prior = 25
    stats["score"] = (stats["count"] / (stats["count"] + prior) * stats["mean"] + prior / (stats["count"] + prior) * global_mean)
    joined = movies.merge(stats, on="MovieID", how="inner").sort_values(["score", "count"], ascending=False)
    return joined.head(limit).to_dict("records")
