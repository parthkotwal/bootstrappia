"""Popularity baseline model — cold-start floor."""

import pandas as pd
import numpy as np

MIN_RATINGS_THRESHOLD = 5


class PopularityModel:
    """
    Non-personalized model that predicts item average ratings.
    Recommendations are ranked by Bayesian average (smoothed by global mean).
    """

    def __init__(self):
        self.item_stats: pd.DataFrame | None = None
        self.global_mean: float = 0.0
        self.all_item_ids: list[int] = []

    def fit(self, train_ratings: pd.DataFrame) -> None:
        """Compute per-item average rating and global mean."""
        self.global_mean = train_ratings["rating"].mean()
        self.all_item_ids = train_ratings["item_id"].unique().tolist()

        stats = train_ratings.groupby("item_id")["rating"].agg(["mean", "count"]).reset_index()
        stats.columns = ["item_id", "avg_rating", "n_ratings"]

        # Bayesian average: (C * m + sum_ratings) / (C + n_ratings)
        # where m = global mean, C = MIN_RATINGS_THRESHOLD
        C = MIN_RATINGS_THRESHOLD
        m = self.global_mean
        stats["score"] = (C * m + stats["avg_rating"] * stats["n_ratings"]) / (C + stats["n_ratings"])

        self.item_stats = stats.set_index("item_id")
        print(f"PopularityModel fitted on {len(train_ratings)} ratings, {len(self.item_stats)} items.")

    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating for (user, item). Returns item avg or global mean if unseen."""
        if self.item_stats is None:
            raise RuntimeError("Model not fitted.")
        if item_id in self.item_stats.index:
            return float(self.item_stats.loc[item_id, "avg_rating"])
        return self.global_mean

    def recommend(self, user_id: int, already_rated_items: set[int], n: int = 10) -> list[int]:
        """Return top-N item IDs by Bayesian score, excluding already-rated items."""
        if self.item_stats is None:
            raise RuntimeError("Model not fitted.")
        candidates = self.item_stats[~self.item_stats.index.isin(already_rated_items)]
        top = candidates.nlargest(n, "score")
        return top.index.tolist()
