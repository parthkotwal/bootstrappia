"""
SVD matrix factorization model — warm-start ceiling.

For evaluation users not seen in training, surprise uses baseline estimates
(global mean + item bias) since user bias is unknown. This is documented
behavior: predictions for unknown users fall back to item-level signal,
which still captures the item quality learned from 60% of users.
"""

import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.prediction_algorithms.predictions import Prediction

SVD_PARAMS = dict(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
RATING_SCALE = (1, 5)


def _build_surprise_dataset(ratings: pd.DataFrame) -> Dataset:
    reader = Reader(rating_scale=RATING_SCALE)
    return Dataset.load_from_df(ratings[["user_id", "item_id", "rating"]], reader)


class SVDModel:
    """
    SVD matrix factorization via scikit-surprise.

    For unknown evaluation users, surprise returns baseline estimates
    (global mean + item bias) with user bias set to 0. This captures
    item-level signal from training data without requiring per-user retraining.
    """

    def __init__(self):
        self.algo: SVD | None = None
        self.all_item_ids: list[int] = []
        self.global_mean: float = 0.0

    def fit(self, train_ratings: pd.DataFrame) -> None:
        """Fit SVD on training data."""
        self.all_item_ids = train_ratings["item_id"].unique().tolist()
        self.global_mean = train_ratings["rating"].mean()

        dataset = _build_surprise_dataset(train_ratings)
        trainset = dataset.build_full_trainset()

        self.algo = SVD(**SVD_PARAMS)
        self.algo.fit(trainset)

        print(f"SVDModel fitted on {len(train_ratings)} ratings, {len(self.all_item_ids)} items.")
        print(
            "  Note: predictions for eval users (not in training) use baseline estimates "
            "(global mean + item bias, user bias=0)."
        )

    def predict(
        self,
        user_id: int,
        item_id: int,
        profile_ratings: pd.DataFrame | None = None,
    ) -> float:
        """Predict rating for (user, item). Unknown users get baseline estimates."""
        if self.algo is None:
            raise RuntimeError("Model not fitted.")
        pred: Prediction = self.algo.predict(str(user_id), str(item_id))
        return float(pred.est)

    def recommend(
        self,
        user_id: int,
        already_rated_items: set[int],
        profile_ratings: pd.DataFrame | None = None,
        n: int = 10,
    ) -> list[int]:
        """Return top-N item IDs by predicted score, excluding already-rated items."""
        if self.algo is None:
            raise RuntimeError("Model not fitted.")
        candidates = [iid for iid in self.all_item_ids if iid not in already_rated_items]
        scored = [
            (iid, self.algo.predict(str(user_id), str(iid)).est)
            for iid in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [iid for iid, _ in scored[:n]]
