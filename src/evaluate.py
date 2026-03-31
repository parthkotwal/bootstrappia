"""Metric computation: RMSE, MAE, NDCG@10, HitRate@10."""

import math
import numpy as np
import pandas as pd
from typing import Any


def _dcg(relevances: list[int]) -> float:
    """Compute DCG for a ranked list of binary relevance scores."""
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevances))


def _ndcg_at_k(recommended: list[int], relevant_items: set[int], k: int = 10) -> float:
    """Compute NDCG@k. Relevance = 1 if item in relevant_items, else 0."""
    top_k = recommended[:k]
    relevances = [1 if iid in relevant_items else 0 for iid in top_k]
    ideal = sorted(relevances, reverse=True)
    dcg = _dcg(relevances)
    idcg = _dcg(ideal)
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_model(
    model: Any,
    eval_profile_ratings: pd.DataFrame,
    eval_test_ratings: pd.DataFrame,
    model_type: str = "popularity",
) -> dict:
    """
    Evaluate a model against evaluation users.

    For each evaluation user:
      - Profile ratings are given to the model.
      - Predictions are made for test set items.
      - A top-10 recommendation list is generated.

    Returns dict with rmse, mae, ndcg_at_10, hit_rate_at_10.
    """
    eval_users = eval_test_ratings["user_id"].unique()
    print(f"\nEvaluating {model_type} on {len(eval_users)} evaluation users ...")

    squared_errors = []
    abs_errors = []
    ndcg_scores = []
    hits = []

    skipped = 0
    for uid in eval_users:
        profile = eval_profile_ratings[eval_profile_ratings["user_id"] == uid]
        test = eval_test_ratings[eval_test_ratings["user_id"] == uid]

        if len(test) == 0:
            skipped += 1
            continue

        already_rated = set(profile["item_id"].tolist())
        relevant_items = set(test[test["rating"] >= 4]["item_id"].tolist())

        # Rating predictions for test items
        for _, row in test.iterrows():
            if model_type == "svd":
                pred = model.predict(uid, int(row["item_id"]), profile_ratings=profile)
            else:
                pred = model.predict(uid, int(row["item_id"]))
            actual = row["rating"]
            squared_errors.append((pred - actual) ** 2)
            abs_errors.append(abs(pred - actual))

        # Top-N recommendations
        if model_type == "svd":
            recs = model.recommend(uid, already_rated, profile_ratings=profile, n=10)
        else:
            recs = model.recommend(uid, already_rated, n=10)

        ndcg_scores.append(_ndcg_at_k(recs, relevant_items, k=10))
        hit = 1 if any(iid in relevant_items for iid in recs) else 0
        hits.append(hit)

    if skipped > 0:
        print(f"  Skipped {skipped} users with empty test sets.")

    metrics = {
        "rmse": float(np.sqrt(np.mean(squared_errors))) if squared_errors else float("nan"),
        "mae": float(np.mean(abs_errors)) if abs_errors else float("nan"),
        "ndcg_at_10": float(np.mean(ndcg_scores)) if ndcg_scores else float("nan"),
        "hit_rate_at_10": float(np.mean(hits)) if hits else float("nan"),
    }

    print(f"  RMSE:         {metrics['rmse']:.4f}")
    print(f"  MAE:          {metrics['mae']:.4f}")
    print(f"  NDCG@10:      {metrics['ndcg_at_10']:.4f}")
    print(f"  HitRate@10:   {metrics['hit_rate_at_10']:.4f}")

    return metrics
