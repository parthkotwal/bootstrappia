"""Main pipeline: load data, split, train, evaluate, save results."""

import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import load_data
from src.splitter import split_users
from src.models.popularity import PopularityModel
from src.models.svd_model import SVDModel
from src.evaluate import evaluate_model
from src.utils import save_json

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "baseline_results.json")


def main():
    print("=" * 60)
    print("Bootstrappia — Phase 1, Step 1: Baseline Pipeline")
    print("=" * 60)

    # 1. Load data
    print("\n[Step 1] Loading data ...")
    ratings, items = load_data()

    # 2. Split users
    print("\n[Step 2] Splitting users ...")
    split = split_users(ratings)
    train_ratings = split["train_ratings"]
    eval_profile = split["eval_profile_ratings"]
    eval_test = split["eval_test_ratings"]

    n_train_users = train_ratings["user_id"].nunique()
    n_eval_users = eval_test["user_id"].nunique()

    # 3. Popularity model — cold-start floor
    print("\n[Step 3] Training Popularity model ...")
    pop_model = PopularityModel()
    pop_model.fit(train_ratings)
    pop_metrics = evaluate_model(pop_model, eval_profile, eval_test, model_type="popularity")

    # 4. SVD model — warm-start ceiling
    print("\n[Step 4] Training SVD model ...")
    svd_model = SVDModel()
    svd_model.fit(train_ratings)
    svd_metrics = evaluate_model(svd_model, eval_profile, eval_test, model_type="svd")

    # 5. Save results
    results = {
        "dataset": "movielens-100k",
        "split_seed": 42,
        "n_train_users": n_train_users,
        "n_eval_users": n_eval_users,
        "warm_start_ceiling": {
            "model": "SVD",
            **svd_metrics,
        },
        "cold_start_floor": {
            "model": "Popularity",
            **pop_metrics,
        },
    }

    print("\n[Step 5] Saving results ...")
    save_json(results, RESULTS_PATH)

    # 6. Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Train users:  {n_train_users}")
    print(f"  Eval users:   {n_eval_users}")
    print()
    print("  Warm-start ceiling (SVD):")
    print(f"    RMSE:         {svd_metrics['rmse']:.4f}")
    print(f"    MAE:          {svd_metrics['mae']:.4f}")
    print(f"    NDCG@10:      {svd_metrics['ndcg_at_10']:.4f}")
    print(f"    HitRate@10:   {svd_metrics['hit_rate_at_10']:.4f}")
    print()
    print("  Cold-start floor (Popularity):")
    print(f"    RMSE:         {pop_metrics['rmse']:.4f}")
    print(f"    MAE:          {pop_metrics['mae']:.4f}")
    print(f"    NDCG@10:      {pop_metrics['ndcg_at_10']:.4f}")
    print(f"    HitRate@10:   {pop_metrics['hit_rate_at_10']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
