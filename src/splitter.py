"""User-based train/test splitting for MovieLens."""

from typing import TypedDict
import numpy as np
import pandas as pd

SEED = 42
MIN_EVAL_RATINGS = 10


class Split(TypedDict):
    train_ratings: pd.DataFrame
    eval_profile_ratings: pd.DataFrame
    eval_test_ratings: pd.DataFrame
    reserved_ratings: pd.DataFrame


def split_users(ratings: pd.DataFrame) -> Split:
    """
    Split users into three groups:
      - 60% warm-start training users
      - 20% evaluation users (profile + test split chronologically)
      - 20% reserved users (for later phases)

    Evaluation users with fewer than MIN_EVAL_RATINGS are moved to reserved.
    Uses fixed random seed 42 for reproducibility.
    """
    rng = np.random.default_rng(SEED)
    all_users = ratings["user_id"].unique()
    shuffled = rng.permutation(all_users)

    n = len(shuffled)
    n_train = int(n * 0.6)
    n_eval = int(n * 0.2)

    train_users = set(shuffled[:n_train])
    eval_users_raw = set(shuffled[n_train : n_train + n_eval])
    reserved_users = set(shuffled[n_train + n_eval :])

    # For evaluation users, check minimum rating count
    user_rating_counts = ratings[ratings["user_id"].isin(eval_users_raw)].groupby("user_id").size()
    sparse_eval_users = set(user_rating_counts[user_rating_counts < MIN_EVAL_RATINGS].index)
    eval_users = eval_users_raw - sparse_eval_users
    reserved_users = reserved_users | sparse_eval_users

    # Build training set
    train_ratings = ratings[ratings["user_id"].isin(train_users)].copy()

    # For each evaluation user, split their ratings chronologically 50/50
    profile_frames = []
    test_frames = []

    for uid in eval_users:
        user_ratings = ratings[ratings["user_id"] == uid].sort_values("timestamp")
        n_user = len(user_ratings)
        cutoff = n_user // 2
        profile_frames.append(user_ratings.iloc[:cutoff])
        test_frames.append(user_ratings.iloc[cutoff:])

    eval_profile_ratings = pd.concat(profile_frames, ignore_index=True) if profile_frames else pd.DataFrame(columns=ratings.columns)
    eval_test_ratings = pd.concat(test_frames, ignore_index=True) if test_frames else pd.DataFrame(columns=ratings.columns)
    reserved_ratings = ratings[ratings["user_id"].isin(reserved_users)].copy()

    print(f"\n=== Split Stats ===")
    print(f"  Train users:    {len(train_users):4d}  |  ratings: {len(train_ratings)}")
    print(f"  Eval users:     {len(eval_users):4d}  |  profile: {len(eval_profile_ratings)}, test: {len(eval_test_ratings)}")
    print(f"  Reserved users: {len(reserved_users):4d}  |  ratings: {len(reserved_ratings)}")
    print(f"  ({len(sparse_eval_users)} eval users moved to reserved for having < {MIN_EVAL_RATINGS} ratings)")

    return Split(
        train_ratings=train_ratings,
        eval_profile_ratings=eval_profile_ratings,
        eval_test_ratings=eval_test_ratings,
        reserved_ratings=reserved_ratings,
    )
