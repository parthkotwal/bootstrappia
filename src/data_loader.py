"""Download and load MovieLens-100K dataset."""

import os
import zipfile
import requests
import pandas as pd

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
ML_DIR = os.path.join(DATA_DIR, "ml-100k")

GENRE_LIST = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def download_data() -> None:
    """Download and extract MovieLens-100K if not already present."""
    if os.path.exists(os.path.join(ML_DIR, "u.data")):
        print("Data already downloaded, skipping.")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "ml-100k.zip")

    print(f"Downloading MovieLens-100K from {DATA_URL} ...")
    response = requests.get(DATA_URL, stream=True)
    response.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete. Extracting ...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)

    os.remove(zip_path)
    print(f"Extracted to {ML_DIR}")


def load_ratings() -> pd.DataFrame:
    """Load u.data and return a DataFrame with user_id, item_id, rating, timestamp."""
    path = os.path.join(ML_DIR, "u.data")
    ratings = pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={"user_id": int, "item_id": int, "rating": float, "timestamp": int},
    )
    return ratings


def load_items() -> pd.DataFrame:
    """Load u.item and return a DataFrame with item_id, title, genres (list)."""
    path = os.path.join(ML_DIR, "u.item")
    # u.item has 24 fields; we only need first 3 + 19 genre flags
    col_names = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRE_LIST
    items = pd.read_csv(
        path,
        sep="|",
        names=col_names,
        encoding="latin-1",
        usecols=["item_id", "title"] + GENRE_LIST,
    )

    def row_to_genres(row):
        return [g for g in GENRE_LIST if row[g] == 1]

    items["genres"] = items.apply(row_to_genres, axis=1)
    return items[["item_id", "title", "genres"]]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download if needed, then load and return (ratings, items)."""
    download_data()
    ratings = load_ratings()
    items = load_items()

    n_users = ratings["user_id"].nunique()
    n_items = ratings["item_id"].nunique()
    n_ratings = len(ratings)
    sparsity = 1.0 - n_ratings / (n_users * n_items)

    print(f"\n=== Dataset Stats ===")
    print(f"  Users:    {n_users}")
    print(f"  Items:    {n_items}")
    print(f"  Ratings:  {n_ratings}")
    print(f"  Sparsity: {sparsity:.4%}")

    return ratings, items
