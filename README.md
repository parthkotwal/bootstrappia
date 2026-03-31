# Bootstrappia

Investigates whether LLM-powered agents can generate synthetic interaction data to bootstrap a recommender system from cold start.

**Phase 1, Step 1** — Data & Evaluation Scaffold: establishes the warm-start ceiling (SVD trained on real data) and cold-start floor (popularity baseline) as performance anchors for later phases.

## Setup

Requires Python 3.12.1 and [uv](https://github.com/astral-sh/uv).

```bash
uv venv
uv pip install -r requirements.txt
```

## Run

```bash
uv run python scripts/run_baselines.py
```

This will:
1. Download MovieLens-100K (if not already present) to `data/ml-100k/`
2. Split users into train (60%), eval (20%), reserved (20%)
3. Train a Popularity model and an SVD model on the training split
4. Evaluate both models on the eval split
5. Save metrics to `results/baseline_results.json`

## Output

`results/baseline_results.json` contains:
- `warm_start_ceiling` — SVD metrics (the best we can do with real data)
- `cold_start_floor` — Popularity metrics (no personalization)

Synthetic agent data from later phases is measured against these two anchors.

## Metrics

| Metric | Description |
|--------|-------------|
| RMSE | Root Mean Squared Error on test ratings |
| MAE | Mean Absolute Error on test ratings |
| NDCG@10 | Ranking quality of top-10 recommendations |
| HitRate@10 | Fraction of users with a relevant item in top-10 |
