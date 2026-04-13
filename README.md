# Hybrid Recommender Engine

[![CI](https://github.com/JRA-CodingLab/hybrid-recommender-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/JRA-CodingLab/hybrid-recommender-engine/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A production-ready hybrid product recommendation engine that blends **ALS collaborative filtering**, **association-rule mining**, and **popularity / recency signals** behind a FastAPI REST API with optional Redis caching and an interactive HTML dashboard.

---

## Features

- **Collaborative Filtering** — Alternating Least Squares via the `implicit` library
- **Association Rules** — Apriori + lift/confidence scoring via `mlxtend`
- **Hybrid Blending** — Configurable weight mix with business-rule overlays (out-of-stock, margin boost, seasonal boost)
- **Recency Decay** — Exponential time weighting so recent interactions count more
- **REST API** — FastAPI with health, recommend, similar-items, frequently-bought-together, train, evaluate, and metrics endpoints
- **Caching** — Redis when available, transparent in-memory fallback
- **Interactive Dashboard** — Single-page HTML/JS frontend
- **Evaluation** — Precision@K, Recall@K, F1 on held-out data

## Quick Start

### 1. Install

```bash
pip install -e ".[dev]"
```

### 2. Run the API

```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

### 3. Train models

```bash
python train.py --data-path data/raw/interactions.csv
```

Or via API:

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"data_path": "data/raw/interactions.csv"}'
```

### 4. Get recommendations

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_42", "user_history": ["item_1", "item_5"], "n_items": 10}'
```

### 5. Dashboard

Open `http://localhost:8000/dashboard` in your browser.

## Input Data Format

CSV with columns:

| Column | Description |
|--------|-------------|
| `user_id` | Unique user identifier |
| `item_id` | Unique item identifier |
| `interaction_type` | One of: `purchase`, `add_to_cart`, `wishlist`, `view`, `click` |
| `timestamp` | ISO 8601 datetime |

## Architecture

```
CSV → DataProcessor → Sparse Matrix + Mappings
                          ↓
         ALSEngine (collaborative filtering)
         BasketRules (association mining)
                          ↓
         BlendRecommender (hybrid combiner)
                          ↓
         FastAPI REST API → HTML Dashboard
```

## Configuration

All settings live in `src/settings.py` as a Python dataclass. Key parameters:

| Setting | Default | Description |
|---------|---------|-------------|
| `ALS_FACTORS` | 50 | Latent factor dimensionality |
| `ALS_ITERATIONS` | 20 | Training iterations |
| `COLLABORATIVE_WEIGHT` | 0.5 | Collaborative filtering weight |
| `POPULARITY_WEIGHT` | 0.2 | Popularity signal weight |
| `ASSOCIATION_WEIGHT` | 0.2 | Association rules weight |
| `RECENCY_WEIGHT` | 0.1 | Recency signal weight |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/recommend` | Get personalised recommendations |
| `POST` | `/similar-items` | Find similar items |
| `POST` | `/frequently-bought-together` | Complementary products |
| `POST` | `/train` | Trigger model training |
| `GET` | `/metrics` | System & model metrics |
| `POST` | `/evaluate` | Evaluate on test data |
| `GET` | `/dashboard` | Interactive HTML dashboard |

## Testing

```bash
pytest -v --cov=src
```

## License

[MIT](LICENSE) © 2026 Juan Ruiz Alonso
