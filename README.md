# Hybrid Recommender Engine

[![CI](https://github.com/JRA-CodingLab/hybrid-recommender-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/JRA-CodingLab/hybrid-recommender-engine/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Deployed on Render](https://img.shields.io/badge/deployed-Render-46E3B7.svg)](https://hybrid-recommender-engine.onrender.com/docs)

A production-ready hybrid product recommendation engine that blends **ALS collaborative filtering**, **association-rule mining**, and **popularity / recency signals** behind a FastAPI REST API with an interactive HTML dashboard.

## 🚀 Live Demo

**API is deployed and publicly accessible:**

- 🔗 **Swagger UI:** [hybrid-recommender-engine.onrender.com/docs](https://hybrid-recommender-engine.onrender.com/docs)
- 🎮 **Dashboard:** [hybrid-recommender-engine.onrender.com/dashboard](https://hybrid-recommender-engine.onrender.com/dashboard)

**Try it now:**
```bash
curl -X POST https://hybrid-recommender-engine.onrender.com/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": "U03", "n_items": 5}'
```

**Deployment Stack:** Docker + Render (free tier) + GitHub CI/CD auto-deploy

---

## Features

- **Collaborative Filtering** — ALS via `implicit` library
- **Association Rules** — Apriori + lift/confidence scoring via `mlxtend`
- **Hybrid Blending** — Configurable weight mix (collaborative 50% + association 30% + popularity 20%)
- **REST API** — /recommend, /similar-items, /frequently-bought-together
- **HTML Dashboard** — Interactive single-page frontend
- **Evaluation** — Precision@K, Recall@K, F1

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/recommend` | Get personalised recommendations |
| `POST` | `/similar-items` | Find similar items |
| `POST` | `/frequently-bought-together` | Complementary products |
| `GET` | `/metrics` | System & model metrics |
| `GET` | `/dashboard` | Interactive HTML dashboard |

## Tech Stack

Python 3.10+ • FastAPI • NumPy / SciPy • implicit • mlxtend • Docker • Redis (optional)

## License

[MIT](LICENSE) © 2026 Juan Ruiz Alonso
