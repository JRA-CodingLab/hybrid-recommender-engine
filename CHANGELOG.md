# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-04-13

### Added

- ALS collaborative filtering engine (`ALSEngine`) using the implicit library
- Association-rule mining (`BasketRules`) with Apriori via mlxtend
- Hybrid blending recommender (`BlendRecommender`) with configurable weights
- Business rule overlays: out-of-stock filtering, margin boost, seasonal boost
- Recency-weighted interaction matrix with exponential decay
- FastAPI REST API with health, recommend, similar-items, frequently-bought-together, train, evaluate, and metrics endpoints
- Background training pipeline with session inference
- Optional Redis caching with transparent in-memory fallback
- Interactive HTML dashboard
- CLI training script
- Dataclass-based configuration with automatic directory creation
- Unit tests for settings, data processor, and blend recommender
- CI workflow with GitHub Actions
- Full project documentation (README, CONTRIBUTING, LICENSE)
