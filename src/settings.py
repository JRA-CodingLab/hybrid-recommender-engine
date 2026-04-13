"""Application configuration via dataclass with automatic directory setup."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class EngineSettings:
    """Central configuration for the hybrid recommender engine."""

    # --- data paths ---
    RAW_DATA_PATH: str = os.path.join("data", "raw")
    PROCESSED_DATA_PATH: str = os.path.join("data", "processed")
    MODEL_SAVE_PATH: str = os.path.join("models", "saved")

    # --- ALS hyper-parameters ---
    ALS_FACTORS: int = 50
    ALS_REGULARIZATION: float = 0.1
    ALS_ITERATIONS: int = 20
    ALS_ALPHA: float = 40.0

    # --- recommendation defaults ---
    TOP_K_RECOMMENDATIONS: int = 20
    SIMILAR_ITEMS_COUNT: int = 10
    MIN_INTERACTIONS_THRESHOLD: int = 1

    # --- hybrid weights (must sum to ≈1.0) ---
    COLLABORATIVE_WEIGHT: float = 0.5
    POPULARITY_WEIGHT: float = 0.2
    ASSOCIATION_WEIGHT: float = 0.2
    RECENCY_WEIGHT: float = 0.1

    # --- Redis (optional caching) ---
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_TTL: int = 3600

    # --- API ---
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False

    # --- metrics ---
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090

    def __post_init__(self) -> None:
        for directory in (
            self.RAW_DATA_PATH,
            self.PROCESSED_DATA_PATH,
            self.MODEL_SAVE_PATH,
        ):
            os.makedirs(directory, exist_ok=True)


# Module-level singleton
config = EngineSettings()
