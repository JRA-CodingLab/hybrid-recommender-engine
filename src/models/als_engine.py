"""ALS collaborative filtering powered by the *implicit* library."""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix

from ..settings import EngineSettings

logger = logging.getLogger(__name__)


class ALSEngine:
    """Train and query an Alternating-Least-Squares matrix factorisation model."""

    def __init__(self, settings: EngineSettings) -> None:
        self.settings = settings
        self.model: Optional[Any] = None
        self.interaction_matrix: Optional[csr_matrix] = None
        self.user_mapping: Dict[Any, int] = {}
        self.item_mapping: Dict[Any, int] = {}
        self.reverse_user_mapping: Dict[int, Any] = {}
        self.reverse_item_mapping: Dict[int, Any] = {}
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        interaction_matrix: csr_matrix,
        user_mapping: Dict[Any, int],
        item_mapping: Dict[Any, int],
    ) -> None:
        """Fit the ALS model on a user×item interaction matrix."""
        from implicit.als import AlternatingLeastSquares

        self.interaction_matrix = interaction_matrix
        self.user_mapping = user_mapping
        self.item_mapping = item_mapping
        self.reverse_user_mapping = {v: k for k, v in user_mapping.items()}
        self.reverse_item_mapping = {v: k for k, v in item_mapping.items()}

        self.model = AlternatingLeastSquares(
            factors=self.settings.ALS_FACTORS,
            regularization=self.settings.ALS_REGULARIZATION,
            iterations=self.settings.ALS_ITERATIONS,
            random_state=42,
            use_gpu=False,
        )

        # implicit expects an item-user matrix
        item_user = interaction_matrix.T.tocsr()
        self.model.fit(item_user)

        self.user_factors = self.model.user_factors
        self.item_factors = self.model.item_factors
        logger.info(
            "ALS trained — factors=%d, iterations=%d",
            self.settings.ALS_FACTORS,
            self.settings.ALS_ITERATIONS,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def recommend_for_user(
        self,
        user_id: Any,
        user_history: List[Any],
        n_items: int = 10,
    ) -> List[Tuple[Any, float]]:
        """Return top-*n* items for a known user, excluding already-seen."""
        if self.model is None or self.interaction_matrix is None:
            return []

        user_idx = self.user_mapping.get(user_id)
        if user_idx is None:
            return []

        user_items = self.interaction_matrix[user_idx]
        ids, scores = self.model.recommend(
            user_idx, user_items, N=n_items, filter_already_liked_items=True
        )

        results: List[Tuple[Any, float]] = []
        for item_idx, score in zip(ids, scores):
            original_id = self.reverse_item_mapping.get(int(item_idx))
            if original_id is not None:
                results.append((original_id, float(score)))
        return results

    def get_similar_items(
        self, item_id: Any, n_items: int = 10
    ) -> List[Tuple[Any, float]]:
        """Find the most similar items to the given one."""
        if self.model is None:
            return []

        item_idx = self.item_mapping.get(item_id)
        if item_idx is None:
            return []

        ids, scores = self.model.similar_items(item_idx, N=n_items + 1)

        results: List[Tuple[Any, float]] = []
        for idx, score in zip(ids, scores):
            if int(idx) == item_idx:
                continue
            original_id = self.reverse_item_mapping.get(int(idx))
            if original_id is not None:
                results.append((original_id, float(score)))
        return results[:n_items]

    def get_popular_items(self, n_items: int = 10) -> List[Tuple[Any, float]]:
        """Return items ranked by aggregate factor magnitude."""
        if self.item_factors is None:
            return []

        factor_norms = np.linalg.norm(self.item_factors, axis=1)
        top_indices = np.argsort(factor_norms)[::-1][:n_items]

        results: List[Tuple[Any, float]] = []
        for idx in top_indices:
            original_id = self.reverse_item_mapping.get(int(idx))
            if original_id is not None:
                results.append((original_id, float(factor_norms[idx])))
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path: Optional[str] = None) -> None:
        path = path or os.path.join(self.settings.MODEL_SAVE_PATH, "als_model.pkl")
        payload = {
            "model": self.model,
            "user_factors": self.user_factors,
            "item_factors": self.item_factors,
            "user_mapping": self.user_mapping,
            "item_mapping": self.item_mapping,
            "reverse_user_mapping": self.reverse_user_mapping,
            "reverse_item_mapping": self.reverse_item_mapping,
            "interaction_matrix": self.interaction_matrix,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        logger.info("ALS model saved to %s", path)

    def load_model(self, path: Optional[str] = None) -> None:
        path = path or os.path.join(self.settings.MODEL_SAVE_PATH, "als_model.pkl")
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        self.model = payload["model"]
        self.user_factors = payload["user_factors"]
        self.item_factors = payload["item_factors"]
        self.user_mapping = payload["user_mapping"]
        self.item_mapping = payload["item_mapping"]
        self.reverse_user_mapping = payload["reverse_user_mapping"]
        self.reverse_item_mapping = payload["reverse_item_mapping"]
        self.interaction_matrix = payload["interaction_matrix"]
        logger.info("ALS model loaded from %s", path)
