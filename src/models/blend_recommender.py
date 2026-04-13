"""Hybrid recommender that blends collaborative, association, and popularity signals."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..settings import EngineSettings
from .als_engine import ALSEngine
from .basket_rules import BasketRules

logger = logging.getLogger(__name__)


class BlendRecommender:
    """Combine multiple recommendation strategies with configurable weights,
    business-rule overlays, and transparent caching."""

    def __init__(
        self,
        settings: EngineSettings,
        als_model: ALSEngine,
        association_model: BasketRules,
    ) -> None:
        self.settings = settings
        self.als = als_model
        self.association = association_model

        # Caching layer — Redis if reachable, else in-process dict
        self._cache: Optional[Any] = None
        self._dict_cache: Dict[str, Any] = {}
        self._init_cache()

        # Basic telemetry
        self.total_served: int = 0
        self.cache_hits: int = 0

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _init_cache(self) -> None:
        try:
            import redis

            r = redis.Redis(
                host=self.settings.REDIS_HOST,
                port=self.settings.REDIS_PORT,
                db=self.settings.REDIS_DB,
            )
            r.ping()
            self._cache = r
            logger.info("Redis cache connected.")
        except Exception:
            self._cache = None
            logger.info("Redis unavailable — falling back to in-memory cache.")

    def _cache_key(self, user_id: Any, user_history: List[Any]) -> str:
        raw = f"{user_id}:{','.join(sorted(str(i) for i in user_history))}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _get_cached(self, key: str) -> Optional[List[Dict[str, Any]]]:
        if self._cache is not None:
            data = self._cache.get(key)
            if data:
                return json.loads(data)
        return self._dict_cache.get(key)

    def _set_cached(self, key: str, value: List[Dict[str, Any]]) -> None:
        serialised = json.dumps(value)
        if self._cache is not None:
            self._cache.setex(key, self.settings.REDIS_TTL, serialised)
        else:
            self._dict_cache[key] = value

    # ------------------------------------------------------------------
    # Core recommendation
    # ------------------------------------------------------------------

    def get_recommendations(
        self,
        user_id: Any,
        user_history: List[Any],
        context: Optional[Dict[str, Any]] = None,
        n_items: int = 10,
    ) -> List[Dict[str, Any]]:
        """Produce ranked recommendations by blending multiple signals."""
        self.total_served += 1

        cache_key = self._cache_key(user_id, user_history)
        cached = self._get_cached(cache_key)
        if cached is not None:
            self.cache_hits += 1
            return cached[:n_items]

        scores: Dict[Any, float] = defaultdict(float)
        reasons: Dict[Any, List[str]] = defaultdict(list)

        # 1. Collaborative filtering
        collab_recs = self.als.recommend_for_user(
            user_id, user_history, n_items=n_items * 2
        )
        for item_id, score in collab_recs:
            scores[item_id] += score * self.settings.COLLABORATIVE_WEIGHT
            reasons[item_id].append("collaborative_filtering")

        # 2. Similar items based on recent history
        for hist_item in user_history[-5:]:
            similar = self.als.get_similar_items(hist_item, n_items=5)
            for item_id, score in similar:
                scores[item_id] += score * 0.2
                reasons[item_id].append("similar_to_history")

        # 3. Frequently bought together
        fbt = self.association.get_frequently_bought_together(
            user_history, n_items=n_items
        )
        for entry in fbt:
            scores[entry["item_id"]] += (
                entry["score"] * self.settings.ASSOCIATION_WEIGHT
            )
            reasons[entry["item_id"]].append("frequently_bought_together")

        # 4. Popularity signal
        popular = self.als.get_popular_items(n_items=n_items)
        for item_id, score in popular:
            scores[item_id] += score * self.settings.POPULARITY_WEIGHT
            reasons[item_id].append("popularity")

        # Remove items already in history
        for h in user_history:
            scores.pop(h, None)

        # Apply business rules
        scores = self._apply_business_rules(scores, context)

        # Sort and format
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_items]
        results = [
            {
                "item_id": item_id,
                "score": round(score, 4),
                "reason": list(set(reasons.get(item_id, ["popularity"]))),
            }
            for item_id, score in ranked
        ]

        self._set_cached(cache_key, results)
        return results

    # ------------------------------------------------------------------
    # Business rules
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_business_rules(
        scores: Dict[Any, float],
        context: Optional[Dict[str, Any]],
    ) -> Dict[Any, float]:
        """Overlay domain heuristics onto raw scores."""
        if not context:
            return scores

        out_of_stock = set(context.get("out_of_stock", []))
        high_margin = set(context.get("high_margin_items", []))
        seasonal = set(context.get("seasonal_items", []))
        seasonal_months: List[int] = context.get("seasonal_months", [])
        current_month = datetime.now(tz=None).month

        filtered: Dict[Any, float] = {}
        for item_id, score in scores.items():
            if item_id in out_of_stock:
                continue
            if item_id in high_margin:
                score *= 1.2
            if item_id in seasonal and current_month in seasonal_months:
                score *= 1.3
            filtered[item_id] = score
        return filtered

    # ------------------------------------------------------------------
    # Explanations
    # ------------------------------------------------------------------

    def get_explanation(
        self,
        user_id: Any,
        item_id: Any,
        user_history: List[Any],
    ) -> List[Dict[str, Any]]:
        """Build a human-readable explanation for why an item was recommended."""
        explanations: List[Dict[str, Any]] = []

        # Collaborative signal
        collab = self.als.recommend_for_user(user_id, user_history, n_items=50)
        collab_ids = {i: s for i, s in collab}
        if item_id in collab_ids:
            explanations.append(
                {
                    "type": "collaborative_filtering",
                    "description": "Users with similar preferences also interacted with this item.",
                    "confidence": round(collab_ids[item_id], 4),
                }
            )

        # Similarity to history
        for hist_item in user_history[-5:]:
            similar = self.als.get_similar_items(hist_item, n_items=20)
            similar_ids = {i: s for i, s in similar}
            if item_id in similar_ids:
                explanations.append(
                    {
                        "type": "item_similarity",
                        "description": f"This item is similar to '{hist_item}' in your history.",
                        "confidence": round(similar_ids[item_id], 4),
                    }
                )
                break

        # Association rule
        fbt = self.association.get_frequently_bought_together(
            user_history, n_items=20
        )
        fbt_ids = {e["item_id"]: e["score"] for e in fbt}
        if item_id in fbt_ids:
            explanations.append(
                {
                    "type": "frequently_bought_together",
                    "description": "Customers who bought items in your history also bought this.",
                    "confidence": round(fbt_ids[item_id], 4),
                }
            )

        if not explanations:
            explanations.append(
                {
                    "type": "popularity",
                    "description": "This item is popular across all users.",
                    "confidence": 0.5,
                }
            )

        return explanations

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def cache_hit_rate(self) -> float:
        if self.total_served == 0:
            return 0.0
        return round(self.cache_hits / self.total_served, 4)
