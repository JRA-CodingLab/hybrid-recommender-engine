"""Association-rule mining for frequently-bought-together recommendations."""

from __future__ import annotations

import logging
import os
import pickle
from collections import defaultdict
from typing import Any, Dict, List, Optional

import pandas as pd

from ..settings import EngineSettings

logger = logging.getLogger(__name__)


class BasketRules:
    """Apriori-based association rule miner for transaction baskets."""

    def __init__(self, settings: EngineSettings) -> None:
        self.settings = settings
        self.rules: Optional[pd.DataFrame] = None
        self.antecedent_map: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    @staticmethod
    def prepare_basket_data(transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Convert transaction rows into a binary basket matrix.

        Expects columns: session_id, item_id.
        Returns a DataFrame with sessions as rows and items as columns (0/1).
        """
        basket = (
            transactions_df.groupby(["session_id", "item_id"])
            .size()
            .unstack(fill_value=0)
        )
        basket = (basket > 0).astype(int)
        return basket

    # ------------------------------------------------------------------
    # Mining
    # ------------------------------------------------------------------

    def mine_rules(
        self,
        basket_df: pd.DataFrame,
        min_support: float = 0.01,
        min_confidence: float = 0.1,
    ) -> None:
        """Run Apriori + association-rule extraction on the basket matrix."""
        from mlxtend.frequent_patterns import apriori, association_rules

        frequent = apriori(basket_df, min_support=min_support, use_colnames=True)
        if frequent.empty:
            logger.warning("No frequent itemsets found — try lowering min_support.")
            self.rules = pd.DataFrame()
            return

        rules = association_rules(frequent, metric="lift", min_threshold=1.0)
        rules = rules[rules["confidence"] >= min_confidence]
        rules = rules.sort_values(["lift", "confidence"], ascending=False)
        self.rules = rules.reset_index(drop=True)

        # Build lookup: item → [{consequent, confidence, lift}, …]
        self.antecedent_map = defaultdict(list)
        for _, row in self.rules.iterrows():
            for ant_item in row["antecedents"]:
                for con_item in row["consequents"]:
                    self.antecedent_map[ant_item].append(
                        {
                            "consequent": con_item,
                            "confidence": float(row["confidence"]),
                            "lift": float(row["lift"]),
                        }
                    )

        logger.info("Mined %d association rules.", len(self.rules))

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_frequently_bought_together(
        self, item_ids: List[Any], n_items: int = 5
    ) -> List[Dict[str, Any]]:
        """Return top complementary items for a set of input items."""
        score_map: Dict[Any, float] = defaultdict(float)

        for item_id in item_ids:
            for entry in self.antecedent_map.get(item_id, []):
                consequent = entry["consequent"]
                if consequent not in item_ids:
                    score_map[consequent] += entry["confidence"] * entry["lift"]

        ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        return [
            {"item_id": item, "score": round(score, 4)}
            for item, score in ranked[:n_items]
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path: Optional[str] = None) -> None:
        path = path or os.path.join(
            self.settings.MODEL_SAVE_PATH, "basket_rules.pkl"
        )
        payload = {
            "rules": self.rules,
            "antecedent_map": dict(self.antecedent_map),
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        logger.info("Basket rules saved to %s", path)

    def load_model(self, path: Optional[str] = None) -> None:
        path = path or os.path.join(
            self.settings.MODEL_SAVE_PATH, "basket_rules.pkl"
        )
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        self.rules = payload["rules"]
        self.antecedent_map = defaultdict(list, payload["antecedent_map"])
        logger.info("Basket rules loaded from %s", path)
