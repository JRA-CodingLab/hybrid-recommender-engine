"""Load, clean, map, and vectorise user–item interaction data."""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .settings import EngineSettings

logger = logging.getLogger(__name__)

# Weights assigned to each interaction type
INTERACTION_WEIGHTS: Dict[str, float] = {
    "purchase": 5.0,
    "add_to_cart": 3.0,
    "wishlist": 2.0,
    "view": 1.0,
    "click": 0.5,
}


class DataProcessor:
    """Transforms raw interaction CSVs into sparse matrices ready for training."""

    def __init__(self, settings: EngineSettings) -> None:
        self.settings = settings
        self.user_mapping: Dict[Any, int] = {}
        self.item_mapping: Dict[Any, int] = {}
        self.reverse_user_mapping: Dict[int, Any] = {}
        self.reverse_item_mapping: Dict[int, Any] = {}

    # ------------------------------------------------------------------
    # Public pipeline
    # ------------------------------------------------------------------

    def load_interaction_data(self, file_path: str) -> pd.DataFrame:
        """Read CSV and apply interaction weights + timestamp parsing."""
        required_cols = {"user_id", "item_id", "interaction_type", "timestamp"}

        df = pd.read_csv(file_path)
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        df["interaction_weight"] = (
            df["interaction_type"]
            .str.lower()
            .map(INTERACTION_WEIGHTS)
            .fillna(1.0)
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        logger.info("Loaded %d interactions from %s", len(df), file_path)
        return df

    def filter_sparse_users_items(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove users/items with fewer interactions than the threshold."""
        threshold = self.settings.MIN_INTERACTIONS_THRESHOLD

        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= threshold].index
        df = df[df["user_id"].isin(valid_users)]

        item_counts = df["item_id"].value_counts()
        valid_items = item_counts[item_counts >= threshold].index
        df = df[df["item_id"].isin(valid_items)]

        logger.info("After filtering: %d interactions", len(df))
        return df

    def create_mappings(self, df: pd.DataFrame) -> None:
        """Build bidirectional user/item → sequential-index dictionaries."""
        unique_users = sorted(df["user_id"].unique())
        unique_items = sorted(df["item_id"].unique())

        self.user_mapping = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_mapping = {iid: idx for idx, iid in enumerate(unique_items)}
        self.reverse_user_mapping = {idx: uid for uid, idx in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: iid for iid, idx in self.item_mapping.items()}

        logger.info(
            "Mappings created — %d users, %d items",
            len(self.user_mapping),
            len(self.item_mapping),
        )

    def create_interaction_matrix(self, df: pd.DataFrame) -> csr_matrix:
        """Build a weighted user×item sparse matrix with recency decay."""
        latest_ts = df["timestamp"].max()
        days_ago = (latest_ts - df["timestamp"]).dt.total_seconds() / 86400.0
        recency_weight = np.exp(-0.05 * days_ago)
        final_weight = df["interaction_weight"].values * recency_weight.values

        row_indices = df["user_id"].map(self.user_mapping).values.astype(np.int32)
        col_indices = df["item_id"].map(self.item_mapping).values.astype(np.int32)

        n_users = len(self.user_mapping)
        n_items = len(self.item_mapping)
        matrix = csr_matrix(
            (final_weight, (row_indices, col_indices)),
            shape=(n_users, n_items),
        )
        logger.info("Interaction matrix: %s", matrix.shape)
        return matrix

    def prepare_training_data(
        self, file_path: str
    ) -> Tuple[csr_matrix, Dict[str, Any]]:
        """Full pipeline: load → filter → map → matrix → persist mappings."""
        df = self.load_interaction_data(file_path)
        df = self.filter_sparse_users_items(df)
        self.create_mappings(df)
        matrix = self.create_interaction_matrix(df)
        self.save_mappings()

        metadata = {
            "num_users": len(self.user_mapping),
            "num_items": len(self.item_mapping),
            "user_mapping": self.user_mapping,
            "item_mapping": self.item_mapping,
            "reverse_user_mapping": self.reverse_user_mapping,
            "reverse_item_mapping": self.reverse_item_mapping,
            "latest_timestamp": str(df["timestamp"].max()),
        }
        return matrix, metadata

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_mappings(self) -> None:
        path = os.path.join(self.settings.PROCESSED_DATA_PATH, "mappings.pkl")
        payload = {
            "user_mapping": self.user_mapping,
            "item_mapping": self.item_mapping,
            "reverse_user_mapping": self.reverse_user_mapping,
            "reverse_item_mapping": self.reverse_item_mapping,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        logger.info("Mappings saved to %s", path)

    def load_mappings(self) -> None:
        path = os.path.join(self.settings.PROCESSED_DATA_PATH, "mappings.pkl")
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        self.user_mapping = payload["user_mapping"]
        self.item_mapping = payload["item_mapping"]
        self.reverse_user_mapping = payload["reverse_user_mapping"]
        self.reverse_item_mapping = payload["reverse_item_mapping"]
        logger.info("Mappings loaded from %s", path)
