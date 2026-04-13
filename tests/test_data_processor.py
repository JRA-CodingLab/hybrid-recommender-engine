"""Tests for the DataProcessor module."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse

from src.data_processor import DataProcessor, INTERACTION_WEIGHTS
from src.settings import EngineSettings


@pytest.fixture
def tmp_settings():
    base = tempfile.mkdtemp()
    return EngineSettings(
        RAW_DATA_PATH=os.path.join(base, "raw"),
        PROCESSED_DATA_PATH=os.path.join(base, "processed"),
        MODEL_SAVE_PATH=os.path.join(base, "models"),
        MIN_INTERACTIONS_THRESHOLD=1,
    )


@pytest.fixture
def sample_csv(tmp_path):
    """Create a minimal interaction CSV."""
    rows = [
        ("u1", "i1", "purchase", "2025-01-10 10:00:00"),
        ("u1", "i2", "view", "2025-01-11 12:00:00"),
        ("u1", "i3", "click", "2025-01-12 08:00:00"),
        ("u2", "i1", "add_to_cart", "2025-01-10 14:00:00"),
        ("u2", "i2", "purchase", "2025-01-11 16:00:00"),
        ("u3", "i3", "wishlist", "2025-01-12 09:00:00"),
    ]
    df = pd.DataFrame(rows, columns=["user_id", "item_id", "interaction_type", "timestamp"])
    path = str(tmp_path / "interactions.csv")
    df.to_csv(path, index=False)
    return path


class TestDataProcessor:
    def test_load_interaction_data(self, tmp_settings, sample_csv):
        dp = DataProcessor(tmp_settings)
        df = dp.load_interaction_data(sample_csv)

        assert "interaction_weight" in df.columns
        assert "timestamp" in df.columns
        assert len(df) == 6
        # Check weight mapping
        purchases = df[df["interaction_type"] == "purchase"]
        assert (purchases["interaction_weight"] == 5.0).all()

    def test_load_missing_columns(self, tmp_settings, tmp_path):
        bad_csv = str(tmp_path / "bad.csv")
        pd.DataFrame({"a": [1]}).to_csv(bad_csv, index=False)

        dp = DataProcessor(tmp_settings)
        with pytest.raises(ValueError, match="missing required columns"):
            dp.load_interaction_data(bad_csv)

    def test_filter_sparse(self, tmp_settings, sample_csv):
        dp = DataProcessor(tmp_settings)
        df = dp.load_interaction_data(sample_csv)
        filtered = dp.filter_sparse_users_items(df)
        # With threshold=1, nothing should be removed
        assert len(filtered) == len(df)

    def test_filter_sparse_removes(self, tmp_settings, sample_csv):
        tmp_settings.MIN_INTERACTIONS_THRESHOLD = 2
        dp = DataProcessor(tmp_settings)
        df = dp.load_interaction_data(sample_csv)
        filtered = dp.filter_sparse_users_items(df)
        # u3 has only 1 interaction → removed; u1 (3) and u2 (2) survive
        assert "u3" not in filtered["user_id"].values
        assert "u1" in filtered["user_id"].values
        assert "u2" in filtered["user_id"].values

    def test_create_mappings(self, tmp_settings, sample_csv):
        dp = DataProcessor(tmp_settings)
        df = dp.load_interaction_data(sample_csv)
        dp.create_mappings(df)

        assert len(dp.user_mapping) == 3
        assert len(dp.item_mapping) == 3
        assert dp.reverse_user_mapping[dp.user_mapping["u1"]] == "u1"
        assert dp.reverse_item_mapping[dp.item_mapping["i2"]] == "i2"

    def test_create_interaction_matrix(self, tmp_settings, sample_csv):
        dp = DataProcessor(tmp_settings)
        df = dp.load_interaction_data(sample_csv)
        dp.create_mappings(df)
        matrix = dp.create_interaction_matrix(df)

        assert issparse(matrix)
        assert matrix.shape == (3, 3)  # 3 users × 3 items
        assert matrix.nnz > 0

    def test_prepare_training_data(self, tmp_settings, sample_csv):
        dp = DataProcessor(tmp_settings)
        matrix, meta = dp.prepare_training_data(sample_csv)

        assert issparse(matrix)
        assert meta["num_users"] == 3
        assert meta["num_items"] == 3
        assert "user_mapping" in meta
        assert "item_mapping" in meta

    def test_save_load_mappings(self, tmp_settings, sample_csv):
        dp = DataProcessor(tmp_settings)
        dp.prepare_training_data(sample_csv)
        original_user = dict(dp.user_mapping)

        dp2 = DataProcessor(tmp_settings)
        dp2.load_mappings()
        assert dp2.user_mapping == original_user


class TestInteractionWeights:
    def test_all_types_present(self):
        expected = {"purchase", "add_to_cart", "wishlist", "view", "click"}
        assert set(INTERACTION_WEIGHTS.keys()) == expected

    def test_weight_ordering(self):
        assert INTERACTION_WEIGHTS["purchase"] > INTERACTION_WEIGHTS["add_to_cart"]
        assert INTERACTION_WEIGHTS["add_to_cart"] > INTERACTION_WEIGHTS["wishlist"]
        assert INTERACTION_WEIGHTS["wishlist"] > INTERACTION_WEIGHTS["view"]
        assert INTERACTION_WEIGHTS["view"] > INTERACTION_WEIGHTS["click"]
