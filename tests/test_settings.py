"""Tests for EngineSettings configuration."""

import os
import tempfile

from src.settings import EngineSettings


class TestEngineSettings:
    def test_default_values(self):
        s = EngineSettings(
            RAW_DATA_PATH=tempfile.mkdtemp(),
            PROCESSED_DATA_PATH=tempfile.mkdtemp(),
            MODEL_SAVE_PATH=tempfile.mkdtemp(),
        )
        assert s.ALS_FACTORS == 50
        assert s.ALS_REGULARIZATION == 0.1
        assert s.ALS_ITERATIONS == 20
        assert s.ALS_ALPHA == 40.0
        assert s.TOP_K_RECOMMENDATIONS == 20
        assert s.SIMILAR_ITEMS_COUNT == 10
        assert s.COLLABORATIVE_WEIGHT == 0.5
        assert s.POPULARITY_WEIGHT == 0.2
        assert s.ASSOCIATION_WEIGHT == 0.2
        assert s.RECENCY_WEIGHT == 0.1
        assert s.API_PORT == 8000
        assert s.DEBUG is False

    def test_directory_creation(self):
        with tempfile.TemporaryDirectory() as base:
            raw = os.path.join(base, "nested", "raw")
            proc = os.path.join(base, "nested", "processed")
            model = os.path.join(base, "nested", "models")

            s = EngineSettings(
                RAW_DATA_PATH=raw,
                PROCESSED_DATA_PATH=proc,
                MODEL_SAVE_PATH=model,
            )
            assert os.path.isdir(raw)
            assert os.path.isdir(proc)
            assert os.path.isdir(model)

    def test_weight_sum(self):
        s = EngineSettings(
            RAW_DATA_PATH=tempfile.mkdtemp(),
            PROCESSED_DATA_PATH=tempfile.mkdtemp(),
            MODEL_SAVE_PATH=tempfile.mkdtemp(),
        )
        total = (
            s.COLLABORATIVE_WEIGHT
            + s.POPULARITY_WEIGHT
            + s.ASSOCIATION_WEIGHT
            + s.RECENCY_WEIGHT
        )
        assert abs(total - 1.0) < 1e-9

    def test_custom_overrides(self):
        s = EngineSettings(
            RAW_DATA_PATH=tempfile.mkdtemp(),
            PROCESSED_DATA_PATH=tempfile.mkdtemp(),
            MODEL_SAVE_PATH=tempfile.mkdtemp(),
            ALS_FACTORS=100,
            API_PORT=9999,
            DEBUG=True,
        )
        assert s.ALS_FACTORS == 100
        assert s.API_PORT == 9999
        assert s.DEBUG is True
