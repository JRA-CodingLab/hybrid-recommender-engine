"""Tests for the BlendRecommender hybrid combiner."""

import tempfile
from unittest.mock import MagicMock, patch

import pytest

from src.models.als_engine import ALSEngine
from src.models.basket_rules import BasketRules
from src.models.blend_recommender import BlendRecommender
from src.settings import EngineSettings


@pytest.fixture
def settings():
    base = tempfile.mkdtemp()
    return EngineSettings(
        RAW_DATA_PATH=f"{base}/raw",
        PROCESSED_DATA_PATH=f"{base}/proc",
        MODEL_SAVE_PATH=f"{base}/model",
        COLLABORATIVE_WEIGHT=0.5,
        POPULARITY_WEIGHT=0.2,
        ASSOCIATION_WEIGHT=0.2,
        RECENCY_WEIGHT=0.1,
    )


@pytest.fixture
def mock_als(settings):
    als = ALSEngine(settings)
    als.model = MagicMock()
    als.user_mapping = {"u1": 0, "u2": 1}
    als.item_mapping = {"i1": 0, "i2": 1, "i3": 2, "i4": 3}
    als.reverse_user_mapping = {0: "u1", 1: "u2"}
    als.reverse_item_mapping = {0: "i1", 1: "i2", 2: "i3", 3: "i4"}
    als.recommend_for_user = MagicMock(
        return_value=[("i3", 0.9), ("i4", 0.7)]
    )
    als.get_similar_items = MagicMock(
        return_value=[("i4", 0.8), ("i3", 0.6)]
    )
    als.get_popular_items = MagicMock(
        return_value=[("i1", 1.0), ("i2", 0.9), ("i3", 0.85), ("i4", 0.7)]
    )
    return als


@pytest.fixture
def mock_assoc(settings):
    assoc = BasketRules(settings)
    assoc.rules = MagicMock()
    assoc.get_frequently_bought_together = MagicMock(
        return_value=[
            {"item_id": "i4", "score": 2.5},
            {"item_id": "i3", "score": 1.8},
        ]
    )
    return assoc


@pytest.fixture
def recommender(settings, mock_als, mock_assoc):
    with patch("src.models.blend_recommender.BlendRecommender._init_cache"):
        rec = BlendRecommender(settings, mock_als, mock_assoc)
        rec._cache = None
        rec._dict_cache = {}
    return rec


class TestBlendRecommender:
    def test_get_recommendations_returns_list(self, recommender):
        results = recommender.get_recommendations(
            user_id="u1",
            user_history=["i1", "i2"],
            n_items=5,
        )
        assert isinstance(results, list)
        assert len(results) > 0

    def test_recommendations_exclude_history(self, recommender):
        results = recommender.get_recommendations(
            user_id="u1",
            user_history=["i1", "i2"],
            n_items=10,
        )
        rec_ids = {r["item_id"] for r in results}
        assert "i1" not in rec_ids
        assert "i2" not in rec_ids

    def test_result_structure(self, recommender):
        results = recommender.get_recommendations(
            user_id="u1",
            user_history=["i1"],
            n_items=5,
        )
        for item in results:
            assert "item_id" in item
            assert "score" in item
            assert "reason" in item
            assert isinstance(item["reason"], list)

    def test_caching(self, recommender):
        results_a = recommender.get_recommendations("u1", ["i1"], n_items=3)
        assert recommender.total_served == 1
        assert recommender.cache_hits == 0

        results_b = recommender.get_recommendations("u1", ["i1"], n_items=3)
        assert recommender.total_served == 2
        assert recommender.cache_hits == 1
        assert results_a == results_b

    def test_cache_hit_rate(self, recommender):
        assert recommender.cache_hit_rate == 0.0
        recommender.get_recommendations("u1", ["i1"])
        recommender.get_recommendations("u1", ["i1"])
        assert recommender.cache_hit_rate == 0.5

    def test_business_rules_filter_oos(self, recommender):
        context = {"out_of_stock": ["i3"]}
        results = recommender.get_recommendations(
            user_id="u1",
            user_history=["i1"],
            context=context,
            n_items=10,
        )
        rec_ids = {r["item_id"] for r in results}
        assert "i3" not in rec_ids

    def test_business_rules_boost_margin(self):
        scores = {"i1": 1.0, "i2": 1.0}
        context = {"high_margin_items": ["i2"]}
        result = BlendRecommender._apply_business_rules(scores, context)
        assert result["i2"] > result["i1"]

    def test_explanation(self, recommender):
        explanations = recommender.get_explanation("u1", "i3", ["i1"])
        assert isinstance(explanations, list)
        assert len(explanations) > 0
        for e in explanations:
            assert "type" in e
            assert "description" in e
            assert "confidence" in e

    def test_no_context_passthrough(self):
        scores = {"i1": 1.0, "i2": 2.0}
        result = BlendRecommender._apply_business_rules(scores, None)
        assert result == scores

    def test_calls_all_signal_sources(self, recommender, mock_als, mock_assoc):
        recommender.get_recommendations("u1", ["i1"], n_items=5)
        mock_als.recommend_for_user.assert_called_once()
        mock_als.get_similar_items.assert_called()
        mock_als.get_popular_items.assert_called_once()
        mock_assoc.get_frequently_bought_together.assert_called_once()
