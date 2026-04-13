"""FastAPI REST API for the hybrid recommender engine."""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from ..data_processor import DataProcessor
from ..models.als_engine import ALSEngine
from ..models.basket_rules import BasketRules
from ..models.blend_recommender import BlendRecommender
from ..settings import config

logger = logging.getLogger(__name__)

# ── Global state ──────────────────────────────────────────────────────
_state: Dict[str, Any] = {
    "recommender": None,
    "als": None,
    "association": None,
    "processor": None,
    "start_time": None,
    "models_loaded": False,
}


# ── Lifespan ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: attempt to load persisted models; Shutdown: log."""
    _state["start_time"] = time.time()
    _state["processor"] = DataProcessor(config)

    als = ALSEngine(config)
    assoc = BasketRules(config)

    als_path = os.path.join(config.MODEL_SAVE_PATH, "als_model.pkl")
    assoc_path = os.path.join(config.MODEL_SAVE_PATH, "basket_rules.pkl")

    if os.path.exists(als_path) and os.path.exists(assoc_path):
        try:
            als.load_model(als_path)
            assoc.load_model(assoc_path)
            _state["models_loaded"] = True
            logger.info("Models loaded from disk.")
        except Exception as exc:
            logger.warning("Failed to load saved models: %s", exc)

    _state["als"] = als
    _state["association"] = assoc
    _state["recommender"] = BlendRecommender(config, als, assoc)

    yield

    logger.info("Shutting down recommender API.")


app = FastAPI(
    title="Hybrid Recommender Engine",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────

class RecommendRequest(BaseModel):
    user_id: str
    user_history: List[str] = []
    context: Optional[Dict[str, Any]] = None
    n_items: int = 10
    include_explanation: bool = False


class SimilarItemsRequest(BaseModel):
    item_id: str
    n_items: int = 10


class FrequentlyBoughtRequest(BaseModel):
    items: List[str]
    n_items: int = 5


class TrainRequest(BaseModel):
    data_path: str
    force_retrain: bool = False


# ── Endpoints ────────────────────────────────────────────────────────────

@app.get("/")
async def health():
    uptime = round(time.time() - _state["start_time"], 1) if _state["start_time"] else 0
    return {
        "status": "healthy",
        "models_loaded": {
            "als": _state["als"].model is not None if _state["als"] else False,
            "association": _state["association"].rules is not None if _state["association"] else False,
        },
        "uptime_seconds": uptime,
        "version": "1.0.0",
    }


@app.post("/recommend")
async def recommend(req: RecommendRequest):
    rec: BlendRecommender = _state["recommender"]
    if rec is None:
        raise HTTPException(503, "Recommender not initialised.")

    results = rec.get_recommendations(
        user_id=req.user_id,
        user_history=req.user_history,
        context=req.context,
        n_items=req.n_items,
    )

    response: Dict[str, Any] = {
        "user_id": req.user_id,
        "recommendations": results,
        "timestamp": datetime.utcnow().isoformat(),
        "request_id": str(uuid.uuid4()),
    }

    if req.include_explanation and results:
        explanations = {}
        for item in results[:5]:  # Limit explanation cost
            explanations[item["item_id"]] = rec.get_explanation(
                req.user_id, item["item_id"], req.user_history
            )
        response["explanations"] = explanations

    return response


@app.post("/similar-items")
async def similar_items(req: SimilarItemsRequest):
    als: ALSEngine = _state["als"]
    if als is None or als.model is None:
        raise HTTPException(503, "ALS model not trained yet.")
    results = als.get_similar_items(req.item_id, n_items=req.n_items)
    return {
        "item_id": req.item_id,
        "similar_items": [
            {"item_id": iid, "score": round(s, 4)} for iid, s in results
        ],
    }


@app.post("/frequently-bought-together")
async def frequently_bought_together(req: FrequentlyBoughtRequest):
    assoc: BasketRules = _state["association"]
    if assoc is None or assoc.rules is None:
        raise HTTPException(503, "Association model not trained yet.")
    results = assoc.get_frequently_bought_together(req.items, n_items=req.n_items)
    return {
        "input_items": req.items,
        "recommendations": results,
    }


@app.post("/train")
async def train(req: TrainRequest, background_tasks: BackgroundTasks):
    if not os.path.exists(req.data_path):
        raise HTTPException(404, f"Data file not found: {req.data_path}")
    background_tasks.add_task(_run_training, req.data_path)
    return {"status": "Training started in background.", "data_path": req.data_path}


@app.get("/metrics")
async def metrics():
    rec: BlendRecommender = _state["recommender"]
    als: ALSEngine = _state["als"]
    uptime = round(time.time() - _state["start_time"], 1) if _state["start_time"] else 0

    return {
        "system": {
            "uptime_seconds": uptime,
            "models_loaded": _state["models_loaded"],
        },
        "recommendations": {
            "total_served": rec.total_served if rec else 0,
            "cache_hits": rec.cache_hits if rec else 0,
            "cache_hit_rate": rec.cache_hit_rate if rec else 0.0,
        },
        "models": {
            "als_factors": config.ALS_FACTORS,
            "als_iterations": config.ALS_ITERATIONS,
            "num_users": len(als.user_mapping) if als else 0,
            "num_items": len(als.item_mapping) if als else 0,
        },
    }


@app.post("/evaluate")
async def evaluate(test_data_path: str, k: int = 10):
    if not os.path.exists(test_data_path):
        raise HTTPException(404, f"Test data not found: {test_data_path}")

    processor: DataProcessor = _state["processor"]
    als: ALSEngine = _state["als"]
    if als is None or als.model is None:
        raise HTTPException(503, "ALS model not trained yet.")

    df = processor.load_interaction_data(test_data_path)
    users = df["user_id"].unique()[:100]

    precisions, recalls = [], []
    for uid in users:
        actual = set(df[df["user_id"] == uid]["item_id"].unique())
        history = list(actual)[:max(1, len(actual) // 2)]
        recs = als.recommend_for_user(uid, history, n_items=k)
        predicted = {r[0] for r in recs}

        if not actual:
            continue
        hits = predicted & actual
        precisions.append(len(hits) / k if k else 0)
        recalls.append(len(hits) / len(actual) if actual else 0)

    avg_p = float(np.mean(precisions)) if precisions else 0.0
    avg_r = float(np.mean(recalls)) if recalls else 0.0
    f1 = (2 * avg_p * avg_r / (avg_p + avg_r)) if (avg_p + avg_r) else 0.0

    return {
        "precision_at_k": round(avg_p, 4),
        "recall_at_k": round(avg_r, 4),
        "f1_score": round(f1, 4),
        "k": k,
        "users_evaluated": len(precisions),
    }


@app.get("/dashboard")
async def dashboard():
    html_path = os.path.join(os.path.dirname(__file__), "..", "..", "dashboard.html")
    abs_path = os.path.abspath(html_path)
    if os.path.exists(abs_path):
        return FileResponse(abs_path, media_type="text/html")
    return HTMLResponse("<h1>Dashboard not found</h1>", status_code=404)


# ── Background training ────────────────────────────────────────────────────────

def _run_training(data_path: str) -> None:
    """Execute the full training pipeline in a background thread."""
    try:
        processor: DataProcessor = _state["processor"]
        als: ALSEngine = _state["als"]
        assoc: BasketRules = _state["association"]

        # Step 1 — interaction matrix
        matrix, meta = processor.prepare_training_data(data_path)

        # Step 2 — ALS
        als.train(matrix, meta["user_mapping"], meta["item_mapping"])
        als.save_model()

        # Step 3 — Association rules (purchases only, session inference)
        raw_df = processor.load_interaction_data(data_path)
        purchases = raw_df[raw_df["interaction_type"].str.lower() == "purchase"].copy()

        if not purchases.empty:
            purchases = purchases.sort_values(["user_id", "timestamp"])
            purchases["time_diff"] = purchases.groupby("user_id")["timestamp"].diff()
            purchases["new_session"] = (
                purchases["time_diff"].isna()
                | (purchases["time_diff"].dt.total_seconds() > 1800)
            )
            purchases["session_id"] = purchases["new_session"].cumsum()

            basket = assoc.prepare_basket_data(purchases[["session_id", "item_id"]])
            if len(basket) > 0:
                assoc.mine_rules(basket)
                assoc.save_model()

        # Step 4 — Rebuild hybrid recommender
        _state["recommender"] = BlendRecommender(config, als, assoc)
        _state["models_loaded"] = True
        logger.info("Training complete.")

    except Exception as exc:
        logger.exception("Training failed: %s", exc)
