"""
Hybrid Recommender Engine - Mock API
Author: Juan Ruiz Alonso
Standalone FastAPI service with 50 items, 20 users, cosine similarity + co-occurrence + popularity blend.
No external dependencies beyond numpy.
"""

from __future__ import annotations
import math, uuid, hashlib
from datetime import datetime
from typing import Any
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Hybrid Recommender Engine", description="Mock recommendation API for CV showcase.", version="1.0.0", docs_url="/docs", redoc_url="/redoc")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Mock Catalog (50 items) ---
ITEMS = {}
_BOOKS = [("B001","Designing Data-Intensive Applications"),("B002","Clean Code"),("B003","Design Patterns"),("B004","The Pragmatic Programmer"),("B005","Introduction to Algorithms"),("B006","Python Crash Course"),("B007","Deep Learning with Python"),("B008","Hands-On Machine Learning"),("B009","Atomic Habits"),("B010","Rich Dad Poor Dad"),("B011","The Lean Startup"),("B012","Zero to One"),("B013","Sapiens"),("B014","Thinking, Fast and Slow"),("B015","The Art of War"),("B016","Dune"),("B017","1984")]
_MOVIES = [("M001","Inception"),("M002","The Dark Knight"),("M003","Interstellar"),("M004","The Matrix"),("M005","Fight Club"),("M006","Pulp Fiction"),("M007","The Godfather"),("M008","Schindler's List"),("M009","The Shawshank Redemption"),("M010","Goodfellas"),("M011","The Lord of the Rings"),("M012","Forrest Gump"),("M013","The Silence of the Lambs"),("M014","Gladiator"),("M015","Parasite"),("M016","Whiplash")]
_PRODUCTS = [("P001","Mechanical Keyboard MK-750"),("P002","Ergonomic Mouse EM-Pro"),("P003",'27" 4K Monitor UW-4K'),("P004","USB-C Hub 10-in-1"),("P005","Noise-Cancelling Headphones NC-X"),("P006","Standing Desk SD-Elite"),("P007","Webcam 4K WC-Ultra"),("P008","LED Desk Lamp DL-Smart"),("P009","External SSD 2TB"),("P010","Laptop Stand LS-Aero"),("P011","Blue Light Glasses BL-Pro"),("P012","Cable Management Kit"),("P013","Portable Charger 20000mAh"),("P014","Wireless Earbuds WE-Fit"),("P015","Smart Water Bottle"),("P016","Desk Organizer DO-Zen"),("P017","Fitness Tracker FT-Band")]
for iid, name in _BOOKS: ITEMS[iid] = {"name": name, "category": "books"}
for iid, name in _MOVIES: ITEMS[iid] = {"name": name, "category": "movies"}
for iid, name in _PRODUCTS: ITEMS[iid] = {"name": name, "category": "products"}

ALL_IDS = list(ITEMS.keys())
rng = np.random.default_rng(42)

# Item feature vectors for cosine similarity
CAT_MAP = {"books": 0, "movies": 1, "products": 2}
ITEM_VECS = {}
for i, iid in enumerate(ALL_IDS):
    cat = ITEMS[iid]["category"]
    v = np.zeros(6)
    v[CAT_MAP[cat]] = 1.0
    v[3] = rng.uniform(0.1, 1.0)  # popularity
    v[4] = rng.uniform(0.0, 1.0)  # price tier
    v[5] = rng.uniform(0.0, 1.0)  # rating
    ITEM_VECS[iid] = v

# 20 Users with purchase histories
USERS = {}
for u in range(1, 21):
    uid = f"U{u:02d}"
    n_items = rng.integers(3, 12)
    history = list(rng.choice(ALL_IDS, size=n_items, replace=False))
    USERS[uid] = history

# Co-occurrence matrix
COOCCUR = {}
for uid, hist in USERS.items():
    for i, a in enumerate(hist):
        for b in hist[i+1:]:
            key = tuple(sorted([a, b]))
            COOCCUR[key] = COOCCUR.get(key, 0) + 1

# Popularity scores
POP = {iid: 0 for iid in ALL_IDS}
for uid, hist in USERS.items():
    for iid in hist:
        POP[iid] += 1

_metrics = {"recommend_calls": 0, "similar_calls": 0, "fbt_calls": 0, "total_recs": 0}

def _cosine(a, b):
    d = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(d / (na * nb)) if na > 0 and nb > 0 else 0.0

# --- Pydantic Models ---
class ServiceInfo(BaseModel):
    service: str = "Hybrid Recommender Engine"
    author: str = "Juan Ruiz Alonso"
    version: str = "1.0.0"
    status: str = "healthy"
    description: str = "Mock recommender combining collaborative filtering, association rules, and popularity."
    endpoints: list[str] = ["/recommend", "/similar-items", "/frequently-bought-together", "/metrics", "/dashboard", "/examples"]
    docs: str = "/docs"
    timestamp: str = ""

class RecommendRequest(BaseModel):
    user_id: str
    n_items: int = Field(default=10, ge=1, le=50)

class SimilarItemsRequest(BaseModel):
    item_id: str
    n_items: int = Field(default=10, ge=1, le=50)

class FBTRequest(BaseModel):
    items: list[str]
    n_items: int = Field(default=5, ge=1, le=20)

class RecItem(BaseModel):
    item_id: str
    name: str
    category: str
    score: float
    strategy: str

# --- Endpoints ---
@app.get("/", tags=["Health"])
async def root():
    return ServiceInfo(timestamp=datetime.utcnow().isoformat() + "Z")

@app.post("/recommend", tags=["Recommendations"])
async def recommend(req: RecommendRequest):
    if req.user_id not in USERS:
        raise HTTPException(404, f"User '{req.user_id}' not found. Valid IDs: U01-U20.")
    hist = set(USERS[req.user_id])
    candidates = [iid for iid in ALL_IDS if iid not in hist]
    scored = []
    for iid in candidates:
        # Association score
        assoc = sum(COOCCUR.get(tuple(sorted([iid, h])), 0) for h in hist) / max(len(hist), 1)
        # Popularity score
        pop = POP[iid] / max(max(POP.values()), 1)
        # Collaborative (user-item affinity via vector similarity)
        user_vec = np.mean([ITEM_VECS[h] for h in hist], axis=0)
        collab = _cosine(user_vec, ITEM_VECS[iid])
        # Blend
        score = 0.5 * collab + 0.3 * assoc + 0.2 * pop
        strategy = "collaborative" if collab > assoc and collab > pop else ("association" if assoc > pop else "popularity")
        scored.append((iid, round(score, 4), strategy))
    scored.sort(key=lambda x: x[1], reverse=True)
    results = [RecItem(item_id=iid, name=ITEMS[iid]["name"], category=ITEMS[iid]["category"], score=s, strategy=st) for iid, s, st in scored[:req.n_items]]
    _metrics["recommend_calls"] += 1
    _metrics["total_recs"] += len(results)
    return {"user_id": req.user_id, "recommendations": [r.dict() for r in results], "generated_at": datetime.utcnow().isoformat() + "Z"}

@app.post("/similar-items", tags=["Recommendations"])
async def similar_items(req: SimilarItemsRequest):
    if req.item_id not in ITEMS:
        raise HTTPException(404, f"Item '{req.item_id}' not found.")
    ref = ITEM_VECS[req.item_id]
    scored = [(iid, round(_cosine(ref, ITEM_VECS[iid]), 4)) for iid in ALL_IDS if iid != req.item_id]
    scored.sort(key=lambda x: x[1], reverse=True)
    results = [RecItem(item_id=iid, name=ITEMS[iid]["name"], category=ITEMS[iid]["category"], score=s, strategy="cosine_similarity") for iid, s in scored[:req.n_items]]
    _metrics["similar_calls"] += 1
    _metrics["total_recs"] += len(results)
    return {"item_id": req.item_id, "similar_items": [r.dict() for r in results], "generated_at": datetime.utcnow().isoformat() + "Z"}

@app.post("/frequently-bought-together", tags=["Recommendations"])
async def fbt(req: FBTRequest):
    unknown = [i for i in req.items if i not in ITEMS]
    if unknown:
        raise HTTPException(404, f"Unknown item IDs: {unknown}")
    scored = {}
    for iid in ALL_IDS:
        if iid in req.items:
            continue
        total = sum(COOCCUR.get(tuple(sorted([iid, inp])), 0) for inp in req.items)
        if total > 0:
            scored[iid] = total / len(req.items)
    top = sorted(scored.items(), key=lambda x: x[1], reverse=True)[:req.n_items]
    results = [RecItem(item_id=iid, name=ITEMS[iid]["name"], category=ITEMS[iid]["category"], score=round(s, 4), strategy="association_rules") for iid, s in top]
    _metrics["fbt_calls"] += 1
    _metrics["total_recs"] += len(results)
    return {"input_items": req.items, "associated_items": [r.dict() for r in results], "generated_at": datetime.utcnow().isoformat() + "Z"}

@app.get("/metrics", tags=["System"])
async def metrics():
    return {"total_users": len(USERS), "total_items": len(ITEMS), **_metrics, "items_by_category": {"books": len(_BOOKS), "movies": len(_MOVIES), "products": len(_PRODUCTS)}}

@app.get("/examples", tags=["System"])
async def examples():
    return {"recommend": {"endpoint": "POST /recommend", "payload": {"user_id": "U03", "n_items": 5}, "valid_user_ids": list(USERS.keys())}, "similar_items": {"endpoint": "POST /similar-items", "payload": {"item_id": "M001", "n_items": 6}}, "frequently_bought_together": {"endpoint": "POST /frequently-bought-together", "payload": {"items": ["B001", "P001"], "n_items": 5}}}

@app.get("/dashboard", response_class=HTMLResponse, tags=["System"])
async def dashboard():
    return "<html><head><title>Recommender Dashboard</title><style>body{font-family:sans-serif;background:#1a1a2e;color:#eee;padding:40px;max-width:900px;margin:0 auto}h1{color:#e94560}h2{color:#0f3460;background:#16213e;padding:12px;border-radius:8px;color:#eee}.card{background:#16213e;padding:20px;border-radius:12px;margin:16px 0}.stat{display:inline-block;margin:12px 24px;text-align:center}.stat .num{font-size:2em;color:#e94560;display:block}.stat .label{color:#aaa;font-size:0.9em}code{background:#0f3460;padding:2px 8px;border-radius:4px}</style></head><body><h1>Hybrid Recommender Engine</h1><p>Author: Juan Ruiz Alonso</p><div class='card'><div class='stat'><span class='num'>50</span><span class='label'>Items</span></div><div class='stat'><span class='num'>20</span><span class='label'>Users</span></div><div class='stat'><span class='num'>3</span><span class='label'>Strategies</span></div></div><h2>API Endpoints</h2><div class='card'><p><code>POST /recommend</code> - Personalised recommendations (collaborative + association + popularity)</p><p><code>POST /similar-items</code> - Cosine similarity on item feature vectors</p><p><code>POST /frequently-bought-together</code> - Co-occurrence association rules</p><p><code>GET /metrics</code> - System metrics</p><p><code>GET /docs</code> - Interactive Swagger UI</p></div><h2>Try It</h2><div class='card'><p>Open <a href='/docs' style='color:#e94560'>/docs</a> for interactive API testing.</p></div></body></html>"

if __name__ == "__main__":
    import os, uvicorn
    uvicorn.run("deploy.main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
