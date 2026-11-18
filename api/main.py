"""
PathSnap API  –  Map-matching prediction service.

Endpoints
─────────
GET  /trips                → list the 10 student-test trips
POST /predict/{trip_id}    → run the ML + Viterbi pipeline and return predicted route

Design rules
────────────
• Ground-truth .route files are NEVER sent to the client or the model.
  They live in `ground_truth/` and are read only by `scoring.score_trip()`.
• The road graph and spatial index are loaded ONCE at startup (`@app.on_event`).
• Every request is logged with a structured JSON logger so evaluators can
  review predictions on Laptop 1.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .data_loader import load_trip_data, TripMeta
from .predictor import DummyPredictor, get_predictor
from .scoring import score_trip, load_ground_truth

# ── Structured logging ───────────────────────────────────────────────
logger = logging.getLogger("pathsnap")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(message)s", datefmt="%H:%M:%S"
    )
)
logger.addHandler(handler)


# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
def _find_test_data_dir() -> Path:
    candidates = [
        BASE_DIR / "src" / "Student folder" / "student_test",
        BASE_DIR / "Student folder" / "Student folder" / "student_test",
        BASE_DIR / "Student folder" / "student_test",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return candidates[0]

DATA_DIR = _find_test_data_dir()
GROUND_TRUTH_DIR = BASE_DIR / "ground_truth"
STATIC_DIR = Path(__file__).resolve().parent / "static"


# ── Lifespan (load heavy objects once) ───────────────────────────────
trip_registry: dict[str, TripMeta] = {}
ground_truths: dict[str, np.ndarray] = {}
predictor = None  # will be set at startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, trip_registry, ground_truths

    t0 = time.perf_counter()
    logger.info("Loading student-test trips …")

    # Discover the 10 student-test trip folders
    if not DATA_DIR.exists():
        logger.error("Data directory not found: %s", DATA_DIR)
        raise RuntimeError(f"Data directory not found: {DATA_DIR}")

    for trip_dir in sorted(DATA_DIR.iterdir()):
        if not trip_dir.is_dir():
            continue
        tid = trip_dir.name
        try:
            meta = load_trip_data(trip_dir)
            trip_registry[tid] = meta
            logger.info("  ✓ trip %s  (%d GPS points, %d nodes, %d arcs)",
                        tid, meta.n_points, meta.n_nodes, meta.n_arcs)
        except Exception as e:
            logger.warning("  ✗ trip %s  skipped: %s", tid, e)

    # Load ground-truth route files into server-only storage
    # Ground truth is kept in a SEPARATE folder (ground_truth/) that the API never exposes
    GROUND_TRUTH_DIR.mkdir(exist_ok=True)
    for tid, meta in trip_registry.items():
        try:
            gt = load_ground_truth(tid, trip_dir=meta.trip_dir, ground_truth_dir=GROUND_TRUTH_DIR)
            ground_truths[tid] = gt
            logger.info("  🔒 ground truth loaded for trip %s (%d arcs)", tid, len(gt))
        except FileNotFoundError:
            logger.info("  ⚠ no ground truth for trip %s (expected for evaluator trips)", tid)

    # Initialise predictor (dummy until B & C deliver)
    predictor = get_predictor()
    logger.info("Predictor ready: %s", type(predictor).__name__)

    elapsed = time.perf_counter() - t0
    logger.info("Startup complete in %.2fs  –  %d trips loaded", elapsed, len(trip_registry))

    yield  # ← app runs here

    logger.info("Shutting down …")


# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="PathSnap API",
    description="GPS map-matching prediction service",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for two-laptop setup; tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve UI static files
if STATIC_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui")


# ── GET /trips ───────────────────────────────────────────────────────
@app.get("/trips")
async def list_trips():
    """Return metadata for the 10 student-test trips (no route/ground-truth)."""
    request_id = uuid.uuid4().hex[:8]
    logger.info("[%s] GET /trips", request_id)

    trips = []
    for tid, meta in trip_registry.items():
        trips.append({
            "trip_id": tid,
            "n_points": meta.n_points,
            "n_nodes": meta.n_nodes,
            "n_arcs": meta.n_arcs,
            "bbox": meta.bbox,
            "track_preview": meta.track_lonlat[:5].tolist(),
        })

    logger.info("[%s] → returning %d trips", request_id, len(trips))
    return {"trips": trips, "count": len(trips)}


# ── POST /predict/{trip_id} ──────────────────────────────────────────
@app.post("/predict/{trip_id}")
async def predict_route(trip_id: str):
    """Run the ML + Viterbi pipeline on the given trip and return the predicted route."""
    request_id = uuid.uuid4().hex[:8]
    logger.info("[%s] POST /predict/%s", request_id, trip_id)

    if trip_id not in trip_registry:
        logger.warning("[%s] trip %s not found", request_id, trip_id)
        raise HTTPException(status_code=404, detail=f"Trip {trip_id} not found")

    meta = trip_registry[trip_id]

    t0 = time.perf_counter()
    try:
        result = predictor.predict(meta)
    except Exception as e:
        logger.error("[%s] prediction failed for %s: %s", request_id, trip_id, e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    pred_time_ms = (time.perf_counter() - t0) * 1000

    # Build the predicted route geometry (list of [lon, lat] segments)
    predicted_arc_ids = result["predicted_arcs"]
    route_coords = []
    for arc_id in predicted_arc_ids:
        if arc_id < len(meta.arcs):
            from_node, to_node = meta.arcs[arc_id]
            if from_node < len(meta.nodes_lonlat) and to_node < len(meta.nodes_lonlat):
                route_coords.append({
                    "arc_id": int(arc_id),
                    "from": meta.nodes_lonlat[from_node].tolist(),
                    "to": meta.nodes_lonlat[to_node].tolist(),
                })

    # Score against ground truth (server-side only; score goes to response, truth does NOT)
    accuracy = None
    pred_tag = "⚡ [REAL MODEL]" if getattr(predictor, "is_real", False) else "⚠️ [DUMMY MODEL]"
    if trip_id in ground_truths:
        accuracy = score_trip(predicted_arc_ids, ground_truths[trip_id])
        logger.info("[%s] %s trip %s  accuracy=%.4f  pred_arcs=%d  gt_arcs=%d  time=%.1fms",
                    request_id, pred_tag, trip_id, accuracy,
                    len(predicted_arc_ids), len(ground_truths[trip_id]), pred_time_ms)
    else:
        logger.info("[%s] %s trip %s  (no ground truth)  pred_arcs=%d  time=%.1fms",
                    request_id, pred_tag, trip_id, len(predicted_arc_ids), pred_time_ms)

    response = {
        "trip_id": trip_id,
        "predicted_arcs": [int(a) for a in predicted_arc_ids],
        "route_geometry": route_coords,
        "n_predicted_arcs": len(predicted_arc_ids),
        "prediction_time_ms": round(pred_time_ms, 1),
        "track": meta.track_lonlat.tolist(),
        "timestamps": meta.track_time.tolist() if hasattr(meta, "track_time") else [],
        "predictor": result.get("predictor", getattr(predictor, "name", "Unknown")),
        "is_real_predictor": getattr(predictor, "is_real", False),
    }

    # Include accuracy only if ground truth exists (never include the truth itself!)
    if accuracy is not None:
        response["accuracy"] = round(accuracy, 4)

    return response


# ── Health ───────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "trips_loaded": len(trip_registry),
        "predictor": getattr(predictor, "name", type(predictor).__name__),
        "is_real_predictor": getattr(predictor, "is_real", False),
    }
