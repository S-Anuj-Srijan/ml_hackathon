"""Predictor module – supports Real ML Model (Member B) and Dummy heuristic.

Features:
- Loads models/best_model.joblib: {"model", "features", "params", "seed", "threshold"}
- Constructs feature columns in the EXACT order specified by `features` (6 features)
- Uses per-trip UTM zone derived dynamically from coordinates
- Clear distinction between Real Predictor and Dummy Predictor in logs and UI
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("pathsnap")

try:
    from data_pipeline import candidate_features
except ImportError:
    candidate_features = None


class BasePredictor(ABC):
    """Interface that every predictor must satisfy."""
    name: str = "BasePredictor"
    is_real: bool = False

    @abstractmethod
    def predict(self, trip_meta) -> dict:
        """Return {"predicted_arcs": list[int], "predictor": str, ...}."""
        ...


class DummyPredictor(BasePredictor):
    """Fast nearest-arc spatial heuristic using cKDTree in per-trip UTM coordinates."""
    name = "DummyPredictor (Nearest Arc Heuristic)"
    is_real = False

    def predict(self, meta) -> dict:
        _, nearest_indices = meta.spatial_index.query(meta.track_xy)

        deduped = [int(nearest_indices[0])]
        for a in nearest_indices[1:]:
            a_int = int(a)
            if a_int != deduped[-1]:
                deduped.append(a_int)

        return {
            "predicted_arcs": deduped,
            "predictor": self.name,
            "is_real": False,
        }


class RealMLPredictor(BasePredictor):
    """Trained ML Model (HistGradientBoosting / Member B).

    Generates candidates via STRtree, computes the exact 6 features,
    and ranks candidates per GPS fix using class probability.
    """
    is_real = True

    def __init__(self, model: Any, features: list[str], threshold: float = 0.5):
        self.model = model
        self.features = features
        self.threshold = threshold
        clf_name = type(model).__name__
        self.name = f"RealModel ({clf_name} - {len(features)} feats)"

    def predict(self, meta) -> dict:
        if meta.network is None or candidate_features is None:
            logger.warning("Network or candidate_features unavailable; falling back to spatial index")
            return DummyPredictor().predict(meta)

        # Generate candidates using per-trip network
        cands, _ = candidate_features(meta.network, meta.track_raw)

        if len(cands) == 0:
            logger.warning("No candidates generated; falling back to nearest arc")
            return DummyPredictor().predict(meta)

        # Extract features in the EXACT order defined in the trained model
        X = cands[self.features]

        # Predict probability for each candidate
        if hasattr(self.model, "predict_proba"):
            cands["p"] = self.model.predict_proba(X)[:, 1]
        else:
            cands["p"] = self.model.predict(X)

        # Pick top-1 candidate per observation
        best = cands.loc[cands.groupby("obs_id")["p"].idxmax()]
        raw_arcs = best["arc_id"].tolist()

        if len(raw_arcs) == 0:
            return DummyPredictor().predict(meta)

        # Deduplicate consecutive repeat arcs
        deduped = [int(raw_arcs[0])]
        for a in raw_arcs[1:]:
            a_int = int(a)
            if a_int != deduped[-1]:
                deduped.append(a_int)

        return {
            "predicted_arcs": deduped,
            "predictor": self.name,
            "is_real": True,
            "n_candidates": len(cands),
        }


# ── Model Factory ────────────────────────────────────────────────────
_MODEL_PATHS = [
    Path(__file__).resolve().parent.parent / "models" / "best_model.joblib",
    Path(__file__).resolve().parent.parent / "best_model.joblib",
]


def get_predictor() -> BasePredictor:
    """Return RealMLPredictor if best_model.joblib exists, else DummyPredictor."""
    model_path = next((p for p in _MODEL_PATHS if p.exists()), None)
    if model_path:
        try:
            import joblib
            loaded = joblib.load(model_path)
            if isinstance(loaded, dict) and "model" in loaded:
                model = loaded["model"]
                features = loaded.get("features", [
                    "distance_m", "offset_fraction", "heading_difference_rad",
                    "speed_mps", "arc_length_m", "distance_rank"
                ])
                threshold = loaded.get("threshold", 0.5)
                logger.info("✓ Loaded real model from %s (%d features: %s)",
                            model_path.name, len(features), features)
                return RealMLPredictor(model, features, threshold)
            else:
                # Bare model object
                logger.info("✓ Loaded model object from %s", model_path.name)
                features = [
                    "distance_m", "offset_fraction", "heading_difference_rad",
                    "speed_mps", "arc_length_m", "distance_rank"
                ]
                return RealMLPredictor(loaded, features)
        except Exception as e:
            logger.error("Failed to load model from %s: %s — falling back to DummyPredictor", model_path, e)

    logger.info("ℹ No best_model.joblib found — running DummyPredictor (nearest arc)")
    return DummyPredictor()
