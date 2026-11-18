"""Server-side scoring — compares predicted arcs to ground truth.

CRITICAL RULE: The ground truth NEVER reaches the client, the model, or
the UI response payload.  Only the numeric accuracy score is returned.
Nothing is hardcoded — truth is always read from .route files.
"""
from __future__ import annotations

import glob
from pathlib import Path
import numpy as np
from backend.paths import PROJECT_ROOT

BASE_DIR = PROJECT_ROOT
DEFAULT_GROUND_TRUTH_DIR = BASE_DIR / "ground_truth"


def load_ground_truth(trip_id: str, trip_dir: Path | None = None, ground_truth_dir: Path | None = None) -> np.ndarray:
    """Read the ground truth .route file for a given trip ID.
    
    Priority order:
    1. ground_truth/<trip_id>.route (evaluator ground truth directory)
    2. trip_dir/*.route (legacy fallback if present)
    """
    gt_dir = ground_truth_dir or DEFAULT_GROUND_TRUTH_DIR
    target_file = gt_dir / f"{trip_id}.route"
    
    if target_file.exists():
        return np.loadtxt(target_file, dtype=np.int64, ndmin=1)
        
    # Check if trip_dir was provided
    if trip_dir is not None:
        trip_dir = Path(trip_dir)
        route_files = glob.glob(str(trip_dir / "*.route"))
        if route_files:
            return np.loadtxt(route_files[0], dtype=np.int64, ndmin=1)

    raise FileNotFoundError(f"No ground truth route found for trip {trip_id} in {gt_dir} or {trip_dir}")


def score_trip(predicted_arcs: list[int], ground_truth: np.ndarray) -> float:
    """Compute accuracy of predicted route vs ground truth.

    Metric: Jaccard similarity — |intersection| / |union| of arc sets.
    This is order-independent and handles routes of different lengths.
    """
    pred_set = set(int(a) for a in predicted_arcs)
    gt_set   = set(int(a) for a in ground_truth)

    if not pred_set and not gt_set:
        return 1.0
    if not pred_set or not gt_set:
        return 0.0

    intersection = pred_set & gt_set
    union        = pred_set | gt_set

    return len(intersection) / len(union)
