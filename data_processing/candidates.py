"""Candidate generation + features for one trip. One row per (GPS observation, candidate arc)."""
import numpy as np, pandas as pd, shapely
from shapely import STRtree
from load_trip import load_trip, label_points

FEATURES = ["dist_m", "dist_rank", "dist_gap_to_best", "along_frac", "cos_heading",
            "speed_mps", "arc_len_m", "n_cands"]


def trip_features(trip, radius=100.0, k=12, labels=None):
    """labels: per-observation reference arc ids (training/student trips only). None at inference time."""
    xy, N, arcs = trip["xy"], trip["node_xy"], trip["arcs"]
    A, B = N[arcs[:, 0]], N[arcs[:, 1]]
    lines = shapely.linestrings(np.stack([A, B], axis=1)); tree = STRtree(lines)
    pts = shapely.points(xy)
    pi, li = tree.query(pts, predicate="dwithin", distance=radius)            # candidate pairs within radius
    dist = shapely.distance(pts[pi], lines[li])
    df = pd.DataFrame({"obs_idx": pi, "arc_id": li, "dist_m": dist})
    df = df.sort_values(["obs_idx", "dist_m"]); df["dist_rank"] = df.groupby("obs_idx").cumcount()
    df = df[df.dist_rank < k].copy()                                          # keep k closest per observation
    pi, li = df.obs_idx.values, df.arc_id.values
    arc_len = np.linalg.norm(B[li] - A[li], axis=1); safe = np.maximum(arc_len, 1e-6)
    df["along_frac"] = shapely.line_locate_point(lines[li], pts[pi]) / safe
    # heading + speed from neighbouring fixes (central difference, metres)
    t = trip["t"]; prev_i = np.maximum(np.arange(len(xy)) - 1, 0); next_i = np.minimum(np.arange(len(xy)) + 1, len(xy) - 1)
    v = (xy[next_i] - xy[prev_i]); dt = np.maximum(t[next_i] - t[prev_i], 1e-6)
    spd = np.linalg.norm(v, axis=1) / dt; hv = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    av = (B[li] - A[li]) / safe[:, None]
    df["cos_heading"] = (hv[pi] * av).sum(1) * (spd[pi] > 0.5)                 # 0 when ~stationary (heading unreliable)
    df["speed_mps"] = spd[pi]; df["arc_len_m"] = arc_len
    df["dist_gap_to_best"] = df.dist_m - df.groupby("obs_idx").dist_m.transform("min")
    df["n_cands"] = df.groupby("obs_idx").arc_id.transform("size")
    if labels is not None:
        df["label"] = (df.arc_id.values == np.asarray(labels)[df.obs_idx.values]).astype(int)
    return df.reset_index(drop=True)


def trip_quality(df, n_obs):
    """Candidate miss rate: share of observations whose reference arc is NOT in the candidate set. Also observations with 0 candidates."""
    hit = df.groupby("obs_idx").label.max(); n_with = df.obs_idx.nunique()
    return {"n_obs": n_obs, "obs_with_no_candidates": n_obs - n_with, "miss_rate": 1 - hit.sum() / n_obs}
