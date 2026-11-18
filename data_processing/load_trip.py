"""Load one PathSnap trip and derive per-observation reference labels.
Raw formats (verified on trip 00000001): .track = lon lat t | .nodes = lon lat | .arcs = from to | .route = arc_id per line.
Arc id = 0-based row index in .arcs; node id = 0-based row index in .nodes. Coordinates are lon, lat.
"""
import glob, os
import numpy as np
from pyproj import Transformer

_TO_M = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)  # UTM 32N: area is lon~8.6E, lat~49.8N


def _one(trip_dir, ext):
    files = glob.glob(os.path.join(trip_dir, f"*.{ext}"))
    if len(files) != 1:
        raise FileNotFoundError(f"{trip_dir}: expected exactly 1 .{ext}, found {len(files)}")
    return files[0]


def load_trip(trip_dir, need_route=True):
    """need_route=False for evaluator trips at inference time (labels must not enter the pipeline)."""
    track = np.loadtxt(_one(trip_dir, "track"), ndmin=2)
    nodes = np.loadtxt(_one(trip_dir, "nodes"), ndmin=2)
    arcs = np.loadtxt(_one(trip_dir, "arcs"), dtype=np.int64, ndmin=2)
    tx, ty = _TO_M.transform(track[:, 0], track[:, 1]); nx, ny = _TO_M.transform(nodes[:, 0], nodes[:, 1])
    out = {"t": track[:, 2], "xy": np.c_[tx, ty], "node_xy": np.c_[nx, ny], "arcs": arcs}
    if need_route:
        out["route"] = np.loadtxt(_one(trip_dir, "route"), dtype=np.int64, ndmin=1)
    return out


def point_seg_dist(P, A, B):
    """Distance in metres from points P (n,2) to segments A->B (m,2) -> (n,m)."""
    AB = B - A; L2 = np.maximum((AB ** 2).sum(1), 1e-9)
    t = np.clip(((P[:, None, :] - A[None]) * AB[None]).sum(2) / L2[None], 0, 1)
    proj = A[None] + t[..., None] * AB[None]
    return np.linalg.norm(P[:, None, :] - proj, axis=2)


def label_points(trip):
    """Assign each GPS point to a route arc using a monotone (non-decreasing) DP over route order.
    ASSUMPTION: the vehicle traverses route arcs in listed order and never goes back. Returns (label_arc_ids, dist_to_label_m).
    The route is reference info, not guaranteed truth -- inspect the large-distance points."""
    r = trip["route"]; arcs = trip["arcs"]; N = trip["node_xy"]
    D = point_seg_dist(trip["xy"], N[arcs[r, 0]], N[arcs[r, 1]])      # (n_points, n_route_arcs)
    n, m = D.shape; dp = np.empty_like(D); bp = np.zeros((n, m), dtype=np.int64)
    dp[0] = D[0]
    for i in range(1, n):
        prev = dp[i - 1]; cm = np.minimum.accumulate(prev)
        # argmin over k<=j of prev[k]
        idx = np.zeros(m, dtype=np.int64); best = 0
        for j in range(m):
            if prev[j] < prev[best]: best = j
            idx[j] = best
        bp[i] = idx; dp[i] = D[i] + cm
    j = np.empty(n, dtype=np.int64); j[-1] = int(np.argmin(dp[-1]))
    for i in range(n - 1, 0, -1): j[i - 1] = bp[i, j[i]]
    return r[j], D[np.arange(n), j]
