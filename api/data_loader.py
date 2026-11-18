"""Data loader for student-test trips. Reads .track / .nodes / .arcs files.

Constructs:
- Projected coordinates in UTM 32N (meters)
- Pre-built spatial index (scipy cKDTree) over road arc midpoints
- Road graph adjacency list

Ground-truth .route files are NEVER loaded here — see scoring.py.
"""
from __future__ import annotations

import glob
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
from typing import Any

import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree

_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)


@dataclass
class TripMeta:
    """Everything the predictor, decoder, and API need (strictly excluding ground-truth)."""
    trip_id: str
    trip_dir: Path
    track_lonlat: np.ndarray        # (n, 2) [lon, lat]
    track_time: np.ndarray          # (n,) timestamps
    track_xy: np.ndarray            # (n, 2) UTM 32N [x, y] in meters
    nodes_lonlat: np.ndarray        # (m, 2) [lon, lat]
    nodes_xy: np.ndarray            # (m, 2) UTM 32N [x, y] in meters
    arcs: np.ndarray                # (k, 2) [from_node, to_node]
    arc_midpoints_xy: np.ndarray    # (k, 2) UTM 32N [x, y] in meters
    spatial_index: Any              # scipy.spatial.cKDTree over arc midpoints
    graph: dict[int, list[tuple[int, int]]]  # from_node -> [(to_node, arc_id), ...]
    n_points: int = 0
    n_nodes: int = 0
    n_arcs: int = 0
    bbox: dict = field(default_factory=dict)


def _one(trip_dir: Path, ext: str) -> str:
    files = glob.glob(str(trip_dir / f"*.{ext}"))
    if len(files) != 1:
        raise FileNotFoundError(f"{trip_dir}: expected 1 .{ext}, found {len(files)}")
    return files[0]


def load_trip_data(trip_dir: Path) -> TripMeta:
    """Load a trip folder (track, nodes, arcs), project to meters, build graph and spatial index."""
    trip_dir = Path(trip_dir)
    tid = trip_dir.name

    track = np.loadtxt(_one(trip_dir, "track"), ndmin=2)
    nodes = np.loadtxt(_one(trip_dir, "nodes"), ndmin=2)
    arcs  = np.loadtxt(_one(trip_dir, "arcs"), dtype=np.int64, ndmin=2)

    track_lonlat = track[:, :2]
    track_time   = track[:, 2]

    # Convert WGS84 (lon, lat) to UTM 32N (meters)
    tx, ty = _TO_UTM.transform(track_lonlat[:, 0], track_lonlat[:, 1])
    track_xy = np.column_stack([tx, ty])

    nx, ny = _TO_UTM.transform(nodes[:, 0], nodes[:, 1])
    nodes_xy = np.column_stack([nx, ny])

    # Precompute arc midpoints in UTM coordinates
    from_nodes = arcs[:, 0]
    to_nodes = arcs[:, 1]
    arc_midpoints_xy = (nodes_xy[from_nodes] + nodes_xy[to_nodes]) / 2.0

    # Build spatial index (cKDTree) over arc midpoints once at startup
    spatial_index = cKDTree(arc_midpoints_xy)

    # Build road network graph adjacency list: node_u -> list of (node_v, arc_id)
    graph: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for arc_id, (u, v) in enumerate(arcs):
        graph[int(u)].append((int(v), int(arc_id)))

    bbox = {
        "min_lon": float(track_lonlat[:, 0].min()),
        "max_lon": float(track_lonlat[:, 0].max()),
        "min_lat": float(track_lonlat[:, 1].min()),
        "max_lat": float(track_lonlat[:, 1].max()),
    }

    return TripMeta(
        trip_id=tid,
        trip_dir=trip_dir,
        track_lonlat=track_lonlat,
        track_time=track_time,
        track_xy=track_xy,
        nodes_lonlat=nodes,
        nodes_xy=nodes_xy,
        arcs=arcs,
        arc_midpoints_xy=arc_midpoints_xy,
        spatial_index=spatial_index,
        graph=dict(graph),
        n_points=len(track),
        n_nodes=len(nodes),
        n_arcs=len(arcs),
        bbox=bbox,
    )
