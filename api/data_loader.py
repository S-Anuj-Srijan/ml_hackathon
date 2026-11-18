"""Prediction input loader; never reads reference route files."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from data_pipeline import Network, read_file, project

@dataclass
class TripMeta:
    trip_id: str
    trip_dir: Path
    track_lonlat: np.ndarray
    track_time: np.ndarray
    track_xy: np.ndarray
    nodes_lonlat: np.ndarray
    arcs: np.ndarray
    network: Network
    @property
    def n_points(self): return len(self.track_time)
    @property
    def n_nodes(self): return len(self.nodes_lonlat)
    @property
    def n_arcs(self): return len(self.arcs)
    @property
    def bbox(self):
        lo, hi = self.track_lonlat.min(0), self.track_lonlat.max(0)
        return dict(min_lon=float(lo[0]), min_lat=float(lo[1]), max_lon=float(hi[0]), max_lat=float(hi[1]))

def load_trip_data(trip_dir):
    trip_dir = Path(trip_dir)
    def read(kind):
        files = list(trip_dir.glob('*.'+kind))
        if len(files) != 1: raise ValueError(f'Expected one {kind} file in {trip_dir}')
        return read_file(files[0], kind)
    track, nodes, arcs = read('track'), read('nodes'), read('arcs')
    if (np.diff(track[:, 2]) <= 0).any() or (np.abs(track[:, :2]) > [180, 90]).any():
        raise ValueError('Invalid coordinates or timestamps')
    network = Network(nodes, arcs)
    return TripMeta(trip_dir.name, trip_dir, track[:, :2], track[:, 2], project(track[:, :2], network.transformer), nodes, arcs, network)
