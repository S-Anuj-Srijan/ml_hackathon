"""Part A: reproducible inventory, route-derived labels, and GPS-only candidates.

Run: python data_pipeline.py --data-root "Student folder/Student folder"
Only train and student_test are read. See A_PHASE1_README.md for methods and handoff.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import shapely
from shapely import STRtree

FEATURE_COLUMNS = ["distance_m", "offset_m", "offset_fraction", "heading_difference_rad",
                   "speed_mps", "arc_length_m", "distance_rank", "from_degree", "to_degree"]
FORMATS = {"track": (3, float), "nodes": (2, float), "arcs": (2, int), "route": (1, int)}


def read_file(path, kind):
    width, dtype = FORMATS[kind]
    data = np.loadtxt(path, dtype=float, ndmin=2)
    if data.shape[1] != width or not np.isfinite(data).all():
        raise ValueError(f"Invalid shape/nonfinite values: {path}")
    if dtype is int and not np.equal(data, np.floor(data)).all():
        raise ValueError(f"Noninteger IDs: {path}")
    return data.astype(dtype)


def project(points, transformer):
    return np.column_stack(transformer.transform(points[:, 0], points[:, 1]))


class Network:
    def __init__(self, nodes, arcs):
        if (np.abs(nodes[:, 0]) > 180).any() or (np.abs(nodes[:, 1]) > 90).any():
            raise ValueError("Implausible longitude/latitude")
        if arcs.min() < 0 or arcs.max() >= len(nodes):
            raise ValueError("Arc endpoint outside node table")
        lon, lat = np.median(nodes, axis=0)
        self.epsg = (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1
        self.transformer = Transformer.from_crs(4326, self.epsg, always_xy=True)
        self.nodes, self.arcs = nodes, arcs
        self.xy = project(nodes, self.transformer)
        self.start, self.end = self.xy[arcs[:, 0]], self.xy[arcs[:, 1]]
        self.delta = self.end - self.start
        self.length = np.linalg.norm(self.delta, axis=1)
        self.lines = shapely.linestrings(np.stack([self.start, self.end], axis=1))
        self.tree = STRtree(self.lines)
        self.degree = np.bincount(arcs.ravel(), minlength=len(nodes))
        graph = coo_matrix((np.ones(len(arcs)), (arcs[:, 0], arcs[:, 1])), shape=(len(nodes), len(nodes))).tocsr()
        self.components = connected_components(graph, directed=False, return_labels=False)
        self.strong_components = connected_components(graph, directed=True, connection="strong", return_labels=False)

    def distances(self, point, ids):
        delta = self.delta[ids]
        frac = np.clip(np.sum((point - self.start[ids]) * delta, axis=1) /
                       np.maximum(self.length[ids] ** 2, 1e-20), 0, 1)
        dist = np.linalg.norm(point - (self.start[ids] + frac[:, None] * delta), axis=1)
        return dist, frac


def align_route(network, xy, timestamps, route):
    """Monotone route-occurrence DP; no claim of observed per-point ground truth.

    Distance is the main cost. A weak elapsed-time progress prior resolves ties
    at route revisits. No route start/end anchoring or constant-speed constraint.
    """
    if route.min() < 0 or route.max() >= len(network.arcs):
        raise ValueError("Route contains invalid arc IDs")
    starts = np.r_[0, np.cumsum(network.length[route])[:-1]]
    total = max(float(network.length[route].sum()), 1)
    elapsed = (timestamps - timestamps[0]) / max(timestamps[-1] - timestamps[0], 1)
    previous = np.zeros(len(route))
    back = np.zeros((len(xy), len(route)), dtype=np.int32)
    for i, point in enumerate(xy):
        dist, fraction = network.distances(point, route)
        progress = (starts + fraction * network.length[route]) / total
        cost = (dist / 20) ** 2 + 0.05 * (progress - elapsed[i]) ** 2
        minima = np.minimum.accumulate(previous)
        indices = np.maximum.accumulate(np.where(previous == minima, np.arange(len(route)), 0))
        back[i] = indices
        previous = cost + minima
    occurrence = np.empty(len(xy), dtype=int)
    occurrence[-1] = np.argmin(previous)
    for i in range(len(xy) - 1, 0, -1):
        occurrence[i - 1] = back[i, occurrence[i]]
    ids = route[occurrence]
    residual = np.array([network.distances(p, np.array([a]))[0][0] for p, a in zip(xy, ids)])
    return ids, occurrence, residual


def candidate_features(network, track, k=32, radius=300):
    """Prediction-safe: takes no route/labels; returns rows and observation metadata."""
    xy = project(track[:, :2], network.transformer)
    dt = np.r_[np.nan, np.diff(track[:, 2])]
    movement = np.vstack([np.zeros(2), np.diff(xy, axis=0)])
    distance = np.linalg.norm(movement, axis=1)
    speed = np.divide(distance, dt, out=np.full(len(dt), np.nan), where=dt > 0)
    heading = np.arctan2(movement[:, 1], movement[:, 0])
    rows, counts = [], []
    for i, point in enumerate(xy):
        ids = network.tree.query(shapely.Point(point), predicate="dwithin", distance=radius)
        dist, frac = network.distances(point, ids)
        order = np.lexsort((ids, dist))[:k]
        ids, dist, frac = ids[order], dist[order], frac[order]
        counts.append(len(ids))
        for rank, (arc, d, f) in enumerate(zip(ids, dist, frac), 1):
            angle = np.arctan2(network.delta[arc, 1], network.delta[arc, 0])
            diff = abs(np.arctan2(np.sin(heading[i] - angle), np.cos(heading[i] - angle)))
            rows.append((i, int(arc), float(d), f * network.length[arc], f,
                         diff if i > 0 and distance[i] > 1 and dt[i] > 0 else np.nan,
                         speed[i], network.length[arc], rank,
                         network.degree[network.arcs[arc, 0]], network.degree[network.arcs[arc, 1]]))
    frame = pd.DataFrame(rows, columns=["obs_id", "arc_id"] + FEATURE_COLUMNS)
    # Fixed physical sentinels, no learned/global imputation: valid values are >= 0.
    frame[['heading_difference_rad', 'speed_mps']] = frame[['heading_difference_rad', 'speed_mps']].fillna(-1.0)
    frame.loc[frame.arc_length_m == 0, 'heading_difference_rad'] = -1.0
    observations = pd.DataFrame({"obs_id": np.arange(len(track)), "timestamp_s": track[:, 2],
                                 "longitude": track[:, 0], "latitude": track[:, 1],
                                 "dt_s": dt, "speed_mps": speed, "candidate_count": counts,
                                 "large_gap": dt > 10})
    return frame, observations


def make_map(network, track, route, output):
    import folium
    m = folium.Map(location=track[:, :2].mean(axis=0)[::-1].tolist(), zoom_start=12)
    # Viewport clip only the display; the spatial index uses the entire network.
    lo, hi = track[:, :2].min(0) - .005, track[:, :2].max(0) + .005
    ends = network.nodes[network.arcs]
    visible = ((ends.max(1) >= lo) & (ends.min(1) <= hi)).all(1)
    roads = folium.FeatureGroup(name="Road network (trip viewport)")
    geometry = {"type": "MultiLineString", "coordinates": ends[visible].tolist()}
    folium.GeoJson(geometry, style_function=lambda _: {"color": "#777", "weight": 1, "opacity": .45}).add_to(roads)
    roads.add_to(m)
    ref = folium.FeatureGroup(name="Reference route")
    folium.GeoJson({"type": "MultiLineString", "coordinates": ends[route].tolist()},
                   style_function=lambda _: {"color": "green", "weight": 4}).add_to(ref)
    ref.add_to(m)
    gps = folium.FeatureGroup(name="Raw GPS")
    for i, (lon, lat, timestamp) in enumerate(track):
        folium.CircleMarker([lat, lon], radius=2, color="blue", fill=True,
                            tooltip=f"Observation {i}, t={timestamp:g}s").add_to(gps)
    gps.add_to(m)
    folium.LayerControl().add_to(m)
    m.get_root().html.add_child(folium.Element('<div style="position:fixed;bottom:25px;left:25px;z-index:9999;background:white;padding:12px">'
        '<b>PathSnap — source data</b><br><span style="color:blue">● Raw GPS</span><br>'
        '<span style="color:green">━ Reference route</span><br><span style="color:#777">━ Road network (viewport)</span></div>'))
    m.fit_bounds([track[:, :2].min(0)[::-1].tolist(), track[:, :2].max(0)[::-1].tolist()])
    m.save(str(output))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("Student folder/Student folder"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/data"))
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--radius", type=float, default=300)
    args = parser.parse_args()
    if args.k < 1 or args.radius <= 0:
        parser.error("k and radius must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    inventory, audits, feature_frames, obs_frames, trials = [], [], [], [], []
    seen, mapped = set(), False
    for partition in ("train", "student_test"):
        trips = sorted((args.data_root / partition).glob("*/*.track"))
        if not trips:
            raise ValueError(f"No trips in {partition}")
        for path in trips:
            trip = path.stem
            if trip in seen:
                raise ValueError(f"Trip appears in multiple partitions: {trip}")
            seen.add(trip)
            arrays, hashes = {}, {}
            for kind in FORMATS:
                file = path.with_suffix('.' + kind)
                arrays[kind] = read_file(file, kind)
                hashes[kind] = hashlib.sha256(file.read_bytes()).hexdigest()
                inventory.append(dict(partition=partition, trip_id=trip, file=file.relative_to(args.data_root).as_posix(),
                                      kind=kind, bytes=file.stat().st_size, rows=len(arrays[kind]),
                                      columns=arrays[kind].shape[1], dtype=str(arrays[kind].dtype), sha256=hashes[kind]))
            track, nodes, arcs, route = (arrays[x] for x in FORMATS)
            route = route.ravel()
            if (np.abs(track[:, 0]) > 180).any() or (np.abs(track[:, 1]) > 90).any():
                raise ValueError(f"Invalid GPS coordinates: {trip}")
            dt = np.diff(track[:, 2])
            if (dt <= 0).any():
                raise ValueError(f"Non-increasing timestamps: {trip}; do not silently reorder")
            net = Network(nodes, arcs)
            xy = project(track[:, :2], net.transformer)
            labels, occurrence, residual = align_route(net, xy, track[:, 2], route)
            features, observations = candidate_features(net, track, args.k, args.radius)
            features["label"] = (features.arc_id.to_numpy() == labels[features.obs_id.to_numpy()]).astype('int8')
            covered = set(features.loc[features.label == 1, "obs_id"])
            observations["true_arc_id"] = labels
            observations["route_occurrence"] = occurrence
            observations["label_residual_m"] = residual
            observations["candidate_hit"] = observations.obs_id.isin(covered)
            for frame in (features, observations):
                frame.insert(0, "trip_id", trip)
                frame.insert(0, "partition", partition)
            # All candidate sensitivity measurements reuse GPS-only candidates; never inject truth.
            for k, radius in sorted(set([(4, 30), (8, 50), (16, 100), (16, 300), (32, 100), (args.k, args.radius)])):
                if k > args.k or radius > args.radius:
                    continue
                eligible = features[(features.distance_rank <= k) & (features.distance_m <= radius)]
                hits = eligible.loc[eligible.label == 1, "obs_id"].nunique()
                trials.append(dict(partition=partition, trip_id=trip, k=k, radius_m=radius,
                                   observations=len(track), misses=len(track)-hits))
            audits.append(dict(partition=partition, trip_id=trip, files=len(list(path.parent.iterdir())), observations=len(track),
                               lon_min=track[:, 0].min(), lon_max=track[:, 0].max(), lat_min=track[:, 1].min(), lat_max=track[:, 1].max(),
                               missing_or_invalid=0, nonpositive_dt=int((dt <= 0).sum()), one_second_steps=int((dt == 1).sum()),
                               steps=len(dt), one_second_share=float((dt == 1).mean()), max_gap_s=dt.max(), gaps_over_10s=int((dt > 10).sum()),
                               nodes=len(nodes), arcs=len(arcs), weak_components=net.components, strong_components=net.strong_components,
                               isolated_nodes=int((net.degree == 0).sum()), zero_length_arcs=int((net.length == 0).sum()),
                               route_arcs=len(route), route_breaks=int((arcs[route[:-1], 1] != arcs[route[1:], 0]).sum()),
                               route_repeated_occurrences=len(route)-len(np.unique(route)), epsg=net.epsg,
                               nodes_hash=hashes['nodes'], arcs_hash=hashes['arcs'],
                               candidate_misses=len(track)-len(covered), no_candidates=int((observations.candidate_count == 0).sum()),
                               label_residual_p95_m=float(np.quantile(residual, .95)), label_residual_over_100m=int((residual > 100).sum())))
            feature_frames.append(features)
            obs_frames.append(observations)
            if not mapped:
                make_map(net, track, route, args.output / "trip_map.html")
                mapped = True
            print(f"{partition}/{trip}: {len(track)} observations, {len(features)} candidates, {len(track)-len(covered)} misses", flush=True)
    pd.DataFrame(inventory).to_csv(args.output / "file_inventory.csv", index=False)
    audit = pd.DataFrame(audits)
    audit.to_csv(args.output / "trip_quality.csv", index=False)
    features = pd.concat(feature_frames, ignore_index=True)
    observations = pd.concat(obs_frames, ignore_index=True)
    handoff = features.rename(columns={'partition': 'split', 'obs_id': 'obs_idx'})
    handoff['split'] = handoff['split'].replace({'student_test': 'student'})
    # Float32 exceeds the precision of source GPS and keeps the handoff compact.
    handoff[FEATURE_COLUMNS] = handoff[FEATURE_COLUMNS].astype('float32')
    handoff.to_parquet(args.output / "features.parquet", index=False, compression='zstd')
    observations.to_parquet(args.output / "observations.parquet", index=False)
    sensitivity = pd.DataFrame(trials).groupby(['partition', 'k', 'radius_m'], as_index=False)[['observations', 'misses']].sum()
    sensitivity['miss_rate'] = sensitivity.misses / sensitivity.observations
    sensitivity.to_csv(args.output / "candidate_sensitivity.csv", index=False)
    summary = dict(k=args.k, radius_m=args.radius, feature_columns=FEATURE_COLUMNS, candidate_rows=len(features),
                   observations=len(observations), partitions={p: dict(trips=len(a), observations=int(a.observations.sum()),
                   misses=int(a.candidate_misses.sum()), miss_rate=float(a.candidate_misses.sum()/a.observations.sum()),
                   no_candidates=int(a.no_candidates.sum())) for p, a in audit.groupby('partition')},
                   unique_networks=len(audit[['nodes_hash', 'arcs_hash']].drop_duplicates()))
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    plots(audit, observations, args.output)
    print(json.dumps(summary, indent=2))


def plots(audit, observations, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].hist(audit.observations, bins=20); ax[0, 0].set(title='Observations per trip', xlabel='Count', ylabel='Trips')
    ax[0, 1].hist(observations.dt_s.dropna(), bins=[.5, 1.5, 2.5, 5, 10, 30, 100, 1000, max(1001, observations.dt_s.max()+1)]); ax[0, 1].set(xscale='log', yscale='log', title='Sampling gaps', xlabel='Seconds', ylabel='Steps')
    ax[1, 0].scatter(observations.longitude, observations.latitude, s=.1, alpha=.15); ax[1, 0].set(title='GPS extent', xlabel='Longitude', ylabel='Latitude')
    scatter = ax[1, 1].scatter(audit.nodes, audit.arcs, c=audit.weak_components); ax[1, 1].set(title='Network size', xlabel='Nodes', ylabel='Directed arcs')
    from matplotlib.ticker import MaxNLocator
    ax[1, 1].xaxis.set_major_locator(MaxNLocator(5))
    fig.colorbar(scatter, ax=ax[1, 1], label='Weak components')
    fig.tight_layout(); fig.savefig(output / 'eda.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    main()
