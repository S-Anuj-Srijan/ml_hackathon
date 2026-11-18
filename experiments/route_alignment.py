
import numpy as np
from pathlib import Path

trip = Path("data/Student folder/train/00000001")

nodes = np.loadtxt(trip / "00000001.nodes")
arcs = np.loadtxt(trip / "00000001.arcs", dtype=int)
route = np.loadtxt(trip / "00000001.route", dtype=int)
track = np.loadtxt(trip / "00000001.track")

# Construct the reference route's directed line segments.
route_arcs = arcs[route]
start = nodes[route_arcs[:, 0]]
end = nodes[route_arcs[:, 1]]

# Local metric coordinates (suitable for this small geographic area).
lat0 = np.mean(track[:, 1])
scale_x = 111320 * np.cos(np.radians(lat0))
scale_y = 111320

def metric(coords):
    return np.column_stack((
        coords[:, 0] * scale_x,
        coords[:, 1] * scale_y
    ))

gps = metric(track[:, :2])
a = metric(start)
b = metric(end)

# Find the closest point on each route segment to each GPS observation.
v = b - a
denom = np.sum(v * v, axis=1)
denom[denom == 0] = 1e-12

distances = np.empty((len(gps), len(route)))

for i, point in enumerate(gps):
    t = np.sum((point - a) * v, axis=1) / denom
    t = np.clip(t, 0, 1)
    projection = a + t[:, None] * v
    distances[i] = np.linalg.norm(projection - point, axis=1)

nearest_index = np.argmin(distances, axis=1)
nearest_distance = distances[
    np.arange(len(gps)), nearest_index
]

print("\nROUTE ALIGNMENT REPORT")
print("----------------------")
print("GPS observations:", len(gps))
print("Reference arcs:", len(route))
print("Median distance (m):", round(float(np.median(nearest_distance)), 2))
print("90th percentile (m):", round(float(np.percentile(nearest_distance, 90)), 2))

for threshold in [20, 50, 100]:
    percentage = np.mean(nearest_distance <= threshold) * 100
    print(f"Within {threshold} m: {percentage:.2f}%")

progress = np.diff(nearest_index)
print("Non-decreasing nearest arc indices:",
      f"{np.mean(progress >= 0) * 100:.2f}%")

print("First GPS point -> route arc index:",
      nearest_index[0])
print("Last GPS point -> route arc index:",
      nearest_index[-1])

print("\nFirst 15 nearest route indices:")
print(nearest_index[:15])
