# Synthetic mini-trip (fake data) to check the code runs: straight road of 3 arcs, noisy points moving along it.
import numpy as np, tempfile, os
from load_trip import load_trip, label_points
d = tempfile.mkdtemp(); rng = np.random.default_rng(1)
lon0, lat0 = 8.58, 49.85
nodes = np.array([[lon0 + 0.001 * k, lat0] for k in range(4)] + [[lon0, lat0 + 0.002]])
arcs = np.array([[0, 1], [1, 2], [2, 3], [1, 0], [0, 4]])      # arcs 0,1,2 forward; 3 reverse; 4 side road
np.savetxt(f"{d}/a.nodes", nodes); np.savetxt(f"{d}/a.arcs", arcs, fmt="%d"); np.savetxt(f"{d}/a.route", [0, 1, 2], fmt="%d")
t = np.arange(1, 61.0); lon = lon0 + 0.003 * (t - 1) / 59; lat = lat0 + rng.normal(0, 1e-5, 60); lon += rng.normal(0, 1e-5, 60)
np.savetxt(f"{d}/a.track", np.c_[lon, lat, t])
tr = load_trip(d); lab, dist = label_points(tr)
print("labels:", lab.tolist()); assert (np.diff(lab) >= 0).all() and set(lab) == {0, 1, 2}; print("OK, max dist (m):", round(dist.max(), 1))
