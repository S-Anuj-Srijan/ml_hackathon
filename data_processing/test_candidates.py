# Synthetic check only (fake trip): code runs, labels line up, miss rate computed.
import numpy as np, tempfile
from load_trip import load_trip, label_points
from candidates import trip_features, trip_quality
exec(open("test_load_trip.py").read().split("tr = load_trip(d)")[0])   # reuse fake-trip builder
tr = load_trip(d); lab, _ = label_points(tr); df = trip_features(tr, 100, 5, lab)
print(df.head(8).round(2).to_string()); print(trip_quality(df, len(lab)))
assert df.groupby("obs_idx").label.sum().max() <= 1 and df.groupby("obs_idx").size().max() <= 5; print("OK")
