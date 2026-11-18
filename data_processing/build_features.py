"""Build data/features.parquet from the raw folders. Usage:
  python build_features.py "<...>/Student folder" [radius] [k]
Expects <root>/train/<trip>/ and <root>/student_test/<trip>/. Never point this at evaluator data."""
import sys, os, time, json
import numpy as np, pandas as pd
from load_trip import load_trip, label_points
from candidates import trip_features, trip_quality, FEATURES

root = sys.argv[1]; radius = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0; k = int(sys.argv[3]) if len(sys.argv) > 3 else 12
frames, report = [], []
for folder, split in (("train", "train"), ("student_test", "student")):
    for trip_id in sorted(os.listdir(os.path.join(root, folder))):
        d = os.path.join(root, folder, trip_id); t0 = time.time()
        try:
            tr = load_trip(d); lab, ld = label_points(tr)
            df = trip_features(tr, radius, k, lab); df.insert(0, "trip_id", int(trip_id)); df.insert(4, "split", split)
            q = trip_quality(df, len(lab)); q.update(trip_id=trip_id, split=split, label_dist_median=float(np.median(ld)),
                                                     label_dist_max=float(ld.max()), n_label_over_50m=int((ld > 50).sum()), secs=round(time.time() - t0, 1))
            frames.append(df); report.append(q); print(q, flush=True)
        except Exception as e:
            print("FAILED", trip_id, repr(e), flush=True); report.append({"trip_id": trip_id, "split": split, "error": repr(e)})
os.makedirs("../data", exist_ok=True); os.makedirs("../results", exist_ok=True)
out = pd.concat(frames, ignore_index=True)
cols = ["trip_id", "obs_idx", "arc_id", "label", "split"] + FEATURES
out[cols].to_parquet("../data/features.parquet", index=False)
pd.DataFrame(report).to_csv("../results/data_quality_per_trip.csv", index=False)
print("rows:", len(out), "| overall miss rate:", 1 - out.groupby(["trip_id", "obs_idx"]).label.max().mean())
