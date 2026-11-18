"""Does dropping features hurt? Compares ALL features vs a REDUCED set (HistGB, group CV on train trips + student test).
Usage (from models/): python compare_features.py ../artifacts/data/features.parquet 20 offset_m,from_degree,to_degree"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sys, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from models.harness import feature_cols, group_cv, evaluate, SEED

path, n_trips, drop = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(",")
df = pd.read_parquet(path); allf = feature_cols(df); assert all(d in allf for d in drop), f"unknown feature in {drop}; have {allf}"
rng = np.random.default_rng(SEED); keep = rng.choice(sorted(df[df.split == "train"].trip_id.unique()), size=n_trips, replace=False)
tr = df[(df.split == "train") & df.trip_id.isin(keep)]; te = df[df.split == "student"]
mk = lambda: HistGradientBoostingClassifier(max_iter=100, random_state=SEED); rows = []
for name, f in (("all", allf), ("reduced", [x for x in allf if x not in drop])):
    t0 = time.time(); cv, _ = group_cv(mk(), tr, f, n_splits=3); r, _, _ = evaluate(mk(), tr, te, f)
    rows.append(dict(set=name, n_features=len(f), cv_macro_f1=cv["cv_macro_f1_mean"], cv_top1=cv["cv_top1_mean"], student_top1=r["top1"],
                     student_macro_f1=r["macro_f1"], student_log_loss=r["log_loss"], infer_ms_per_1k=r["inference_ms_per_1k_cands"], secs=time.time() - t0))
print(f"dropped: {drop}  | {len(keep)} train trips"); print(pd.DataFrame(rows).round(4).to_string(index=False))
