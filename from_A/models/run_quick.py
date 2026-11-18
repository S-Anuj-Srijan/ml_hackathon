"""First real model pass on A's features.parquet, on a SUBSET of training trips so it finishes fast.
Usage (from models/):  python run_quick.py ../artifacts/data/features.parquet [n_train_trips=20] [n_splits=3] [models=comma,list]
Student-test trips are always used in full. The subset size is recorded in the output: report it honestly (this is not the final run)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sys, json, time, os
import numpy as np, pandas as pd
from models.harness import zoo, group_cv, evaluate, feature_cols, seed_all, SEED

path = sys.argv[1]; n_trips = int(sys.argv[2]) if len(sys.argv) > 2 else 20; n_splits = int(sys.argv[3]) if len(sys.argv) > 3 else 3
names = sys.argv[4].split(",") if len(sys.argv) > 4 else ["LogReg", "GaussianNB", "HistGB", "RandomForest", "ExtraTrees"]
seed_all(); df = pd.read_parquet(path); feats = feature_cols(df)
rng = np.random.default_rng(SEED); train_trips = sorted(df[df.split == "train"].trip_id.unique())
keep = set(rng.choice(train_trips, size=min(n_trips, len(train_trips)), replace=False))
train = df[(df.split == "train") & df.trip_id.isin(keep)]; stu = df[df.split == "student"]
print(f"features: {feats}\ntrain: {len(keep)} trips / {len(train):,} rows | student: {stu.trip_id.nunique()} trips / {len(stu):,} rows | folds={n_splits}", flush=True)
res = {}; Z = zoo()
for n in names:
    fam, model = Z[n]; t0 = time.time()
    cv, folds = group_cv(model, train, feats, n_splits=n_splits); test, _, _ = evaluate(model, train, stu, feats)
    res[n] = {"family": fam, "config": str(model), "seed": SEED, "n_train_trips": len(keep), "cv_folds": n_splits, **cv, **{f"student_{k}": v for k, v in test.items()}}
    print(f"{n:13s} cvF1={cv['cv_macro_f1_mean']:.3f}±{cv['cv_macro_f1_std']:.3f}  student top1={test['top1']:.3f} F1={test['macro_f1']:.3f} "
          f"logloss={test['log_loss']:.3f}  [{time.time()-t0:.0f}s]", flush=True)
os.makedirs("results", exist_ok=True); json.dump(res, open("results/quick_comparison.json", "w"), indent=2, default=float)
print("saved results/quick_comparison.json  (SUBSET run: not the final comparison)")
