"""Feature selection evidence on TRAINING trips only (never student/evaluator data), group-aware by trip.
Produces: correlation heatmap, permutation importance (held-out folds), drop-one-feature ablation.
Usage (from models/): python feature_selection.py ../artifacts/data/features.parquet [n_train_trips=20] [n_splits=3]"""
import sys, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
from harness import feature_cols, obs_metrics, SEED

path = sys.argv[1]; n_trips = int(sys.argv[2]) if len(sys.argv) > 2 else 20; n_splits = int(sys.argv[3]) if len(sys.argv) > 3 else 3
df = pd.read_parquet(path); df = df[df.split == "train"]; feats = feature_cols(df)
rng = np.random.default_rng(SEED); keep = rng.choice(sorted(df.trip_id.unique()), size=min(n_trips, df.trip_id.nunique()), replace=False)
df = df[df.trip_id.isin(keep)].reset_index(drop=True); out = "results/feature_selection"; os.makedirs(out, exist_ok=True)
print(f"{len(keep)} training trips, {len(df):,} rows, features: {feats}")
mk = lambda: HistGradientBoostingClassifier(max_iter=100, random_state=SEED)

# 1) correlation (redundancy check)
c = df[feats].corr(); c.to_csv(f"{out}/correlation.csv")
fig, ax = plt.subplots(figsize=(7, 6)); im = ax.imshow(c, vmin=-1, vmax=1, cmap="coolwarm"); ax.set_xticks(range(len(feats))); ax.set_yticks(range(len(feats)))
ax.set_xticklabels(feats, rotation=60, ha="right"); ax.set_yticklabels(feats); fig.colorbar(im); ax.set_title("Feature correlation (train subset)"); fig.tight_layout(); fig.savefig(f"{out}/correlation.png", dpi=200); plt.close(fig)

# 2) permutation importance on held-out folds + 3) drop-one ablation (observation-level top-1 and log loss)
perm, drop, base = [], [], []
for a, b in GroupKFold(n_splits).split(df, df.label, groups=df.trip_id):
    tr, te = df.iloc[a], df.iloc[b]; m = mk().fit(tr[feats], tr.label); p0 = m.predict_proba(te[feats])[:, 1]
    t0, l0 = obs_metrics(te, p0)["top1"], log_loss(te.label, np.clip(p0, 1e-7, 1 - 1e-7)); base.append((t0, l0))
    for f in feats:
        tp = te.copy(); tp[f] = rng.permutation(tp[f].values); p = m.predict_proba(tp[feats])[:, 1]
        perm.append((f, t0 - obs_metrics(te, p)["top1"], log_loss(te.label, np.clip(p, 1e-7, 1 - 1e-7)) - l0))
        rest = [x for x in feats if x != f]; m2 = mk().fit(tr[rest], tr.label); p2 = m2.predict_proba(te[rest])[:, 1]
        drop.append((f, obs_metrics(te, p2)["top1"] - t0, log_loss(te.label, np.clip(p2, 1e-7, 1 - 1e-7)) - l0))
    print("fold done", flush=True)
P = pd.DataFrame(perm, columns=["feature", "top1_drop", "logloss_increase"]).groupby("feature").mean().sort_values("top1_drop", ascending=False)
D = pd.DataFrame(drop, columns=["feature", "top1_change_when_dropped", "logloss_change_when_dropped"]).groupby("feature").mean().loc[P.index]
R = P.join(D); R.to_csv(f"{out}/importance_and_ablation.csv"); print(R.round(4).to_string()); print("baseline top1 / logloss (mean over folds):", np.mean(base, axis=0).round(4))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
R.top1_drop[::-1].plot.barh(ax=ax[0], title="Permutation importance (top-1 drop)"); R.top1_change_when_dropped[::-1].plot.barh(ax=ax[1], title="Drop-one ablation (top-1 change)")
fig.tight_layout(); fig.savefig(f"{out}/importance_and_ablation.png", dpi=200); print("saved to", out)
