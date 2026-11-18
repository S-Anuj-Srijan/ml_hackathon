"""Model-zoo + group-aware CV harness for PathSnap (Member B).

Expected feature table (parquet/csv) from Member A, one row per (observation, candidate arc):
  trip_id, obs_idx, arc_id, label (1 if arc == reference arc else 0),
  split in {"train", "student"}, plus numeric feature columns.
Evaluator trips must NEVER be in this table.
"""
import json, time, random, os
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score, confusion_matrix
from sklearn.base import clone

SEED = 42
ID_COLS = ["trip_id", "obs_idx", "arc_id", "label", "split"]
THRESH = 0.5  # fixed on purpose: do not tune on student/evaluator data


def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed)


# Features to leave out of every run, set once per CMD window:  set PATHSNAP_DROP_FEATURES=offset_m,from_degree,to_degree
DROP = [c for c in os.environ.get("PATHSNAP_DROP_FEATURES", "").split(",") if c]


def feature_cols(df):
    return [c for c in df.columns if c not in ID_COLS and c not in DROP]


def obs_metrics(df, proba):
    """Observation-level Top-1 and candidate miss rate.
    Miss = true arc absent from candidate set -> counted WRONG in top-1 (denominator = all observations)."""
    d = df[["trip_id", "obs_idx", "label"]].copy()
    d["p"] = proba
    g = d.groupby(["trip_id", "obs_idx"], sort=False)
    has_pos = g["label"].max()
    top = d.loc[g["p"].idxmax()].set_index(["trip_id", "obs_idx"])["label"]
    n = len(has_pos)
    return {"top1": float(top.sum() / n), "miss_rate": float(1 - has_pos.mean()),
            "top1_given_hit": float(top[has_pos == 1].mean())}


def cand_metrics(y, proba):
    pred = (proba >= THRESH).astype(int)
    p = np.clip(proba, 1e-7, 1 - 1e-7)
    return {"macro_f1": f1_score(y, pred, average="macro"),
            "precision": precision_score(y, pred, average="macro", zero_division=0),
            "recall": recall_score(y, pred, average="macro", zero_division=0),
            "log_loss": log_loss(y, p, labels=[0, 1])}


def evaluate(model, tr, te, feats):
    t0 = time.perf_counter(); m = clone(model).fit(tr[feats], tr["label"]); fit_s = time.perf_counter() - t0
    t0 = time.perf_counter(); proba = m.predict_proba(te[feats])[:, 1]; inf_s = time.perf_counter() - t0
    out = {**cand_metrics(te["label"].values, proba), **obs_metrics(te, proba),
           "train_time_s": fit_s, "inference_ms_per_1k_cands": 1000 * inf_s / (len(te) / 1000)}
    return out, m, proba


def group_cv(model, train, feats, n_splits=5, seed=SEED):
    """GroupKFold by trip_id -> no trip leaks across folds."""
    gkf = GroupKFold(n_splits=n_splits)
    rows = []
    for k, (a, b) in enumerate(gkf.split(train, train["label"], groups=train["trip_id"])):
        r, _, _ = evaluate(model, train.iloc[a], train.iloc[b], feats); r["fold"] = k; rows.append(r)
    f = pd.DataFrame(rows)
    return {"cv_macro_f1_mean": f.macro_f1.mean(), "cv_macro_f1_std": f.macro_f1.std(),
            "cv_top1_mean": f.top1.mean(), "cv_logloss_mean": f.log_loss.mean()}, f


def zoo():
    """>=5 models, different families. Edit configs; record whatever you use in the report."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    return {
        "LogReg": ("Linear", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED))),
        "GaussianNB": ("Probabilistic", GaussianNB()),
        "KNN": ("Instance-based", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance"))),
        "RandomForest": ("Ensemble/bagging", RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=SEED)),
        "ExtraTrees": ("Ensemble/bagging", ExtraTreesClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=SEED)),
        "HistGB": ("Boosting", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=SEED)),
        "MLP": ("Neural net", make_pipeline(StandardScaler(), MLPClassifier((64, 32), early_stopping=True, max_iter=300, random_state=SEED))),
    }


def run_all(df, out_json="results/model_comparison.json"):
    import os; os.makedirs(os.path.dirname(out_json), exist_ok=True); seed_all()
    train, stu = df[df.split == "train"], df[df.split == "student"]
    feats = feature_cols(df); results = {}
    for name, (family, model) in zoo().items():
        cv, folds = group_cv(model, train, feats)
        test, _, _ = evaluate(model, train, stu, feats)
        results[name] = {"family": family, "config": str(model), "seed": SEED, **cv,
                         **{f"student_{k}": v for k, v in test.items()}}
        folds.to_csv(out_json.replace(".json", f"_{name}_folds.csv"), index=False)
        print(name, round(cv["cv_macro_f1_mean"], 4), round(test["top1"], 4))
    json.dump(results, open(out_json, "w"), indent=2, default=float)
    return pd.DataFrame(results).T
