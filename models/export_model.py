"""Fit the FINAL model on all 80 training trips and export what C (HMM) and D (API) need.
Usage (from models/):  python export_model.py ../artifacts/data/features.parquet results/optuna/summary.json
Respects PATHSNAP_DROP_FEATURES (set it the same way you did for tuning).
Writes:
  best_model.joblib             {"model", "features", "params", "seed", "threshold"}   (for D's API)
  results/probs_train_oof.parquet   out-of-fold probability per candidate on training trips (for C to tune the decoder honestly)
  results/probs_student.parquet     probability per candidate on the 10 student trips, from the model fitted on all training trips
  results/export_info.json          what was used, so the report can state it"""
import sys, json, time, os
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from harness import feature_cols, obs_metrics, SEED, THRESH

path, summ = sys.argv[1], sys.argv[2]
params = json.load(open(summ))["best_params"]; df = pd.read_parquet(path); feats = feature_cols(df)
tr, st = df[df.split == "train"].reset_index(drop=True), df[df.split == "student"].reset_index(drop=True)
mk = lambda: HistGradientBoostingClassifier(random_state=SEED, **params)
print(f"{len(feats)} features: {feats}\nparams: {params}\ntrain {tr.trip_id.nunique()} trips / {len(tr):,} rows | student {st.trip_id.nunique()} trips", flush=True)
os.makedirs("results", exist_ok=True); ids = ["trip_id", "obs_idx", "arc_id", "label"]
oof = np.zeros(len(tr), dtype=np.float32); t0 = time.time()
for k, (a, b) in enumerate(GroupKFold(5).split(tr, tr.label, groups=tr.trip_id)):      # whole trips held out each fold
    oof[b] = mk().fit(tr.loc[a, feats], tr.label[a]).predict_proba(tr.loc[b, feats])[:, 1]; print(f"oof fold {k+1}/5 done", flush=True)
o = tr[ids].copy(); o["p"] = oof; o.to_parquet("results/probs_train_oof.parquet", index=False)
t1 = time.time(); final = mk().fit(tr[feats], tr.label); fit_s = time.time() - t1
ps = final.predict_proba(st[feats])[:, 1].astype(np.float32); s = st[ids].copy(); s["p"] = ps; s.to_parquet("results/probs_student.parquet", index=False)
joblib.dump({"model": final, "features": feats, "params": params, "seed": SEED, "threshold": THRESH}, "best_model.joblib")
info = {"features": feats, "n_features": len(feats), "params": params, "fit_seconds_all_train": round(fit_s, 1),
        "oof_top1_train": obs_metrics(tr, oof)["top1"], "student_top1": obs_metrics(st, ps)["top1"],
        "student_miss_rate": obs_metrics(st, ps)["miss_rate"], "n_train_trips": int(tr.trip_id.nunique())}
json.dump(info, open("results/export_info.json", "w"), indent=2, default=float); print(json.dumps(info, indent=2, default=float))
print("saved best_model.joblib + results/probs_train_oof.parquet + results/probs_student.parquet")
