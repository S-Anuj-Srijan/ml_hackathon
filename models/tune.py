"""Bayesian (TPE) hyperparameter search with group-aware CV on the 80 training trips only."""
import json, time, optuna, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from harness import group_cv, evaluate, feature_cols, SEED

def make(params):
    return HistGradientBoostingClassifier(random_state=SEED, **params)

def space(trial):  # edit to match the model you pick; log the ranges in the report
    return dict(learning_rate=trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
                max_iter=trial.suggest_int("max_iter", 100, 500),
                max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 8, 64),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 200, log=True),
                l2_regularization=trial.suggest_float("l2_regularization", 1e-6, 10, log=True))

def run(df, n_trials=40, out="results/optuna"):
    import os; os.makedirs(out, exist_ok=True)
    train, stu = df[df.split == "train"], df[df.split == "student"]; feats = feature_cols(df)
    def objective(trial):  # objective: mean CV macro F1 (maximise). Never touches student/evaluator data.
        cv, _ = group_cv(make(space(trial)), train, feats, n_splits=5)
        return cv["cv_macro_f1_mean"]
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    t0 = time.time(); study.optimize(objective, n_trials=n_trials); cost = time.time() - t0
    study.trials_dataframe().to_csv(f"{out}/trials.csv", index=False)
    base, _, _ = evaluate(make({}), train, stu, feats)
    tuned, _, _ = evaluate(make(study.best_params), train, stu, feats)
    cv_base, _ = group_cv(make({}), train, feats); cv_tuned, _ = group_cv(make(study.best_params), train, feats)
    res = {"best_params": study.best_params, "n_trials": n_trials, "search_seconds": cost,
           "untuned": {**cv_base, **base}, "tuned": {**cv_tuned, **tuned}}
    json.dump(res, open(f"{out}/summary.json", "w"), indent=2, default=float); return res
