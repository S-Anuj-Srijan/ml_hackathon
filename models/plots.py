"""Comparison plots from REAL results/model_comparison.json (written by harness.run_all).
Never put plots made from the synthetic smoke-test data into the report."""
import json, sys, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    return json.load(open(path))


def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.tight_layout(); fig.savefig(f"{out}/{name}.png", dpi=200); plt.close(fig)


def plot_cv_f1(res, out):
    names = list(res); m = [res[n]["cv_macro_f1_mean"] for n in names]; s = [res[n]["cv_macro_f1_std"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, m, yerr=s, capsize=4); ax.set_ylabel("CV macro F1 (mean ± std, 5 group folds)")
    ax.set_title("Cross-validation macro F1 by model"); plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    _save(fig, out, "cv_macro_f1")


def plot_student(res, out):
    names = list(res); x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w/2, [res[n]["student_top1"] for n in names], w, label="Top-1 accuracy")
    ax.bar(x + w/2, [res[n]["student_macro_f1"] for n in names], w, label="Macro F1")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right"); ax.legend()
    ax.set_title("Student-test performance (10 trips)"); _save(fig, out, "student_top1_f1")


def plot_logloss(res, out):
    names = list(res)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, [res[n]["student_log_loss"] for n in names]); ax.set_ylabel("Log loss (lower is better)")
    ax.set_title("Student-test log loss"); plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    _save(fig, out, "student_log_loss")


def plot_cv_vs_student(res, out):
    """Overfitting check: big gap between CV and student scores is a warning sign."""
    names = list(res); x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w/2, [res[n]["cv_macro_f1_mean"] for n in names], w, label="CV macro F1")
    ax.bar(x + w/2, [res[n]["student_macro_f1"] for n in names], w, label="Student macro F1")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right"); ax.legend()
    ax.set_title("CV vs student macro F1"); _save(fig, out, "cv_vs_student")


def plot_mlp_loss(fitted_pipeline, out):
    """Only for models with epoch history (MLP). Pass the fitted pipeline; do NOT fake curves for others."""
    mlp = fitted_pipeline[-1]; fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(mlp.loss_curve_, label="train loss")
    if getattr(mlp, "validation_scores_", None) is not None:
        ax.plot(mlp.validation_scores_, label="validation score (accuracy)")
    ax.set_xlabel("epoch"); ax.legend(); ax.set_title("MLP training history"); _save(fig, out, "mlp_history")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "results/model_comparison.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "results/plots"
    r = load(src)
    for f in (plot_cv_f1, plot_student, plot_logloss, plot_cv_vs_student): f(r, out)
    print("saved plots to", out)
