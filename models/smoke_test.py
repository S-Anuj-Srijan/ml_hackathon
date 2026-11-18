import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# SMOKE TEST ONLY: random synthetic data to check the code runs. These numbers mean nothing; never report them.
import numpy as np, pandas as pd
from models.harness import run_all
rng = np.random.default_rng(0); rows = []
for t in range(30):
    for o in range(40):
        k = 6; true = rng.integers(k)
        for c in range(k):
            d = abs(rng.normal(0 if c == true else 30, 15))
            rows.append((t, o, c, int(c == true), "train" if t < 24 else "student", d, rng.normal(), rng.random()))
df = pd.DataFrame(rows, columns=["trip_id", "obs_idx", "arc_id", "label", "split", "distance_m", "heading_difference_rad", "speed_mps"])
from data_pipeline import FEATURE_COLUMNS
for column in FEATURE_COLUMNS:
    if column not in df: df[column] = rng.random(len(df))
print(run_all(df, "smoke_results/model_comparison.json")[["cv_macro_f1_mean", "student_top1", "student_log_loss"]])
