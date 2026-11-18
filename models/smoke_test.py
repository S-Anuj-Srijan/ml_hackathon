# SMOKE TEST ONLY: random synthetic data to check the code runs. These numbers mean nothing; never report them.
import numpy as np, pandas as pd
from harness import run_all
rng = np.random.default_rng(0); rows = []
for t in range(30):
    for o in range(40):
        k = 6; true = rng.integers(k)
        for c in range(k):
            d = abs(rng.normal(0 if c == true else 30, 15))
            rows.append((t, o, c, int(c == true), "train" if t < 24 else "student", d, rng.normal(), rng.random()))
df = pd.DataFrame(rows, columns=["trip_id", "obs_idx", "arc_id", "label", "split", "dist_m", "heading_diff", "speed_ratio"])
print(run_all(df, "smoke_results/model_comparison.json")[["cv_macro_f1_mean", "student_top1", "student_log_loss"]])
