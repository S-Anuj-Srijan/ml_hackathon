# ml_hackathon

## Part A: data and candidate features

Install in a virtual environment and place the supplied data under
`Student folder/Student folder/{train,student_test}/<trip_id>/`:

```bash
python -m pip install -r requirements-data.txt
python data_pipeline.py --data-root "Student folder/Student folder"
python -m unittest test_data_pipeline -v
```

Outputs are committed in `artifacts/data/`: `features.parquet`,
`observations.parquet`, file inventory, per-trip quality checks, candidate-radius
sensitivity, summary JSON, `eda.png`, and `trip_map.html` (open in a browser).
Raw data and local dependency directories are excluded from Git.

For B, the table matches `models/harness.py` directly:

```python
import pandas as pd
from models.harness import run_all
df = pd.read_parquet("artifacts/data/features.parquet")
run_all(df)
```

One row represents one observation/candidate pair. `label` is binary;
`split` is `train` or `student`; trip IDs are strings preserving leading zeros.
All remaining columns are finite numeric features. See
[data_quality_notes.md](data_quality_notes.md) before interpreting scores.
Use the separate observation table for full evaluation denominators and for C's
timestamps/gaps; it includes observations with zero candidates. Do not pass its
reference arc, alignment residual, or route occurrence into a model.

`Network` and `candidate_features` in `data_pipeline.py` can be reused for
inference without loading `.route`. Cache each trip's network/index at startup.
Arc IDs are local to a trip; never use them across networks without the trip ID.

The existing model code belongs to B; this change supplies A's data handoff.
