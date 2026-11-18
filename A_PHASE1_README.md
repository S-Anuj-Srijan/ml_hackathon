# Part A - Phase 1: data, EDA, candidates and features

This is the complete record of Part A's implemented pipeline, measured results,
validation, assumptions and handoff to B/C/D. It covers 80 training trips and
10 student-test trips. Evaluator data is excluded.

## What was delivered

| Requirement | Implementation and evidence |
| --- | --- |
| Inventory and file formats | 360 files; sizes, SHA-256, rows, columns and parsed dtypes in `file_inventory.csv` |
| GPS and timestamp checks | Coordinate bounds, ordering, one-second steps and large gaps in `trip_quality.csv` |
| Road network audit | Node/arc counts, weak/strong components, isolated nodes, zero-length arcs and per-trip network hashes |
| Reference labels | Chronological route-occurrence alignment in `align_route`; assumptions below |
| Map | `trip_map.html`: raw GPS, road network, reference route, legend and layer control |
| Candidate generation | Per-network UTM projection and STRtree; nearest 32 arcs within 300 metres |
| Feature engineering | Nine numeric features, computed independently of reference routes |
| Coverage assessment | Six K/radius configurations in `candidate_sensitivity.csv` |
| Handoff | `features.parquet`, complete observation audit, EDA PNG, notes and tests |

## Files to use

| File | Purpose |
| --- | --- |
| `data_pipeline.py` | Canonical Part A pipeline; loading, auditing, labeling, candidates, features, map and plots |
| `test_data_pipeline.py` | Geometry, label-order, empty-candidate and full-artifact tests |
| `requirements-data.txt` | Part A dependencies |
| `artifacts/data/features.parquet` | Model-ready candidate rows for B |
| `artifacts/data/observations.parquet` | Every GPS observation, timestamps, gap flags and reference-label audit for evaluation/C |
| `artifacts/data/file_inventory.csv` | Per-file inventory and source hashes |
| `artifacts/data/trip_quality.csv` | Per-trip checks and coverage |
| `artifacts/data/candidate_sensitivity.csv` | Coverage at different K/radius settings |
| `artifacts/data/summary.json` | Final counts, settings, feature names and miss rates |
| `artifacts/data/eda.png` | Sampling, trajectory-size, geographic and network-size plots |
| `artifacts/data/trip_map.html` | Interactive first-training-trip map |
| `data_quality_notes.md` | Short required summary; full explanations are here |

The older `data_processing/` scripts and their two maps were removed because
this pipeline supersedes them and their defaults differed. Model code in
`models/` remains B's work. Supplied raw data and source guidelines stay local
and are excluded from Git.

## Setup and reproduction

From the repository root, use a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-data.txt
.\.venv\Scripts\python.exe data_pipeline.py --data-root "Student folder/Student folder"
.\.venv\Scripts\python.exe -m unittest test_data_pipeline -v
```

Expected input layout:

```text
Student folder/Student folder/
  train/<trip_id>/<trip_id>.track, .nodes, .arcs, .route
  student_test/<trip_id>/<trip_id>.track, .nodes, .arcs, .route
```

Defaults are `--k 32 --radius 300 --output artifacts/data`. To experiment without
overwriting the checked-in handoff, choose another output directory. The build
writes all outputs after processing the trips; allow memory for millions of rows.
The unit tests validate the canonical `artifacts/data` output and its default
settings. Open the HTML in a browser; preview the PNG directly.

## Feature-table schema

Each row is one observation/candidate arc pair. Identifiers are `trip_id`
(string preserving leading zeros), `obs_idx` (zero-based), and `arc_id`
(zero-based, local to the trip network). `split` is `train` or `student`.
`label` is 1 when the candidate matches the inferred reference arc, otherwise 0.
Only the following nine columns are model inputs:

| Feature | Meaning |
| --- | --- |
| `distance_m` | GPS-to-segment distance in metres |
| `offset_m` | Projected distance from directed arc start |
| `offset_fraction` | Offset divided by arc length, bounded to 0-1; 0 for zero-length arcs |
| `heading_difference_rad` | Absolute wrapped GPS/arc heading difference, 0-pi; -1 when undefined |
| `speed_mps` | Previous GPS displacement divided by actual elapsed seconds; -1 at first observation |
| `arc_length_m` | Projected endpoint-to-endpoint arc length |
| `distance_rank` | One-based distance rank, with arc ID breaking ties |
| `from_degree` | Total incoming/outgoing incidence at start node |
| `to_degree` | Total incoming/outgoing incidence at end node |

This is manual feature engineering, not automated feature selection. No feature
importance, ablation study or model-based selection was performed in Part A.
Those experiments belong with model evaluation and must use training-only,
trip-grouped cross-validation.

## Handoff to B, C and D

B can load the table directly with the existing harness (install B's model
dependencies separately):

```python
import pandas as pd
from models.harness import run_all

df = pd.read_parquet("artifacts/data/features.parquet")
run_all(df)
```

For C, observation keys are `partition`, `trip_id`, `obs_id`; `partition` uses
`student_test` rather than the feature table's `student`. Match `obs_id` to
`obs_idx` and normalize the split name when joining. Use `timestamp_s`, `dt_s`
and `large_gap` to handle temporal breaks. Audit fields `true_arc_id`,
`route_occurrence`, `label_residual_m` and `candidate_hit` are reference metadata,
not prediction features. `true_arc_id` is an inferred label despite its name.

D can construct and cache `Network(nodes, arcs)` and call
`candidate_features(network, track)` for inference without loading `.route`.
Both nodes and tracks use longitude first. Candidate generation returns
`obs_id` and an observation table; attach trip IDs and normalize schema as needed.
Do not pass reference metadata to the prediction API/UI. No models, HMM,
FastAPI endpoints or UI are implemented by Part A.


## Reproduction and scope

Run `python data_pipeline.py` and `python -m unittest test_data_pipeline -v`.
The pipeline reads only `train` and `student_test`. No evaluator partition is
included. Raw files are not committed. `file_inventory.csv` records exact bytes,
SHA-256, parsed dtype, rows and columns for every source file. `trip_quality.csv`
contains extents, timestamps, graph connectivity, label residuals and coverage.
Nonfinite values, invalid IDs/coordinates, malformed columns and non-increasing
timestamps cause an explicit failure instead of silent deletion or reordering.

## Coordinates and graph

Input order is longitude, latitude, timestamp for tracks; longitude, latitude for
nodes. The first training trip is in Germany, but the full collection spans
multiple continents and both hemispheres. Some longitudes exceed 90 degrees,
which also rules out treating the first column as latitude. Folium uses latitude first.
Metric distances use each network's median-coordinate UTM zone with
`always_xy=True`. Arcs are straight directed segments between indexed nodes;
the dataset provides no intermediate polyline vertices. Distances are therefore
to that representation, not necessarily to the full physical road geometry.
Node degree is total incoming plus outgoing incidence, including parallel arcs.
Both weak and strong connected component counts include isolated nodes.
Network equality uses exact node and arc file hashes, not matching row counts.

## Per-observation reference labels: inferred, not directly supplied

`.route` contains a chronological sequence of zero-based arc IDs, not one label
per GPS point. It supplies no entry/exit timestamps. We project each GPS point
onto every route occurrence and use dynamic programming to minimize `(distance_m / 20)^2`, subject to nondecreasing route-occurrence index.
Repeated arc IDs retain distinct occurrences, which matters for loops.

A very weak tie-breaking prior adds `0.05 * (route_progress - elapsed_fraction)^2`
per observation. Route progress uses cumulative segment length plus projected
offset; elapsed fraction uses the actual timestamp span. This assumes elapsed
time is weakly informative about progress, not that travel speed is constant.
The dominant cost is spatial distance. Starts and ends are not forced onto route
endpoints. The algorithm permits skipped route arcs between observations and
does not impose a maximum speed; projected offset can move backward within the
same arc due to GPS jitter. Large gaps (>10 seconds) are flagged, not interpolated.

These are **route-derived proxy labels**, not independently measured per-point
ground truth. Ambiguity remains at junctions, parallel streets, long gaps and
route revisits. `label_residual_m` and `route_occurrence` are audit metadata only.
No rows are removed merely because alignment residual is high. Classifier scores
and candidate miss rates measure agreement with this labeling assumption.

## Candidates and features

The default is at most 32 nearest directed arcs within 300 metres. A Shapely
STRtree queries exact point-to-segment distance, including long arcs whose
midpoints/endpoints are far away. Sort by distance then arc ID for reproducible
ties. There is no forced insertion of a reference arc or nearest-outside-radius
fallback. Candidate construction has no access to route data.

The nine model features are `distance_m`, `offset_m`, `offset_fraction`,
`heading_difference_rad`, `speed_mps`, `arc_length_m`, `distance_rank`,
`from_degree`, `to_degree`. Offset is measured from the directed arc's start;
rank is one-based. Heading is the wrapped absolute difference (0 to pi) between
the previous GPS displacement and directed arc heading. Speed uses actual time
difference, not assumed 1 Hz. The first point's speed and undefined heading
(first point, displacement <=1 metre, or zero-length arc) use -1, outside their physical ranges.
No learned imputation or scaling is fitted globally.

`features.parquet` matches B's harness schema. **Only its nine feature columns
should enter a classifier.** `observations.parquet` preserves every GPS point,
including points with no candidates; the reference arc, candidate hit, route
occurrence and residual must never become input features. The split and trip IDs
are metadata. Train/CV groups must be whole trips.

Miss rate is the number of observations without their inferred reference arc in
the candidate set divided by all observations, including zero-candidate points.
`candidate_sensitivity.csv` compares K/radius configurations without inserting
labels into candidates. Parameters should be chosen from training coverage;
student coverage is reported separately. B's current `obs_metrics` groups only
candidate rows, so it omits zero-candidate observations: use the observation
table for the complete denominator whenever any such points exist.

## Map and EDA

The Folium map shows the first training trip's raw GPS, reference route and road
network with a legend and layer control. Only the displayed road layer is clipped
to the trip bounding box plus 0.005 degrees; feature queries use the full network.
The HTML needs internet access for basemap tiles and Leaflet assets. The EDA PNG
shows observations per trip, sampling gaps, GPS extent and graph sizes.

## Assistance

Implementation and validation were assisted by Codex. Numerical findings below
come from the actual supplied files and executed pipeline; labels remain subject
to the assumptions above.

## Measured source-data findings

- 80 training trips (208,417 points) and 10 student trips (19,160 points):
  227,577 observations total. Every trip has the four expected files, with the
  verified 3/2/2/1 column formats. No nonfinite or out-of-range coordinates or
  non-increasing timestamps were found.
- 219,934 of 227,487 timestamp steps are exactly one second (96.68%). There are
  617 gaps over 10 seconds across 42 trips. The largest is 153,064 seconds in
  training trip `00000090`; trip `00000078` has a 52,810-second gap. Sequence
  decoding should explicitly break/restart across long gaps.
- GPS bounds are longitude -100.418538 to 137.237029 and latitude -34.640291 to
  61.699590. A single fixed UTM zone would be inappropriate for this collection.
- All 90 node/arc file pairs have distinct hashes. Summed across per-trip
  networks there are 2,024,174 nodes and 4,061,945 directed arcs (these sums are
  not a deduplicated global map). There are no isolated nodes, 106 zero-length
  arcs, and 1 to 421 weak components per network.
- There are 14 disconnected consecutive reference-arc pairs across 11 routes.
  Labels retain chronological order across these breaks; they do not pretend
  the supplied reference is a fully connected path.
- The initial K=16, radius=100m build missed 22,319/208,417 training observations
  (10.71%) and 22/19,160 student observations (0.115%). This motivated widening
  the candidate search to K=32, radius=300m, using training coverage.
- Direct nearest-reference checks (without monotonic alignment) already put
  719 points in trip `00000002`, 2,978 in `00000033`, and 16,157 in `00000054`
  more than 100m from every reference arc. Their nearest-route median distances
  are 5.67m, 122.41m and 138.59m respectively. Poor coverage here cannot be
  attributed only to an implementation's route-order constraint. Possible
  causes include GPS displacement and reference limitations; the data alone
  does not establish the cause. Do not treat inferred labels as unquestionable.

## Final handoff

| Partition | Observations | Misses | Miss rate | Zero-candidate points |
| --- | ---: | ---: | ---: | ---: |
| train | 208,417 | 8,361 | 4.0117% | 0 |
| student | 19,160 | 0 | 0% | 0 |

K=32/radius=300m yields 6,172,468 candidate rows. K=16/radius=300m still misses
6.40% of training observations, so both radius and K matter. Remaining misses
are retained rather than repaired with ground truth. This is a documented
coverage ceiling for B/C, not evidence of a trained model's accuracy. Consider
label-residual sensitivity analyses using training data before trusting model
comparisons. The present table has no zero-candidate points, so B's current
candidate-row denominator agrees with the full observation denominator here.

Four automated tests passed: long-segment spatial lookup and directed heading,
zero-candidate preservation, chronological route alignment, and full-table
uniqueness/labels/counts/radius/partition invariants. A sampled logistic fit/predict
checks integration with B's existing harness; it is not a reported experiment.
Feature floats are float32 and Parquet uses Zstandard compression. Source
coordinates and the separate observation audit retain float64 precision.
