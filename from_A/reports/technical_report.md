# PathSnap Technical Report

Distributed GPS map matching | Complete A-D implementation and measured student-development evaluation | 18 November 2026

This report is generated from repository artifacts. It reports actual experiments, not target scores. The available dataset contains 80 training trips and 10 student-testing trips; the reserved evaluator partition has not been used. Codex assisted implementation and documentation.

# 2.2 Dataset understanding and exploratory analysis

The raw formats are longitude/latitude/time tracks, longitude/latitude nodes, directed endpoint-index arcs, and chronological reference arc sequences. The pipeline validates dimensions, finite values, coordinate ranges, integer IDs and strictly increasing timestamps; it fails instead of silently cleaning malformed data. All 360 source files are inventoried with SHA-256, size, shape and dtype.

| Partition | Trips | Observations | Candidate miss rate |
| --- | --- | --- | --- |
| student_test | 10 | 19160 | 0.0000% |
| train | 80 | 208417 | 4.0117% |

EDA found 617 gaps above 10 seconds across 42 trips, with a largest gap of 153,064 seconds. There were no nonfinite/out-of-range coordinates or non-increasing timestamps. Networks span both hemispheres, so each uses its median-coordinate UTM zone. There are 90 distinct network hashes, 106 zero-length arcs, and 14 disconnected consecutive reference pairs across 11 routes. These are source-data findings from train and student partitions, not evaluator analysis.

Reference routes do not contain per-observation labels. A monotone dynamic program aligns each GPS point to route occurrences using squared spatial distance and a weak elapsed-progress tie breaker. Labels are proxies with ambiguous junctions, loops and displaced observations. High residual observations remain in the data. There is no ground-truth insertion into candidates.

Nine manual features describe distance, projected offset and offset fraction, heading difference, measured speed, arc length, distance rank, and endpoint degree. Distances and headings provide local geometric agreement; speed and graph degree describe motion and junction ambiguity. Undefined speed/heading use -1. The classifier input is an explicit allowlist; IDs, labels and route-derived audit fields are excluded. No automated feature-selection claim is made for this release.

![EDA from the supplied training and student partitions.](../artifacts/data/eda.png)

# 3. Model experimentation and comparison

All five models were fitted on all 80 training trips and evaluated with 3-fold GroupKFold. Scaling is fitted inside each training fold. Model choice maximizes mean CV observation Top-1 accuracy, matching the road-selection objective. Student data is evaluated in full and does not choose hyperparameters. Seed is 42. Candidate F1, precision and recall are macro averages at a fixed 0.5 threshold; log loss measures binary candidate probability quality.

| Model | CV F1 | Student Top-1 | Student F1 | Log loss | Continuity |
| --- | --- | --- | --- | --- | --- |
| LogReg | 0.7334 | 0.7989 | 0.8053 | 0.0609 | 0.8927 |
| GaussianNB | 0.6955 | 0.5761 | 0.6567 | 0.2583 | 0.8655 |
| DecisionTree | 0.8598 | 0.8138 | 0.8990 | 0.0385 | 0.8986 |
| ExtraTrees | 0.8632 | 0.8096 | 0.8947 | 0.0413 | 0.8955 |
| HistGB | 0.8604 | 0.7834 | 0.8876 | 0.0592 | 0.8442 |

| Model | CV Top-1 | Fit seconds | Inference ms / 1k candidates |
| --- | --- | --- | --- |
| LogReg | 0.6708 | 10.35 | 0.189 |
| GaussianNB | 0.5110 | 1.30 | 0.560 |
| DecisionTree | 0.7005 | 161.89 | 0.187 |
| ExtraTrees | 0.7008 | 93.83 | 3.021 |
| HistGB | 0.7015 | 62.87 | 5.213 |

Timing values are local wall-clock measurements, not isolated hardware benchmarks. The HistGB comparison ran alongside another model to reduce total turnaround; resource contention can affect timing. Model selection uses CV quality rather than these timing differences.

Student candidate macro F1 is highest for DecisionTree, while student observation Top-1 is highest for DecisionTree. These metrics ask different questions: thresholded classification across mostly negative candidate rows versus choosing one road per observation. High candidate accuracy can be dominated by negatives. The chosen primary metric is grouped-CV Top-1; log loss and F1 remain supporting diagnostics.

## LogReg

Family: Linear. A scaled linear baseline tests whether geometric cues combine additively. Fast and interpretable, but cannot directly represent complex feature interactions. Objective: Logistic loss with L2 regularization; LBFGS optimization.

Initial configuration: Pipeline(steps=[('standardscaler', StandardScaler()),
                ('logisticregression',
                 LogisticRegression(max_iter=300, random_state=42))])

Important initial values: C=1, L2 penalty, LBFGS, max_iter=300; StandardScaler in each fold.

Student results: Top-1 0.7989, macro F1 0.8053, log loss 0.0609, macro precision 0.9277, macro recall 0.7398, candidate accuracy 0.9780. CV F1 0.7334 +/- 0.0744.

The student-minus-CV F1 gap is +0.0719. This compares different trip populations and label quality, so it cannot by itself establish overfitting or underfitting. Fold dispersion and student performance provide generalization evidence; no artificial training curves are produced.

## GaussianNB

Family: Probabilistic. A low-cost probability baseline. Correlated distance, offset and rank violate its independence assumption and can make confidence poorly calibrated. Objective: Gaussian class-conditional likelihood with conditional feature independence.

Initial configuration: GaussianNB()

Important initial values: var_smoothing=1e-9; class priors estimated from each training fold.

Student results: Top-1 0.5761, macro F1 0.6567, log loss 0.2583, macro precision 0.6668, macro recall 0.6479, candidate accuracy 0.9531. CV F1 0.6955 +/- 0.0923.

The student-minus-CV F1 gap is -0.0388. This compares different trip populations and label quality, so it cannot by itself establish overfitting or underfitting. Fold dispersion and student performance provide generalization evidence; no artificial training curves are produced.

## DecisionTree

Family: Tree. Captures feature thresholds and interactions. A single tree is unstable and produces coarse probability estimates; depth and leaf-size constraints limit variance. Objective: Greedy Gini impurity minimization.

Initial configuration: DecisionTreeClassifier(max_depth=12, min_samples_leaf=30, random_state=42)

Important initial values: max_depth=12, min_samples_leaf=30, Gini criterion.

Student results: Top-1 0.8138, macro F1 0.8990, log loss 0.0385, macro precision 0.8998, macro recall 0.8983, candidate accuracy 0.9854. CV F1 0.8598 +/- 0.0285.

The student-minus-CV F1 gap is +0.0392. This compares different trip populations and label quality, so it cannot by itself establish overfitting or underfitting. Fold dispersion and student performance provide generalization evidence; no artificial training curves are produced.

## ExtraTrees

Family: Ensemble. Averages nonlinear trees to reduce variance. More memory and latency than linear methods; random thresholds may miss precise geometric boundaries. Objective: Randomized split ensembles using Gini impurity.

Initial configuration: ExtraTreesClassifier(max_depth=16, min_samples_leaf=20, n_estimators=40,
                     n_jobs=4, random_state=42)

Important initial values: 40 trees, max_depth=16, min_samples_leaf=20, max_features=1.0, four workers.

Student results: Top-1 0.8096, macro F1 0.8947, log loss 0.0413, macro precision 0.9147, macro recall 0.8767, candidate accuracy 0.9855. CV F1 0.8632 +/- 0.0217.

The student-minus-CV F1 gap is +0.0315. This compares different trip populations and label quality, so it cannot by itself establish overfitting or underfitting. Fold dispersion and student performance provide generalization evidence; no artificial training curves are produced.

## HistGB

Family: Boosting. Learns nonlinear interactions with bounded trees and efficient histogram bins. Sensitive to complexity and probability calibration; random row-level early stopping is disabled to avoid within-trip validation. Objective: Binary log loss optimized by histogram gradient boosting.

Initial configuration: HistGradientBoostingClassifier(early_stopping=False, max_leaf_nodes=15,
                               random_state=42)

Important initial values: 100 iterations, learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=20, L2=0, early_stopping=False.

Student results: Top-1 0.7834, macro F1 0.8876, log loss 0.0592, macro precision 0.8873, macro recall 0.8878, candidate accuracy 0.9837. CV F1 0.8604 +/- 0.0287.

The student-minus-CV F1 gap is +0.0272. This compares different trip populations and label quality, so it cannot by itself establish overfitting or underfitting. Fold dispersion and student performance provide generalization evidence; no artificial training curves are produced.

![Actual model comparison across all five algorithms.](../artifacts/experiments/plots/comparison.png)

![Candidate-level student confusion matrices, class order 0 and 1.](../artifacts/experiments/plots/confusion.png)

# 4. Bayesian hyperparameter optimisation

The baseline CV leader exceeds the runner-up by only 0.076 percentage points. This is a small observed advantage, not a statistical claim of definitive superiority. The predeclared selection rule identifies one family for the finite tuning budget.

Chosen model: HistGB. The exact model-specific search space is encoded in mlops/train.py: LogReg C [0.01,10] log; GaussianNB smoothing [1e-12,1e-5] log; tree/ExtraTrees depth [8,22] and minimum leaf samples [5,100] log; HistGB learning rate [0.03,0.2] log, leaf nodes [8,40], L2 [1e-4,10] log, iterations [60,160]. Only the selected family is searched.

Optuna TPE completed 4 trials, including three startup trials. Objective: mean training-only group-CV Top-1. Further optimization was stopped to prioritize the working demonstration. One interrupted trial was marked failed and excluded from selection. Best parameters: {"learning_rate": 0.061053157936408836, "max_leaf_nodes": 39, "l2_regularization": 0.4570563099801455, "max_iter": 120}. Tuned model deployed: True.

| Metric | Untuned | Tuned |
| --- | --- | --- |
| cv_macro_f1_mean | 0.86039 | 0.86683 |
| cv_top1_mean | 0.70155 | 0.70614 |
| student_top1 | 0.78335 | 0.81273 |
| student_macro_f1 | 0.88756 | 0.89801 |
| student_log_loss | 0.05917 | 0.03611 |
| student_inference_ms_per_1k_cands | 5.21268 | 8.09932 |

Selection and promotion follow CV, not the student score. The tuned model is retained only when CV Top-1 is at least the untuned score. Final refitting uses all training trips. The limited completed search is not evidence of a global optimum; no further accuracy tuning is needed to run this demonstration.

# 5. HMM and Viterbi reconstruction

Hidden states are directed candidate arcs. Emissions are clipped positive-class classifier probabilities normalized across candidates per observation. Transition weights are exp(-abs(network distance - GPS displacement)/40 m), normalized over reachable next candidates. Network distance includes the remaining previous arc, bounded directed Dijkstra distance between endpoints, and next-arc offset. Same-arc movement permits 5 m backward jitter. The shortest-path bound is 2,000 m.

Viterbi uses score[t,j] = log emission[t,j] + max_i(score[t-1,i] + log transition[i,j]), then backtracks the maximizing predecessors. Missing candidates output -1 and break the chain. Gaps above 10 seconds or fully unreachable transitions restart the chain and are explicitly reported. These parameters are fixed engineering defaults; no decoder tuning is claimed.

| Metric | Classifier | Classifier + HMM |
| --- | --- | --- |
| top1 | 0.81273 | 0.84285 |
| continuity | 0.90136 | 0.96031 |
| route_jaccard | 0.67341 | 0.76810 |
| exact_sequence | 0.00000 | 0.00000 |

HMM changes pointwise agreement by +0.03011. Improvement is not guaranteed: transition assumptions may disagree with proxy labels at parallel roads, sharp turns, long gaps or reference-route discontinuities. Per-trip evidence is in sequence_metrics.csv. Continuity is strict endpoint adjacency, not general graph reachability. Route Jaccard ignores order; exact sequence match uses the entire proxy arc sequence. Pointwise means are observation-weighted; continuity is pair-weighted; exact-sequence match is trip-weighted.

| Trip | Classifier Top-1 | HMM Top-1 | Breaks |
| --- | --- | --- | --- |
| 00000000 | 0.9241 | 0.9489 | 0 |
| 00000007 | 0.9018 | 0.8892 | 0 |
| 00000009 | 0.8170 | 0.8561 | 0 |
| 00000040 | 0.8661 | 0.9109 | 0 |
| 00000052 | 0.9313 | 0.9885 | 0 |
| 00000056 | 0.8605 | 0.9106 | 0 |
| 00000065 | 0.7956 | 0.8438 | 3 |
| 00000076 | 0.7840 | 0.8553 | 24 |
| 00000077 | 0.6249 | 0.6188 | 0 |
| 00000091 | 0.8204 | 0.8490 | 0 |

# 6. Geographical visualisation

artifacts/experiments/route_map.html is the interactive Folium comparison, with raw GPS, road network, reference and predicted segment layers plus layer controls. The static figure below is generated from the same actual coordinates and predictions. It is an exported geometry view; interactive basemap tiles need internet access.

![Student route: blue GPS, grey network, green reference, red HMM-selected segments.](../artifacts/experiments/route_map.png)

# 7. API, UI, deployment and MLOps

Laptop 1 hosts the backend API, model and graph. Laptop 2 opens its independent frontend/index.html and connects to Laptop 1 via the configured LAN API URL. The portable ZIP includes local map assets; no model or reference data is bundled. The UI requests /trips and POST /predict/{trip_id}; only prediction geometry, input GPS and aggregate scores are returned. Prediction and reference scoring are separate. The API reports proxy-label point accuracy and explicitly named route Jaccard. No dummy fallback remains.

The release validates model SHA-256 and feature schema before startup, fails on invalid trips or incorrect trip count, and exports data/source fingerprints, dependency versions and seed. /health, /ready and /metrics support operations. Structured JSON prediction logs record request ID, model version, latency and score. Docker mounts data/references read-only; CI runs synthetic tests without raw data. README documents installation, reproduction, environment configuration and rollback.

Automated API verification passed for 10 real student trips, with one predicted arc per observation and no reference arrays in responses. Scope: ten student-development trips via FastAPI TestClient; no physical two-laptop or Docker execution. GitHub publication, reserved evaluator scores and a physical two-laptop LAN test are not established by these checks.

| Trip | HTTP | Points | API seconds |
| --- | --- | --- | --- |
| 00000000 | 200 | 2503 | 7.20 |
| 00000007 | 200 | 1354 | 3.87 |
| 00000009 | 200 | 1049 | 8.23 |
| 00000040 | 200 | 2995 | 12.50 |
| 00000052 | 200 | 262 | 4.11 |
| 00000056 | 200 | 3958 | 29.69 |
| 00000065 | 200 | 768 | 7.04 |
| 00000076 | 200 | 1486 | 24.36 |
| 00000077 | 200 | 3911 | 25.35 |
| 00000091 | 200 | 874 | 6.91 |

# Reproducibility and remaining demonstration checks

Commands: python data_pipeline.py; python -m mlops.train; python -m mlops.evaluate; python -m unittest discover -s tests -v; python -m mlops.verify_release; python -m mlops.report. The README includes the complete file tree and operational runbook. All reported values can be traced to compact JSON/CSV artifacts; the model manifest identifies source hashes and dataset hash. No epoch-wise curves are claimed for the five selected algorithms.

Known limits: proxy labels, incomplete candidate coverage in training, a small student partition, fixed decoder defaults, bounded road-distance searches, and simplified straight arc geometry. This release predicts the observed segments; it does not insert unobserved connecting arcs into output geometry. No independently verified physical ground truth is available. Run the physical LAN/container checks and evaluator setup on the actual assessment machines, and publish the repository before submission.
