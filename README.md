# PathSnap: GPS map matching and MLOps

PathSnap turns noisy GPS observations into road-segment predictions using a trained classifier and a directed-network HMM/Viterbi decoder. The FastAPI service runs on Laptop 1; Laptop 2 opens its browser UI over the LAN. Reference routes stay on the server and are used only after inference for scoring.

## Deliverables A, B, C and D

| Part | Delivered implementation | Evidence |
|---|---|---|
| A: data and features | File validation, inventory, per-network UTM, route-derived labels, STRtree candidate lookup, nine features, EDA | `A_PHASE1_README.md`, `artifacts/data/` |
| B: models | Five algorithm comparisons, trip-grouped CV, Bayesian TPE search, trained artifact and manifest | `mlops/train.py`, `artifacts/experiments/`, `artifacts/model/` |
| C: sequence reconstruction | Directed shortest-road distances, normalized emissions/transitions, log-space Viterbi, explicit gap/restart policy | `decoder.py`, `mlops/evaluate.py`, sequence metrics and route maps |
| D: API, UI and MLOps | Verified model loading, prediction/scoring separation, health/readiness, request logs, counters, Docker, CI, tests | `api/`, `tests/`, `Dockerfile`, `compose.yaml`, `.github/workflows/ci.yml` |

The consolidated report is generated as `reports/technical_report.md` and `output/pdf/PathSnap_Technical_Report.pdf`. Measured results are in `artifacts/experiments/`; do not substitute synthetic unit-test scores for them.

## File structure

```text
ml_hackathon/
  README.md                     Setup, structure, architecture, operations
  PathSnap_README.md             Supplied assignment requirements
  PathSnap_README_assets/        Images extracted from the supplied guidelines
  A_PHASE1_README.md             Detailed Part A methodology and evidence
  data_quality_notes.md          Data limitations and label assumptions
  data_pipeline.py              Canonical preprocessing and feature extraction
  test_data_pipeline.py         Geometry, alignment and full artifact tests
  decoder.py                    Directed-network HMM and Viterbi
  models/                       Original exploration scripts and shared harness
    harness.py                  Metrics and trip-grouped validation
    run_quick.py                Historical subset experiment entry point
    feature_selection.py        Training-only feature diagnostics
    compare_features.py         Feature-subset comparison
    tune.py                     Original HistGB tuning entry point
    export_model.py             Original HistGB export entry point
    plots.py                    Model result visualizations
  mlops/
    train.py                    Canonical complete training and export command
    compare_one.py              Optional independent comparison checkpoint worker
    evaluate.py                 Ten-trip sequence evaluation and maps
    report.py                   Measured-results Markdown/PDF report generator
    verify_release.py           Release integrity and API smoke checks
  api/
    main.py                     Lifecycle, endpoints, server-side scoring, logs
    data_loader.py              Validated GPS/network inputs without labels
    predictor.py                Checksum-verified model, shared features, decoder
    scoring.py                  Server-only route loading and set overlap
    static/index.html           Leaflet evaluator UI
  tests/test_system.py           Decoder, artifact and API isolation tests
  artifacts/
    data/                       Inventory, audit, parquet tables, EDA, source map
    experiments/                Model/fold/trial results, plots, sequence maps
    model/                      best_model.joblib and best_model.json manifest
  reports/                      Technical report source and validation evidence
  output/pdf/                   Consolidated submission PDF
  ground_truth/                 Server-only student reference routes
  Student folder/Student folder/  Local raw train/ and student_test/ (ignored)
  requirements-data.txt         Data pipeline dependencies
  requirements.txt              Full application/training/report dependencies
  requirements-lock.txt         Exact validated dependency versions
  Dockerfile / compose.yaml     Container deployment with read-only data mounts
  .github/workflows/ci.yml       Synthetic quality gates without private raw data
  run_server.py                 LAN server launcher
```

`models/` retains earlier team experiments for traceability. Use `python -m mlops.train` for the deployment artifact; the older `export_model.py` bundle is not the release format consumed by the new API.

## Installation

Use Python 3.13 from the repository root. A clean virtual environment avoids mixing existing Anaconda or machine packages.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
```

For development against version ranges, use `requirements.txt` instead. On Linux/macOS activate with `source .venv/bin/activate`. The local `.runtime` directory, if present, is an ignored execution environment created during validation, not a deployment dependency.

## Reproduce the complete pipeline

Raw data must have this layout (trip IDs preserve leading zeros):

```text
Student folder/Student folder/
  train/<trip_id>/<trip_id>.track .nodes .arcs .route
  student_test/<trip_id>/<trip_id>.track .nodes .arcs .route
```

```powershell
python data_pipeline.py --data-root "Student folder/Student folder"
python -m unittest test_data_pipeline -v
python -m mlops.train --folds 3 --trials 8
python -m mlops.evaluate
python -m unittest discover -s tests -v
python -m mlops.verify_release
python -m mlops.report
```

Training uses all 80 training trips and three GroupKFold folds. No trip crosses a fold. Five models are evaluated on all ten student trips; model selection uses mean CV observation Top-1 accuracy. TPE searches the selected model's hyperparameters on training folds only. This release stopped after four completed trials (three startup and one adaptive); a fifth interrupted trial was marked failed. Further tuning was stopped to prioritize the demonstration. The command above describes the original eight-trial reproduction budget; do not run it for the presentation. Untuned and tuned models use identical splits and metrics. If tuning worsens CV Top-1, the untuned model is retained. All fitted preprocessing stays inside model pipelines. HistGB's random row-level early stopping is disabled.

Training checkpoints each completed model and retains Optuna trials in local SQLite. Resuming validates data, source and environment fingerprints. Use a new `--output` directory when changing code/data/configuration; do not mix old results. The selected artifact always exports to `artifacts/model/`. Run evaluation and reporting after a successful export. Full training needs substantial memory for 6.17 million candidate rows; it is not run in CI.

## Features and leakage boundaries

The exact allowlist is `distance_m`, `offset_m`, `offset_fraction`, `heading_difference_rad`, `speed_mps`, `arc_length_m`, `distance_rank`, `from_degree`, `to_degree`. Training and inference call the same geometry functions. Candidate generation keeps at most 32 closest directed segments within 300 metres, without inserting the reference arc. Projection uses the network's local UTM zone, not a single global zone.

`.route` is a chronological arc sequence, not a label per GPS point. Monotone spatial alignment creates proxy point labels. Reported point accuracy therefore means agreement with this assumption, not independent positioning truth. The audit retains large gaps, residuals, and missing candidates. Training candidate misses are 4.0117%; student misses are zero in the supplied artifact. Evaluator data is neither required nor read by training/evaluation commands.

## HMM behavior and metrics

States are candidate directed arcs. Emission scores are clipped classifier probabilities normalized within each observation. Transitions compare directed shortest-network travel distance with measured GPS displacement, with exponential scale 40 m and a 2,000 m search bound. A 5 m same-arc backward tolerance handles jitter. Transition rows are normalized over reachable candidates. Viterbi maximizes summed log probability and backtracks each contiguous segment.

Gaps above 10 seconds restart decoding; unreachable transitions restart with an explicit break; absent candidates produce `-1`. These choices are fixed engineering defaults, not claimed optimized parameters. No line is fabricated between disconnected selected road segments. Output contains one arc ID per observation. Geometry contains actual selected segments, not interpolated full-road paths between observations.

Candidate metrics: accuracy, macro precision/recall/F1, log loss and confusion matrix. Observation Top-1 includes candidate misses. The current dataset has no zero-candidate observations; the original harness groups candidate rows and must be extended with a full observation denominator before evaluating a dataset that does. Sequence evaluation uses every observation, including missing predictions. Route continuity is the share of adjacent observation pairs staying on the same arc or sharing directed endpoints; valid multi-edge traversals between sparse observations may fail this strict metric. Route Jaccard compares arc sets and ignores ordering; exact sequence match is reported separately.

## Run the two-laptop demonstration

**Use the exported model for the presentation. Do not retrain or tune during the demo.** Only Laptop 1 needs Python, the project, model and data. Laptop 2 needs a browser. Connect both to the same Wi-Fi or phone hotspot; avoid guest networks that isolate devices.

On this prepared development laptop, the validated local runtime can be used directly:

```powershell
$env:PYTHONPATH='.runtime'
$env:PYTHONIOENCODING='utf-8'
& 'C:\Users\text2\anaconda3\python.exe' run_server.py --host 0.0.0.0 --port 8000
```

On another machine, complete the clean virtual-environment installation above first, then use the standard command below.

Laptop 1:

```powershell
python run_server.py --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/ui/` locally. On Laptop 2, connected to the same LAN, open `http://<LAPTOP_1_IP>:8000/ui/`. The UI defaults to the page's API origin. Permit inbound TCP 8000 on the trusted LAN if needed. Keep the receiving terminal visible: each prediction logs request ID, trip ID, model version, duration, point count and aggregate score. The UI uses CDN Leaflet assets, which need internet access. Basemap tiles are optional and off by default so tile-provider failures do not hide predictions. API inference itself does not need internet. The generated Folium comparison uses road geometry without remote tiles.

Alternatively run `docker compose up --build`. Data and truth are read-only volume mounts; they are not copied into the image. The container health check probes readiness. Docker and a physical second laptop must be tested on the actual demonstration machines.

### Five-minute presentation script

1. **Problem (30 seconds):** “GPS is noisy. PathSnap assigns observations to directed road segments and reconstructs a consistent sequence.” Show the architecture and A–D table in this README.
2. **Laptop 1: server (45 seconds):** Start the server and open `/ready`. Explain that startup validates the model checksum, feature schema and ten input trips. Keep the terminal visible to show incoming requests.
3. **Laptop 2: live prediction (90 seconds):** Open the LAN URL printed by Laptop 1. Select a trip and press Predict. Show the GPS and predicted road layers, zoom into a junction, and point out processing time and matched observations. Wait for the request to finish before switching trips.
4. **Explain A–C (60 seconds):** “A validates data and builds nine geometric features; B compares five classifiers with trip-separated validation; C uses directed-network Viterbi decoding.” Explain that displayed accuracy is agreement with route-derived proxy labels. Do not promise perfect positioning.
5. **Explain D/MLOps (45 seconds):** Show the corresponding server request log, model version, `/metrics`, tests and release validation. Explain reproducible dependencies, artifact fingerprints, CI checks and rollback using a matching model/manifest pair.
6. **Close (30 seconds):** Show the report and comparison map. State that the demonstrated flow is browser → API → shared feature extraction → classifier → decoder → map, with reference scoring performed separately on the server.

### Rehearsal and fallback

- Before presenting, open `/ready` and run one prediction from **Laptop 2**. Test a second trip. Keep both laptops awake and plugged in.
- On Laptop 2, use Laptop 1's Wi-Fi IPv4 address, never `localhost` or `0.0.0.0`. Find it with `ipconfig` if needed; ignore VPN/virtual-adapter addresses.
- If the remote page fails but local `/ready` works, check the IP, common Wi-Fi and trusted-network firewall permission for Python/TCP 8000. A phone hotspot is a useful fallback for client-isolated campus Wi-Fi.
- Leaflet assets require internet on the browser; load the UI before presenting. Basemap tiles are optional. Keep the PDF and `artifacts/experiments/route_map.png` available as a static backup.
- If connectivity fails during the talk, demonstrate locally on Laptop 1 and explicitly describe the network issue. Do not describe a local fallback as a successful two-machine test.

## Evaluation configuration

Defaults serve the ten **student-development** trips, not the ten reserved evaluator trips. For final evaluation, configure the supplied evaluator input and separate server-only references at runtime:

```powershell
$env:PATHSNAP_DATA_DIR='C:\evaluation\inputs'
$env:PATHSNAP_TRUTH_DIR='C:\evaluation\references'
$env:PATHSNAP_EXPECTED_TRIPS='10'
$env:PATHSNAP_MODEL='artifacts/model/best_model.joblib'
python run_server.py
```

Inputs contain track/nodes/arcs files; references contain `<trip_id>.route`. The API never exposes reference arrays or loads them in the predictor. If a reference is absent, inference works and accuracy is omitted. Startup fails for invalid input, wrong trip count, missing/corrupted model, or incompatible feature schema. There is no dummy fallback.

## API and MLOps operations

| Endpoint | Purpose |
|---|---|
| `GET /health` | Process liveness |
| `GET /ready` | Loaded trip count and model version; 503 before readiness |
| `GET /trips` | Input metadata only |
| `POST /predict/{trip_id}` | Real classifier/HMM result and aggregate server-side scoring |
| `GET /metrics` | Per-process request/error counts and total prediction seconds |
| `GET /docs` | OpenAPI interactive documentation |
| `GET /ui/` | Evaluator UI |

The model manifest records SHA-256, dataset/source hashes, versions, seed, trip IDs and selection policy. Only load artifacts you trust: joblib is Python serialization. Before promotion, run tests, all-ten-trip verification, and inspect the measured comparisons. Keep the previous model **and its matching JSON manifest** to roll back; set `PATHSNAP_MODEL` to that pair and restart. A checksum detects accidental corruption; it is not an authenticity signature.

CI checks synthetic decoder edge cases, label isolation, checksum rejection and API behavior without requiring private data. Full local release checks cover actual artifacts and all student trips. Logs and `/metrics` are lightweight per-process observability, not a durable monitoring platform. Use one worker for the hackathon deployment; every worker loads its own models and networks. CORS is configurable with `PATHSNAP_CORS_ORIGINS` (comma-separated origins). This LAN demo has no authentication or TLS; add these before public hosting.

## Submission and honest limitations

See the generated technical report and `reports/release_validation.json` for measured evidence. The raw data is local and ignored. Commit source, compact experiment tables, report, plots and `artifacts/model/` weights/manifest together. Large transient probability arrays and the tuning database are ignored but reproducible from training. The existing GitHub repository was visibly public during validation. The new local changes still need to be committed and pushed before claiming submission. Reserved evaluator results and a physical two-laptop run cannot be claimed from student-data validation. Codex assisted implementation; the team must understand and explain each design and experiment.
