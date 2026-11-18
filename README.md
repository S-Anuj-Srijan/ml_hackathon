# PathSnap: GPS map matching and MLOps

PathSnap turns noisy GPS observations into road-segment predictions using a trained classifier and a directed-network HMM/Viterbi decoder. The **backend runs on Laptop 1**. The **portable frontend runs on Laptop 2** and calls Laptop 1 over the LAN. Reference routes stay on the server and are used only after inference for scoring.

Read the single [consolidated project report](REPORT.md) for A?D, measured results, contributor credits and the complete MLOps runbook.

## Send the frontend to another laptop

Build the transfer archive with `python scripts/package_frontend.py`, then send **`dist/PathSnap-frontend.zip`**. ZIPs are generated locally and excluded from Git. It contains only the `frontend/` folder: HTML, CSS, JavaScript, local map assets and instructions. No model, dataset, reference routes or Python ML dependencies are included.

1. **Laptop 1:** Keep the full repository here and run `.\backend\start_backend.ps1` from the repository root. Leave the terminal open and note its Wi-Fi API URL, e.g. `http://192.168.1.20:8000`.
2. **Laptop 2:** Extract the ZIP completely, then double-click `frontend/index.html`. Enter Laptop 1's URL in **Backend URL** and click **Connect**.
3. Click a trip to predict and draw its GPS/road route. Both laptops must be on the same Wi-Fi or hotspot.

Laptop 2 needs only a browser. If its browser blocks requests from a local file, the included optional `start_frontend.bat` serves the folder at `http://localhost:8080` using Python 3. Core map rendering uses bundled assets and needs no internet; only the optional basemap does. Full instructions: [backend README](backend/README.md) and [frontend README](frontend/README.md).

Rebuild the ZIP after editing the frontend with `python scripts/package_frontend.py`. Its file list and SHA-256 checksums are recorded in `dist/frontend-manifest.json`.

Verified locally: 11 backend/deployment tests, frontend JavaScript/configuration checks, all 16 ZIP files extracted and served independently, and a real prediction over HTTP with the separate frontend origin. Evidence: [package validation](reports/frontend_package_validation.json) and [HTTP validation](reports/split_http_validation.json). A physical second-laptop connection still needs a rehearsal on your Wi-Fi.

## Deliverables A, B, C and D

| Part | Delivered implementation | Evidence |
|---|---|---|
| A: data and features | File validation, inventory, per-network UTM, route-derived labels, STRtree candidate lookup, nine features, EDA | `A_PHASE1_README.md`, `artifacts/data/` |
| B: models | Five algorithm comparisons, trip-grouped CV, Bayesian TPE search, trained artifact and manifest | `mlops/train.py`, `artifacts/experiments/`, `artifacts/model/` |
| C: sequence reconstruction | Directed shortest-road distances, normalized emissions/transitions, log-space Viterbi, explicit gap/restart policy | `decoder.py`, `mlops/evaluate.py`, sequence metrics and route maps |
| D: API, UI and MLOps | Separate portable UI and API, verified model loading, prediction/scoring separation, health/readiness, logs, Docker, CI, tests | `frontend/`, `backend/`, `tests/`, `Dockerfile`, `compose.yaml`, `.github/workflows/ci.yml` |

The single submission report is **[REPORT.md](REPORT.md)**. `python -m mlops.report` refreshes its measured results while preserving its operations appendix; intermediate Markdown/PDF outputs remain local and ignored. Measured results are in `artifacts/experiments/`; do not substitute synthetic unit-test scores for them.

## File structure

```text
ml_hackathon/
  README.md                     Setup, structure, architecture, operations
  REPORT.md                     Single consolidated A?D and MLOps report
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
  backend/                      Laptop 1 API application
    api/main.py                 Lifecycle, endpoints, server-side scoring, logs
    api/data_loader.py          Validated GPS/network inputs without labels
    api/predictor.py             Checksum-verified model, shared features, decoder
    api/scoring.py               Server-only route loading and set overlap
    paths.py                    Working-directory-independent resource paths
    run_server.py               Backend entry point, prints LAN API URLs
    start_backend.ps1           Windows launcher for prepared environment
    requirements.txt            References the root dependency lock
    README.md                   Laptop 1 setup and configuration
  frontend/                     Copy this entire folder to Laptop 2
    index.html                  Standalone page and controls
    styles.css / app.js          UI styles, API client and route drawing
    config.js                   Optional default backend URL
    vendor/leaflet/              Bundled map library, images and license
    serve.py / start_frontend.bat Optional local HTTP hosting
    README.md                   Laptop 2 connection instructions
  scripts/package_frontend.py    Builds and verifies the portable ZIP
  dist/PathSnap-frontend.zip     Ready-to-send frontend package
  dist/frontend-manifest.json   ZIP/file checksums
  tests/test_system.py           Decoder, artifact and API isolation tests
  tests/test_deployment.py       Resource paths, CORS and independent deployment
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
  run_server.py                 Compatibility wrapper for backend/run_server.py
```

`models/` retains earlier team experiments for traceability. Use `python -m mlops.train` for the deployment artifact; the older `export_model.py` bundle is not the release format consumed by the new API.

## Installation

Install these dependencies on **Laptop 1 only**. Use Python 3.13 from the repository root. A clean virtual environment avoids mixing existing Anaconda or machine packages.

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

**Use the exported model; no retraining or tuning is needed.** Only Laptop 1 needs the repository, data, model and ML environment. Laptop 2 gets the frontend ZIP. Avoid guest networks that isolate clients.

On this prepared Windows backend laptop, run:

```powershell
.\backend\start_backend.ps1
```

Or, using the validated local runtime directly:

```powershell
$env:PYTHONPATH='.runtime'
& "$env:USERPROFILE\anaconda3\python.exe" backend/run_server.py --host 0.0.0.0 --port 8000
```

With a clean environment installed, use `python backend/run_server.py`. The root `python run_server.py` is also supported. Relative resource paths are resolved from the repository root even when launched inside `backend/`.

On Laptop 2, extract `dist/PathSnap-frontend.zip`, open `frontend/index.html`, enter `http://<LAPTOP_1_WIFI_IP>:8000` and click **Connect**. Selecting a trip starts its prediction. Do not enter `/ui/` in the API field, and do not use Laptop 2's `localhost` as the backend address. The frontend remembers successful connections when browser storage is available; `frontend/config.js` can also prefill the backend URL before sharing.

Keep the backend terminal visible: it logs request ID, trip ID, model version, duration, point count and aggregate score. Core inference and frontend assets work without internet; the optional OpenStreetMap basemap needs internet and is off by default. The frontend sends input requests and receives predictions/scores; model weights and references stay on Laptop 1.

If local HTML requests are blocked by the browser, run the optional `frontend/start_frontend.bat` on Laptop 2 (Python 3 required) and open `http://localhost:8080`. Enter the same Laptop 1 API URL there. Both file-origin and local HTTP frontend CORS are supported by the default backend configuration.

For a single-machine preview, open `http://localhost:8000/ui/`. The API also runs without any frontend files. Alternatively, `docker compose up --build` starts a backend-only container with read-only data/truth mounts. Docker and the physical two-laptop connection must be rehearsed on the actual machines.

### Five-minute presentation script

1. **Problem (30 seconds):** “GPS is noisy. PathSnap assigns observations to directed road segments and reconstructs a consistent sequence.” Show the architecture and A–D table in this README.
2. **Laptop 1: server (45 seconds):** Start the server and open `/ready`. Explain that startup validates the model checksum, feature schema and ten input trips. Keep the terminal visible to show incoming requests.
3. **Laptop 2: live prediction (90 seconds):** Open the extracted frontend, enter the LAN API URL printed by Laptop 1, click Connect, then click a trip. Show the GPS and predicted road layers, zoom into a junction, and point out processing time and matched observations. Wait for the request to finish before switching trips.
4. **Explain A–C (60 seconds):** “A validates data and builds nine geometric features; B compares five classifiers with trip-separated validation; C uses directed-network Viterbi decoding.” Explain that displayed accuracy is agreement with route-derived proxy labels. Do not promise perfect positioning.
5. **Explain D/MLOps (45 seconds):** Show the corresponding server request log, model version, `/metrics`, tests and release validation. Explain reproducible dependencies, artifact fingerprints, CI checks and rollback using a matching model/manifest pair.
6. **Close (30 seconds):** Show the report and comparison map. State that the demonstrated flow is browser → API → shared feature extraction → classifier → decoder → map, with reference scoring performed separately on the server.

### Files intentionally excluded from Git

Raw datasets, local Python environments, secrets (`.env`/private keys), logs/caches, generated parquet tables, duplicate source maps, frontend ZIPs and local PDF/report build outputs are ignored. The selected model/manifest, compact experiment evidence, plots and all application source remain tracked. Removing previously tracked generated files from the index does not delete their local copies or erase old Git history. On a fresh clone, supply the raw trip data separately and run the documented pipeline to regenerate training tables.

### Rehearsal and fallback

- Before presenting, open `/ready` and run one prediction from **Laptop 2**. Test a second trip. Keep both laptops awake and plugged in.
- On Laptop 2, use Laptop 1's Wi-Fi IPv4 address, never `localhost` or `0.0.0.0`. Find it with `ipconfig` if needed; ignore VPN/virtual-adapter addresses.
- If the remote page fails but local `/ready` works, check the IP, common Wi-Fi and trusted-network firewall permission for Python/TCP 8000. A phone hotspot is a useful fallback for client-isolated campus Wi-Fi.
- Leaflet assets are bundled in the frontend; core map geometry works without internet. Online basemap tiles are optional. Keep the PDF and `artifacts/experiments/route_map.png` available as a static backup.
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

Backend startup prints the saved five-model comparison, signed student Top-1
differences from Logistic Regression, untuned/tuned results and classifier/HMM
improvements. These are recorded development scores, not a new training run.
Restart the backend to see them, or preview with `python -m backend.model_comparison`.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Process liveness |
| `GET /ready` | Loaded trip count and model version; 503 before readiness |
| `GET /trips` | Input metadata only |
| `POST /predict/{trip_id}` | Real classifier/HMM result and aggregate server-side scoring |
| `GET /metrics` | Per-process request/error counts and total prediction seconds |
| `GET /docs` | OpenAPI interactive documentation |
| `GET /ui/` | Optional preview when the frontend folder exists; not included in backend Docker image |

The model manifest records SHA-256, dataset/source hashes, versions, seed, trip IDs and selection policy. Only load artifacts you trust: joblib is Python serialization. Before promotion, run tests, all-ten-trip verification, and inspect the measured comparisons. Keep the previous model **and its matching JSON manifest** to roll back; set `PATHSNAP_MODEL` to that pair and restart. A checksum detects accidental corruption; it is not an authenticity signature.

CI checks synthetic decoder edge cases, label isolation, checksum rejection and API behavior without requiring private data. Full local release checks cover actual artifacts and all student trips. Logs and `/metrics` are lightweight per-process observability, not a durable monitoring platform. Use one worker for the hackathon deployment; every worker loads its own models and networks. CORS is configurable with `PATHSNAP_CORS_ORIGINS` (comma-separated origins). This LAN demo has no authentication or TLS; add these before public hosting.

## Submission and honest limitations

See the generated technical report and `reports/release_validation.json` for measured evidence. The raw data is local and ignored. Commit source, compact experiment tables, report, plots and `artifacts/model/` weights/manifest together. Large transient probability arrays and the tuning database are ignored but reproducible from training. The existing GitHub repository was visibly public during validation. Use the pushed branch and commit to identify the submitted source version; local validation does not prove hosted CI success. Reserved evaluator results and a physical two-laptop run cannot be claimed from student-data validation. Codex assisted implementation; the team must understand and explain each design and experiment.

<!-- measured-results -->
## Measured release results

| Model | CV Top-1 | Student Top-1 | Student macro F1 | Student log loss |
|---|---:|---:|---:|---:|
| LogReg | 0.6708 | 0.7989 | 0.8053 | 0.0609 |
| GaussianNB | 0.5110 | 0.5761 | 0.6567 | 0.2583 |
| DecisionTree | 0.7005 | 0.8138 | 0.8990 | 0.0385 |
| ExtraTrees | 0.7008 | 0.8096 | 0.8947 | 0.0413 |
| HistGB | 0.7015 | 0.7834 | 0.8876 | 0.0592 |

Selected family: **HistGB**. TPE trials: **4**. Tuned artifact deployed: **True**.

Final classifier Top-1: **81.27%**; classifier + HMM: **84.28%**. Strict continuity changes from **90.14%** to **96.03%**. These are student results against route-derived proxy labels.

All **10 student trips** passed API verification. The data and system/deployment test suites passed (4 + 11 tests). Docker/physical two-laptop deployment and reserved evaluator results remain unverified.

[Consolidated report](REPORT.md) | [Interactive route map](artifacts/experiments/route_map.html) | [Release checks](reports/release_validation.json)
