# PathSnap – GPS Map-Matching Pipeline

> Predict which road arcs a vehicle traversed from noisy GPS traces.

---

## 👥 Team Breakdown & Roles

| Member | Owns | Deliverable |
|:------:|:-----|:------------|
| **A** | Data loading, EDA, Folium maps, candidate generation, features | `features.parquet` to B and C |
| **B** | Model zoo, CV harness, comparison plots, Optuna | Saved model and probabilities to C and D |
| **C** | Road-graph connectivity, HMM/Viterbi, route continuity metric | Decoder module to D |
| **D** | Repo structure, FastAPI, UI, two-laptop setup, README | Working end-to-end skeleton |

---

## 🚀 Quick Start (FastAPI & UI)

### 1. Install dependencies

```bash
# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# Install all requirements
pip install -r requirements.txt
```

### 2. Run the API & UI server (Laptop 1)

```bash
# Starts on 0.0.0.0:8000 (accessible locally & on LAN)
python run_server.py
```

Console displays:
- Local UI: `http://localhost:8000/ui`
- API Docs: `http://localhost:8000/docs`
- **Laptop 2 LAN URL**: `http://<laptop1-ip>:8000/ui`

### 3. Open the UI (Laptop 2 or Laptop 1)

Open any browser and navigate to:
```
http://localhost:8000/ui        # On Laptop 1
http://<laptop1-ip>:8000/ui     # On Laptop 2
```

- **Select any of the 10 trips** in the sidebar.
- See the raw GPS track (pink) and the predicted route (teal) rendered interactively on Leaflet.
- View accuracy score, predicted arc count, and inference latency.
- Click **"🚀 Predict All Trips"** to batch-run all 10 test trips.

---

## 🌐 Two-Laptop Setup Instructions

Evaluators will view request logs on **Laptop 1** while operating the UI on **Laptop 2**.

### Laptop 1 (Server Host)
1. Ensure both laptops are connected to the same Wi-Fi / local network.
2. Allow incoming traffic on port 8000 through the firewall (Windows):
   ```cmd
   netsh advfirewall firewall add rule name="PathSnap API" dir=in action=allow protocol=TCP localport=8000
   ```
3. Run `python run_server.py`. The console will print the exact IP address to use on Laptop 2.

### Laptop 2 (Client UI)
1. Open a browser and visit: `http://<laptop1-ip>:8000/ui`
2. The UI automatically resolves the API host from the page origin.
3. Test requests and observe live structured logs streaming on Laptop 1's terminal.

---

## 🔒 Server-Side Ground-Truth Isolation

- Ground-truth `.route` files live in `ground_truth/` and are strictly read **only** by `api/scoring.py`.
- Ground truth **never reaches the model or the client/UI response payload**.
- Only the computed numerical accuracy score (Jaccard similarity) is returned to the UI.
- Nothing is hardcoded; all scores are computed dynamically from data files.

---

## ⚡ Performance & Startup Architecture

- **Load Once at Startup**: Road network graph (`adjacency list`) and spatial index (`scipy.spatial.cKDTree` in projected UTM 32N coordinates) are constructed once during server startup, never per request.
- **Sub-millisecond inference**: Nearest-arc queries execute in < 2ms without recomputing midpoints or graph distances.
- **Structured logging**: Every incoming request is assigned a UUID and logged with method, trip ID, accuracy, arc counts, and latency in milliseconds.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| `GET` | `/trips` | Returns list and metadata for all 10 student-test trips |
| `POST` | `/predict/{trip_id}` | Runs ML + Viterbi pipeline, returns predicted route & accuracy |
| `GET` | `/health` | Health check and predictor status |
| `GET` | `/ui` | Interactive Leaflet map frontend |
| `GET` | `/docs` | Interactive Swagger API documentation |

---

## 📊 Part A: Data & Candidate Features (Member A)

```bash
python -m pip install -r requirements-data.txt
python data_pipeline.py --data-root "Student folder/Student folder"
python -m unittest test_data_pipeline -v
```

Outputs in `artifacts/data/`:
- `features.parquet` (one row per observation-candidate pair)
- `observations.parquet`
- File inventory and per-trip quality checks
- `eda.png` and `trip_map.html`

---

## 📁 Repository Structure

```
ml_hackathon/
├── api/                            # FastAPI Backend & UI (Member D)
│   ├── main.py                     # API app, endpoints, lifespan, logging
│   ├── data_loader.py              # TripMeta, UTM 32N projection, cKDTree index
│   ├── predictor.py                # DummyPredictor & integration factory
│   ├── scoring.py                  # Isolated ground-truth evaluation
│   └── static/
│       └── index.html              # Interactive Leaflet.js map UI
├── artifacts/data/                 # Generated feature tables & EDA plots (Member A)
├── data_processing/                # Candidate generation & feature engineering
│   ├── build_features.py
│   ├── candidates.py
│   ├── eda.py
│   └── folium_map.py
├── ground_truth/                   # Server-only evaluator ground truth (.route files)
├── models/                         # ML Model zoo, Optuna tuning, evaluation (Member B)
│   ├── harness.py
│   ├── run_quick.py
│   ├── feature_selection.py
│   └── tune.py
├── run_server.py                   # Server startup script with LAN auto-discovery
├── requirements.txt                # Unified project dependencies
├── data_pipeline.py                # End-to-end data pipeline script
└── README.md
```
