# PathSnap backend — Laptop 1

This folder owns the API, input loading, model inference and server-side scoring.
Keep the repository on Laptop 1. Send **only `frontend/` (or its ZIP)** to Laptop 2.
The model, reference routes, datasets and Python environment stay here.

## Start on the prepared Windows laptop

From the repository root:

```powershell
.\backend\start_backend.ps1
```

The launcher uses `.venv` if installed, or the existing local `.runtime` with
Anaconda. You can override its interpreter with `PATHSNAP_PYTHON`.
If your PowerShell policy prevents running scripts, use the direct command:

```powershell
$env:PYTHONPATH='.runtime'
& "$env:USERPROFILE\anaconda3\python.exe" backend/run_server.py
```

For a new backend machine, install Python 3.13 and dependencies once:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe backend/run_server.py
```

`python backend/run_server.py` and `cd backend; python run_server.py` both work
with the installed environment. The root `run_server.py` remains a compatibility
launcher. No training or tuning is needed to demonstrate the exported model.

## Connect Laptop 2

1. Keep the backend terminal open and confirm `/ready` reports `ready` and ten trips.
2. Choose the printed API address belonging to your Wi-Fi adapter, for example
   `http://192.168.1.20:8000`. Use `ipconfig` to distinguish VPN/virtual adapters.
3. On Laptop 2, extract the frontend ZIP, open `index.html`, enter that address and
   click **Connect**. A trip click starts prediction.
4. Both laptops must use the same LAN/hotspot. If needed, allow Python/TCP 8000 on
   the trusted network in Laptop 1's firewall. Do not disable the firewall.

The default CORS setting `*` supports a local file (`Origin: null`) and an optional
frontend web server. There are no browser credentials. To restrict origins when
using HTTP hosting, set `PATHSNAP_CORS_ORIGINS` to the frontend origins, e.g.
`http://localhost:8080,http://127.0.0.1:8080`, then restart. The API is intended for
the trusted presentation LAN; it has no authentication or TLS.

## Server resources

Paths are resolved from the repository root, independently of the launch directory:

| Resource | Default | Override |
|---|---|---|
| Prediction inputs | `Student folder/Student folder/student_test/` | `PATHSNAP_DATA_DIR` |
| Server-only references | `ground_truth/` | `PATHSNAP_TRUTH_DIR` |
| Trained weights + matching JSON | `artifacts/model/best_model.joblib` | `PATHSNAP_MODEL` |
| Required trip count | `10` | `PATHSNAP_EXPECTED_TRIPS` |

Shared ML code remains at the repository root: `data_pipeline.py`, `decoder.py`,
`models/` and `mlops/`. Training artifacts and input data are backend resources;
they are intentionally excluded from the frontend ZIP. This avoids moving raw data
or changing the validated training artifacts just to deploy the UI separately.

The API runs even if `frontend/` is absent. When that folder is present, `/ui/`
provides an optional preview using exactly the same frontend files. The Docker
image contains the backend only; run `docker compose up --build` from the root.
Docker and a physical two-laptop network require rehearsal on the target machines.

Use `/health`, `/ready`, `/metrics` and `/docs` for operations. Model checksum,
feature-schema and dependency validation still happen before readiness. Reference
routes are used after prediction for aggregate scoring and are not sent to the UI.

## Model improvement in the terminal

Each backend startup prints the five-model comparison, including CV Top-1,
student Top-1, student macro F1 and the student Top-1 difference from Logistic
Regression. A second section compares untuned versus tuned performance, followed
by classifier versus HMM Top-1 and continuity. Changes are **percentage points**;
negative differences are shown honestly.

These are saved development results from `artifacts/experiments/`, not new
training or scores for the current request. The serving model version is printed
separately; after replacing a model, regenerate its evaluation records before
presenting the comparisons as its results. Missing or invalid comparison files
produce a warning without stopping inference.

Restart an already-running server to see the new startup summary. To preview it
without starting another server or retraining, run from the repository root:

```powershell
python -m backend.model_comparison
```
