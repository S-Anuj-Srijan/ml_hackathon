# PathSnap frontend — Laptop 2

Copy this **entire folder**, including `vendor/`, to Laptop 2. It contains no model, dataset, training code or Python dependencies.

1. Start the backend on Laptop 1 and note its LAN address, for example `http://192.168.1.20:8000`.
2. Connect both laptops to the same Wi-Fi or phone hotspot.
3. On Laptop 2, double-click `index.html` to open it in a browser. No installation is needed.
4. Enter Laptop 1's URL in **Backend URL** and click **Connect**. Include `http://` and port `8000`; do not append `/ui` or `/trips`.
5. Click any trip to run its prediction and display the GPS track and predicted road route. **Predict All Trips** runs every loaded trip.

The successful URL is remembered when browser storage is available. You can also prefill `apiBaseUrl` in `config.js` before sending the folder. Using `localhost` on Laptop 2 points to Laptop 2, so enter Laptop 1's actual network IP instead.

## Optional local web server

If your browser blocks API requests from a local HTML file, install/use Python 3 and double-click `start_frontend.bat` (Windows), then open **http://localhost:8080**. On any OS, run `python serve.py` from this folder. This server serves only the frontend on Laptop 2; enter Laptop 1's backend URL in the page as before.

If the browser asks to access devices on your local network, allow that request for this demo. If Connect fails, confirm Laptop 1's `/ready` page opens from Laptop 2, the backend is listening on `0.0.0.0:8000`, and its firewall allows the server on your trusted network. Guest Wi-Fi can isolate devices; use a shared hotspot if necessary. The backend must allow the frontend origin through CORS; its default demo configuration does this. A local-file browser request uses the origin `null`.

## What is included

```text
frontend/
├── index.html               # Page and controls
├── styles.css               # Layout and theme
├── app.js                   # API connection, trips and map drawing
├── config.js                # Optional default backend URL
├── serve.py                 # Optional Python static server, port 8080
├── start_frontend.bat       # Windows launcher for serve.py
├── README.md                # These instructions
└── vendor/leaflet/          # Pinned Leaflet 1.9.4 JS, CSS, images and license
```

All interface and map-rendering assets are local. No internet connection is needed for the GPS and road geometry; the connection to Laptop 1 is required for predictions. The optional **Online basemap** layer uses OpenStreetMap and requires internet. It is off by default.

When this same folder is served by the backend at `/ui/`, it connects to that backend automatically unless a URL is explicitly set in `config.js`.
