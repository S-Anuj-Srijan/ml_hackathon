// ── State ────────────────────────────────────────────────────────
    let trips = [];
    let predictions = Object.create(null);   // trip_id → prediction result
    let activeTrip = null;
    let apiGeneration = 0;
    let inflight = 0;
    let connectedApiUrl = '';
    let toastTimer;
    const STORAGE_KEY = 'pathsnap.backendUrl';

    // Map layers
    let map, trackLayer, routeLayer, markersLayer;

    // ── Init ─────────────────────────────────────────────────────────
    document.addEventListener('DOMContentLoaded', () => {
        const apiUrlInput = document.getElementById('api-url');
        initMap();
        const configured = window.PATHSNAP_CONFIG?.apiBaseUrl || '';
        const mounted = /^https?:$/.test(window.location.protocol) && /^\/ui(?:\/|$)/.test(window.location.pathname);
        let saved = '';
        try { saved = window.localStorage.getItem(STORAGE_KEY) || ''; } catch (_) { /* file:// storage may be unavailable */ }
        apiUrlInput.value = configured || (mounted ? window.location.origin : saved);
        document.getElementById('connection-form').addEventListener('submit', event => {
            event.preventDefault();
            loadTrips();
        });
        document.getElementById('btn-predict-all').addEventListener('click', predictAll);
        if (apiUrlInput.value) loadTrips();
        else showTripMessage("Enter Laptop 1's backend URL above, then click Connect. Select a trip to draw its prediction.");
    });

    function normalizeApiUrl(value) {
        let url;
        try { url = new URL(value.trim()); } catch (_) { throw new Error('Enter a full URL, for example http://192.168.1.20:8000'); }
        if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash || !/^\/*$/.test(url.pathname)) {
            throw new Error('Use only an http:// or https:// server address and port, with no path, credentials, or query.');
        }
        return url.origin;
    }

    async function requestJson(url, options = {}, timeoutMs = 30000) {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), timeoutMs);
        try {
            const response = await fetch(url, {...options, signal: controller.signal, credentials: 'omit'});
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } finally { clearTimeout(timer); }
    }

    function showTripMessage(message, isError = false) {
        const element = document.createElement('p');
        element.className = 'trip-message' + (isError ? ' error' : '');
        element.textContent = message;
        document.getElementById('trip-list').replaceChildren(element);
    }

    // ── Map setup ────────────────────────────────────────────────────
    function initMap() {
        map = L.map('map', {
            zoomControl: false,
            attributionControl: false,
        }).setView([50, 10], 5);

        // Geometry works without remote tiles; an optional basemap is available.
        const basemap = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            maxZoom: 19, attribution: '&copy; OpenStreetMap contributors'
        });
        L.control.layers({}, {'Online basemap (optional)': basemap}, {position: 'topright'}).addTo(map);

        // Apply dark filter to map tiles
        document.querySelector('.leaflet-tile-pane').style.filter =
            'invert(1) hue-rotate(180deg) brightness(0.8) contrast(1.2) saturate(0.3)';

        L.control.zoom({ position: 'bottomleft' }).addTo(map);
        L.control.attribution({ position: 'bottomleft' }).addTo(map)
            .addAttribution('© <a href="https://osm.org/">OpenStreetMap</a>');

        trackLayer   = L.layerGroup().addTo(map);
        routeLayer   = L.layerGroup().addTo(map);
        markersLayer = L.layerGroup().addTo(map);
    }

    // ── Load trips ───────────────────────────────────────────────────
    async function loadTrips() {
        let url;
        try { url = normalizeApiUrl(document.getElementById('api-url').value); }
        catch (error) { toast(error.message, true); return; }
        document.getElementById('api-url').value = url;
        const generation = ++apiGeneration;
        connectedApiUrl = '';
        predictions = Object.create(null);
        trips = [];
        activeTrip = null;
        inflight = 0;
        showSpinner(false);
        document.getElementById('btn-predict-all').disabled = true;
        document.getElementById('btn-predict-all').textContent = '🚀 Predict All Trips';
        document.getElementById('btn-connect').textContent = 'Connecting…';
        document.getElementById('info-overlay').classList.remove('show');
        document.getElementById('trip-count').textContent = 'Loading trips';
        if (trackLayer) trackLayer.clearLayers();
        if (routeLayer) routeLayer.clearLayers();
        if (markersLayer) markersLayer.clearLayers();
        setStatus('loading', 'Connecting…');
        showTripMessage(`Connecting to ${url}…`);

        try {
            const data = await requestJson(`${url}/trips`);
            if (generation !== apiGeneration) return;
            if (!Array.isArray(data.trips) || data.trips.some(trip => !trip || !['string', 'number'].includes(typeof trip.trip_id))) {
                throw new Error('The server returned an invalid trip list.');
            }
            trips = data.trips.map(trip => ({...trip, trip_id: String(trip.trip_id)}));
            connectedApiUrl = url;
            try { window.localStorage.setItem(STORAGE_KEY, url); } catch (_) { /* Saving is optional. */ }
            renderTripList();
            setStatus('ok', 'Connected');
            document.getElementById('trip-count').textContent = `${trips.length} trips`;
            document.getElementById('btn-predict-all').disabled = trips.length === 0;
            toast(`Loaded ${trips.length} trips`);
        } catch (e) {
            if (generation !== apiGeneration) return;
            setStatus('err', 'Disconnected');
            document.getElementById('trip-count').textContent = '0 trips';
            showTripMessage(`Cannot reach the backend at ${url}. Check Laptop 1's server, its IP address, firewall and shared network. If your browser blocks a local file connection, use start_frontend.bat (Python required). ${e.name === 'AbortError' ? 'Connection timed out.' : e.message}`, true);
            toast('Connection failed – check API URL', true);
        } finally {
            if (generation === apiGeneration) document.getElementById('btn-connect').textContent = 'Connect';
        }
    }

    // ── Render trip list ─────────────────────────────────────────────
    function renderTripList() {
        const list = document.getElementById('trip-list');
        list.replaceChildren();
        if (!trips.length) { showTripMessage('Connected, but this backend has no trips loaded.'); return; }

        trips.forEach(trip => {
            const card = document.createElement('div');
            card.className = 'trip-card';
            card.dataset.tripId = trip.trip_id;
            card.tabIndex = 0;
            card.setAttribute('role', 'button');
            const badge = document.createElement('span');
            badge.className = 'accuracy-badge';
            const title = document.createElement('div');
            title.className = 'trip-id';
            title.textContent = `Trip #${trip.trip_id}`;
            const meta = document.createElement('div');
            meta.className = 'trip-meta';
            for (const value of [`📍 ${trip.n_points} pts`, `🔗 ${trip.n_arcs} arcs`, `⬡ ${trip.n_nodes} nodes`]) {
                const span = document.createElement('span');
                span.textContent = value;
                meta.appendChild(span);
            }
            card.append(badge, title, meta);

            card.addEventListener('click', () => selectTrip(trip.trip_id));
            card.addEventListener('keydown', event => {
                if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); selectTrip(trip.trip_id); }
            });
            list.appendChild(card);
        });
    }

    // ── Select a trip ────────────────────────────────────────────────
    async function selectTrip(tripId) {
        // Update active card
        document.querySelectorAll('.trip-card').forEach(c => c.classList.toggle('active', c.dataset.tripId === tripId));

        activeTrip = tripId;

        // If we already have a prediction, show it
        if (predictions[tripId]) {
            showPrediction(tripId, predictions[tripId]);
            return;
        }

        // Otherwise, run prediction
        await predictTrip(tripId);
    }

    // ── Predict a single trip ────────────────────────────────────────
    async function predictTrip(tripId) {
        const url = connectedApiUrl;
        if (!url) return false;
        const generation = apiGeneration;
        inflight++;
        showSpinner(true);

        try {
            const data = await requestJson(`${url}/predict/${encodeURIComponent(tripId)}`, {method: 'POST'}, 300000);
            if (generation !== apiGeneration) return false;
            predictions[tripId] = data;
            if (activeTrip === tripId) showPrediction(tripId, data);
            updateBadge(tripId, data.accuracy);
            toast(`Trip #${tripId} – prediction ready`);
            return true;
        } catch (e) {
            if (generation === apiGeneration) toast(`Prediction failed for trip #${tripId}: ${e.name === 'AbortError' ? 'request timed out' : e.message}`, true);
            return false;
        } finally {
            if (generation === apiGeneration) {
                inflight--;
                showSpinner(inflight > 0);
            }
        }
    }

    // ── Predict all trips ────────────────────────────────────────────
    async function predictAll() {
        const btn = document.getElementById('btn-predict-all');
        btn.disabled = true;
        btn.textContent = '⏳ Predicting…';

        const generation = apiGeneration;
        let succeeded = 0;
        for (const trip of trips) {
            if (generation !== apiGeneration) return;
            activeTrip = trip.trip_id;
            document.querySelectorAll('.trip-card').forEach(c => c.classList.toggle('active', c.dataset.tripId === trip.trip_id));
            if (await predictTrip(trip.trip_id)) succeeded++;
            // Brief pause for visual feedback
            await new Promise(r => setTimeout(r, 200));
        }

        if (generation !== apiGeneration) return;
        btn.textContent = `${succeeded}/${trips.length} Predicted`;
        setTimeout(() => {
            if (generation !== apiGeneration) return;
            btn.textContent = '🚀 Predict All Trips';
            btn.disabled = false;
        }, 2000);

        // Show summary
        const accs = Object.values(predictions)
            .filter(p => p.accuracy !== null && p.accuracy !== undefined)
            .map(p => p.accuracy);
        if (accs.length > 0) {
            const avg = (accs.reduce((a, b) => a + b, 0) / accs.length * 100).toFixed(1);
            toast(`${succeeded}/${trips.length} succeeded. Mean trip accuracy: ${avg}%`);
        } else toast(`${succeeded}/${trips.length} predictions succeeded`);
    }

    // ── Show prediction on map ───────────────────────────────────────
    function showPrediction(tripId, data) {
        // Clear layers
        trackLayer.clearLayers();
        routeLayer.clearLayers();
        markersLayer.clearLayers();

        const track = data.track;    // [[lon, lat], ...]
        const route = data.route_geometry;  // [{from: [lon,lat], to: [lon,lat]}, ...]

        if (!track || track.length === 0) return;

        // Draw GPS track (pink dots + line)
        const trackLatLngs = track.map(p => [p[1], p[0]]);

        L.polyline(trackLatLngs, {
            color: '#fd79a8',
            weight: 3,
            opacity: 0.7,
            dashArray: '8,6',
        }).addTo(trackLayer);

        // GPS point dots (subtle)
        trackLatLngs.forEach((ll, i) => {
            if (i % Math.max(1, Math.floor(track.length / 50)) === 0) {
                L.circleMarker(ll, {
                    radius: 2.5,
                    fillColor: '#fd79a8',
                    fillOpacity: 0.6,
                    color: 'transparent',
                }).addTo(trackLayer);
            }
        });

        // Draw predicted route (teal lines)
        if (route && route.length > 0) {
            route.forEach(seg => {
                L.polyline(
                    [[seg.from[1], seg.from[0]], [seg.to[1], seg.to[0]]],
                    { color: '#00cec9', weight: 4, opacity: 0.9 }
                ).addTo(routeLayer);
            });
        }

        // Start & end markers
        const startIcon = L.divIcon({
            html: '<div style="width:14px;height:14px;background:#55efc4;border-radius:50%;border:2px solid #fff;box-shadow:0 0 8px rgba(85,239,196,0.6);"></div>',
            iconSize: [14, 14],
            iconAnchor: [7, 7],
            className: '',
        });

        const endIcon = L.divIcon({
            html: '<div style="width:14px;height:14px;background:#d63031;border-radius:50%;border:2px solid #fff;box-shadow:0 0 8px rgba(214,48,49,0.6);"></div>',
            iconSize: [14, 14],
            iconAnchor: [7, 7],
            className: '',
        });

        L.marker(trackLatLngs[0], { icon: startIcon }).addTo(markersLayer)
            .bindPopup('Start');
        L.marker(trackLatLngs[trackLatLngs.length - 1], { icon: endIcon }).addTo(markersLayer)
            .bindPopup('End');

        // Fit map to bounds
        const allPoints = [...trackLatLngs];
        if (route) {
            route.forEach(seg => {
                allPoints.push([seg.from[1], seg.from[0]]);
                allPoints.push([seg.to[1], seg.to[0]]);
            });
        }
        map.fitBounds(L.latLngBounds(allPoints).pad(0.1));

        // Update info overlay
        const overlay = document.getElementById('info-overlay');
        overlay.classList.add('show');
        document.getElementById('info-trip-id').textContent = tripId;
        document.getElementById('info-points').textContent = track.length;
        document.getElementById('info-arcs').textContent = data.n_predicted_arcs;
        document.getElementById('info-accuracy').textContent =
            data.accuracy !== null && data.accuracy !== undefined
                ? `${(data.accuracy * 100).toFixed(1)}%`
                : 'N/A';
        document.getElementById('info-time').textContent = `${data.prediction_time_ms}ms`;
    }

    // ── Badge ────────────────────────────────────────────────────────
    function updateBadge(tripId, accuracy) {
        const card = [...document.querySelectorAll('.trip-card')].find(element => element.dataset.tripId === tripId);
        const badge = card?.querySelector('.accuracy-badge');
        if (!badge || accuracy === null || accuracy === undefined) return;

        const pct = (accuracy * 100).toFixed(0);
        badge.textContent = `${pct}%`;
        badge.classList.add('show');
        badge.classList.remove('high', 'mid', 'low');
        if (accuracy >= 0.5) badge.classList.add('high');
        else if (accuracy >= 0.25) badge.classList.add('mid');
        else badge.classList.add('low');
    }

    // ── Helpers ──────────────────────────────────────────────────────
    function setStatus(type, text) {
        const el = document.getElementById('status');
        const cls = type === 'ok' ? 'ok' : type === 'err' ? 'err' : 'loading';
        const dot = document.createElement('span');
        dot.className = `dot ${cls}`;
        el.replaceChildren(dot, document.createTextNode(` ${text}`));
    }

    function showSpinner(show) {
        document.getElementById('spinner').classList.toggle('show', show);
    }

    function toast(msg, isError = false) {
        const el = document.getElementById('toast');
        el.textContent = msg;
        el.className = 'toast show' + (isError ? ' error' : '');
        clearTimeout(toastTimer);
        toastTimer = setTimeout(() => el.classList.remove('show'), isError ? 7000 : 3000);
    }
