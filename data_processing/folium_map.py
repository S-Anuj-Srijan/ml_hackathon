"""Interactive Folium map for one trip: raw GPS, nearby road network, reference route, optional predicted route.
Usage: python folium_map.py "<trip folder>" [out.html]   (predicted route: call make_map(..., predicted_arcs=[...]))"""
import sys, numpy as np, folium, shapely
from shapely import STRtree
from pyproj import Transformer
from load_trip import load_trip

_TO_LL = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)


def _ll(xy):  # metric -> [(lat, lon)] for folium
    lon, lat = _TO_LL.transform(xy[:, 0], xy[:, 1]); return list(zip(lat, lon))


def make_map(trip, out="trip_map.html", net_radius=150.0, predicted_arcs=None, title=""):
    xy, N, arcs, r = trip["xy"], trip["node_xy"], trip["arcs"], trip["route"]
    A, B = N[arcs[:, 0]], N[arcs[:, 1]]; lines = shapely.linestrings(np.stack([A, B], axis=1))
    _, li = STRtree(lines).query(shapely.points(xy), predicate="dwithin", distance=net_radius); near = np.unique(li)  # only arcs near the trip
    c = _ll(xy[len(xy) // 2:len(xy) // 2 + 1])[0]; m = folium.Map(location=c, zoom_start=15, tiles="OpenStreetMap", control_scale=True)
    g_net = folium.FeatureGroup(f"Road network (arcs within {net_radius:.0f} m of GPS: {len(near)})", show=True)
    for a in near: folium.PolyLine(_ll(np.array([A[a], B[a]])), color="#888", weight=2, opacity=0.6, tooltip=f"arc {a}").add_to(g_net)
    g_ref = folium.FeatureGroup("Reference route", show=True)
    for a in r: folium.PolyLine(_ll(np.array([A[a], B[a]])), color="#1a9850", weight=5, opacity=0.9, tooltip=f"ref arc {a}").add_to(g_ref)
    g_gps = folium.FeatureGroup("Raw GPS", show=True); ll = _ll(xy)
    for i, p in enumerate(ll): folium.CircleMarker(p, radius=2.5, color="#2166ac", fill=True, fill_opacity=0.8, weight=0, tooltip=f"obs {i}, t={trip['t'][i]:.0f}s").add_to(g_gps)
    for g in (g_net, g_ref, g_gps): g.add_to(m)
    legend = ('<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:8px 12px;border:1px solid #999;font-size:13px">'
              f'<b>{title or "Trip"}</b><br><span style="color:#2166ac">&#9679;</span> Raw GPS<br><span style="color:#1a9850">&#9644;</span> Reference route<br>'
              '<span style="color:#888">&#9644;</span> Road network<br>')
    if predicted_arcs is not None:
        g_pred = folium.FeatureGroup("Predicted route", show=True)
        for a in predicted_arcs: folium.PolyLine(_ll(np.array([A[a], B[a]])), color="#d73027", weight=4, dash_array="8", tooltip=f"pred arc {a}").add_to(g_pred)
        g_pred.add_to(m); legend += '<span style="color:#d73027">&#9644;</span> Predicted route<br>'
    m.get_root().html.add_child(folium.Element(legend + "</div>")); folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([min(ll), max(ll)]); m.save(out); return out


if __name__ == "__main__":
    d = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else "trip_map.html"
    print(make_map(load_trip(d), out, title=d.rstrip("\\/").split("\\")[-1].split("/")[-1]))
