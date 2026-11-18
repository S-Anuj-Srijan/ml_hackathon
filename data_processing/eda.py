"""Dataset inventory + EDA over all trips. Usage: python eda.py "<...>/Student folder"
Writes results/eda_inventory.csv and results/eda_plots/*.png. Uses TRAIN + STUDENT-TEST trips only (never evaluator data)."""
import sys, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from load_trip import load_trip


def summarize(d, split):
    tr = load_trip(d); xy, t, arcs, N, r = tr["xy"], tr["t"], tr["arcs"], tr["node_xy"], tr["route"]
    dt = np.diff(t); n_nodes = len(N)
    g = coo_matrix((np.ones(len(arcs)), (arcs[:, 0], arcs[:, 1])), shape=(n_nodes, n_nodes))
    ncomp, comp = connected_components(g, directed=False)
    used = np.zeros(n_nodes, bool); used[arcs.ravel()] = True
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    rl = np.linalg.norm(N[arcs[r, 1]] - N[arcs[r, 0]], axis=1).sum()
    return dict(trip=os.path.basename(d), split=split, n_points=len(xy),
                n_nan_or_inf=int((~np.isfinite(tr["xy"])).any(1).sum() + (~np.isfinite(t)).sum()),
                t_non_monotonic=int((dt <= 0).sum()), pct_1s_steps=float((dt == 1).mean() * 100), max_gap_s=float(dt.max()),
                n_gaps_gt_5s=int((dt > 5).sum()), duration_s=float(t[-1] - t[0]),
                track_len_m=float(seg.sum()), max_jump_m=float(seg.max()), bbox_w_m=float(np.ptp(xy[:, 0])), bbox_h_m=float(np.ptp(xy[:, 1])),
                n_nodes=n_nodes, n_arcs=len(arcs), isolated_nodes=int((~used).sum()), n_components=int(ncomp),
                largest_comp_pct=float(np.bincount(comp).max() / n_nodes * 100),
                n_route_arcs=len(r), route_len_m=float(rl), route_arc_ids_valid=bool(((r >= 0) & (r < len(arcs))).all()),
                files_bytes=int(sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d))))


if __name__ == "__main__":
    root = sys.argv[1]; rows = []
    for folder, split in (("train", "train"), ("student_test", "student")):
        for trip in sorted(os.listdir(os.path.join(root, folder))):
            try: rows.append(summarize(os.path.join(root, folder, trip), split)); print(rows[-1]["trip"], rows[-1]["n_points"], flush=True)
            except Exception as e: print("FAILED", trip, repr(e)); rows.append(dict(trip=trip, split=split, error=repr(e)))
    os.makedirs("../results/eda_plots", exist_ok=True); df = pd.DataFrame(rows); df.to_csv("../results/eda_inventory.csv", index=False)
    print(df.groupby("split").size().to_string()); print(df.describe().T[["min", "50%", "max"]].round(1).to_string())
    ok = df[df.get("error").isna()] if "error" in df else df
    for col, title in (("n_points", "GPS points per trip"), ("max_gap_s", "Largest time gap per trip (s)"),
                       ("n_nodes", "Road-network nodes per trip"), ("n_components", "Connected components per trip"),
                       ("max_jump_m", "Largest jump between consecutive fixes (m)")):
        fig, ax = plt.subplots(figsize=(6, 3.5)); ax.hist(ok[col], bins=25); ax.set_title(title + " (train+student)"); ax.set_xlabel(col)
        fig.tight_layout(); fig.savefig(f"../results/eda_plots/{col}.png", dpi=200); plt.close(fig)
