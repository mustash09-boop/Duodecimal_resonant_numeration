from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def weighted_distance(x1: float, y1: float, f1: int, x2: float, y2: float, f2: int, time_weight: float) -> float:
    xy = math.hypot(x1 - x2, y1 - y2)
    return xy + abs(int(f1) - int(f2)) * time_weight


def assign_parent_clusters(
    df: pd.DataFrame,
    *,
    max_frame_gap: int,
    max_xy_distance: float,
    time_weight: float,
) -> pd.DataFrame:
    out = df.copy()
    out["parent_cluster_token"] = ""
    out["parent_cluster_freq_hz"] = 0.0
    out["parent_frame_gap"] = -1
    out["parent_xy_distance"] = -1.0
    out["lineage_role"] = "unassigned_dense_peak"

    clusters = out[out["component_type"] == "resonance_cluster"].copy()
    peaks = out[out["component_type"] == "dense_peak"].copy()

    out.loc[clusters.index, "lineage_role"] = "cluster_core"
    out.loc[clusters.index, "parent_cluster_token"] = out.loc[clusters.index, "note_token"].astype(str)
    out.loc[clusters.index, "parent_cluster_freq_hz"] = out.loc[clusters.index, "freq_hz"].astype(float)
    out.loc[clusters.index, "parent_frame_gap"] = 0
    out.loc[clusters.index, "parent_xy_distance"] = 0.0

    if len(clusters) == 0 or len(peaks) == 0:
        return out

    cluster_rows = []
    for idx, r in clusters.iterrows():
        cluster_rows.append(
            {
                "idx": idx,
                "frame_index": int(r.get("frame_index", 0)),
                "x12": float(r.get("x12", 0.0)),
                "y12": float(r.get("y12", 0.0)),
                "note_token": str(r.get("note_token", "")),
                "freq_hz": float(r.get("freq_hz", 0.0)),
            }
        )

    for idx, r in peaks.iterrows():
        frame = int(r.get("frame_index", 0))
        x = float(r.get("x12", 0.0))
        y = float(r.get("y12", 0.0))

        best = None
        for c in cluster_rows:
            frame_gap = abs(frame - c["frame_index"])
            if frame_gap > max_frame_gap:
                continue
            xy_distance = math.hypot(x - c["x12"], y - c["y12"])
            if xy_distance > max_xy_distance:
                continue
            score = weighted_distance(x, y, frame, c["x12"], c["y12"], c["frame_index"], time_weight)
            if best is None or score < best["score"]:
                best = {
                    "cluster": c,
                    "frame_gap": frame_gap,
                    "xy_distance": xy_distance,
                    "score": score,
                }

        if best is not None:
            c = best["cluster"]
            out.at[idx, "parent_cluster_token"] = c["note_token"]
            out.at[idx, "parent_cluster_freq_hz"] = c["freq_hz"]
            out.at[idx, "parent_frame_gap"] = int(best["frame_gap"])
            out.at[idx, "parent_xy_distance"] = float(best["xy_distance"])
            out.at[idx, "lineage_role"] = "spawned_dense_peak"

    return out


def save_png(event_name: str, df: pd.DataFrame, out_png: Path) -> None:
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    specs = [
        ("unassigned_dense_peak", "unassigned peaks", 7, 0.12, "#9ca3af"),
        ("spawned_dense_peak", "spawned peaks", 14, 0.45, "#2563eb"),
        ("cluster_core", "cluster core", 38, 0.88, "#dc2626"),
    ]

    for role, label, base_size, alpha, color in specs:
        sub = df[df["lineage_role"] == role]
        if len(sub) == 0:
            continue
        sizes = [max(base_size, float(a) * base_size * 5.0) for a in sub["relative_amp"]]
        ax.scatter(sub["x12"], sub["y12"], sub["z_time"], s=sizes, alpha=alpha, c=color, label=label)

    ax.set_title(f"Percussion cluster lineage 3D: {event_name}")
    ax.set_xlabel("x12")
    ax.set_ylabel("y12")
    ax.set_zlabel("time_sec")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close()


def save_html(event_name: str, df: pd.DataFrame, out_html: Path) -> None:
    traces = []
    specs = [
        ("unassigned_dense_peak", "unassigned peaks", 0.18, "#9ca3af"),
        ("spawned_dense_peak", "spawned peaks", 0.52, "#2563eb"),
        ("cluster_core", "cluster core", 0.88, "#dc2626"),
    ]

    for role, label, opacity, color in specs:
        sub = df[df["lineage_role"] == role]
        if len(sub) == 0:
            continue
        hover = []
        for _, r in sub.iterrows():
            hover.append(
                f"token={r['note_token']}<br>"
                f"hz={float(r['freq_hz']):.2f}<br>"
                f"time={float(r['time_sec']):.4f}<br>"
                f"amp={float(r['amplitude']):.6f}<br>"
                f"role={r['lineage_role']}<br>"
                f"parent={r.get('parent_cluster_token','')}<br>"
                f"parent_gap={int(r.get('parent_frame_gap',-1))}<br>"
                f"parent_xy={float(r.get('parent_xy_distance',-1.0)):.4f}"
            )
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "name": label,
                "x": sub["x12"].tolist(),
                "y": sub["y12"].tolist(),
                "z": sub["z_time"].tolist(),
                "text": hover,
                "hoverinfo": "text",
                "marker": {
                    "size": [max(2.5, float(a) * 12.0) for a in sub["relative_amp"].tolist()],
                    "opacity": opacity,
                    "color": color,
                },
            }
        )

    payload = json.dumps(traces, ensure_ascii=False)
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{event_name} percussion cluster lineage</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
<h2>Percussion cluster lineage 3D: {event_name}</h2>
<p>cluster_core = stable resonance clusters, spawned_dense_peak = attached peaks, unassigned_dense_peak = free residual peaks.</p>
<div id="plot" style="width:100%;height:900px;"></div>
<script>
const traces = {payload};
const layout = {{
  scene: {{
    xaxis: {{title: "x12"}},
    yaxis: {{title: "y12"}},
    zaxis: {{title: "time_sec"}}
  }},
  margin: {{l: 0, r: 0, b: 0, t: 40}},
  legend: {{orientation: "h"}}
}};
Plotly.newPlot("plot", traces, layout);
</script>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")


def summarize_event(event_name: str, df: pd.DataFrame, out_csv: Path, out_png: Path, out_html: Path) -> dict:
    assigned = df[df["lineage_role"] == "spawned_dense_peak"]
    top_parents = []
    if len(assigned):
        agg = (
            assigned.groupby("parent_cluster_token", as_index=False)
            .agg(
                spawned_count=("note_token", "count"),
                mean_parent_freq_hz=("parent_cluster_freq_hz", "mean"),
                mean_xy_distance=("parent_xy_distance", "mean"),
                mean_frame_gap=("parent_frame_gap", "mean"),
            )
            .sort_values(["spawned_count", "mean_parent_freq_hz"], ascending=[False, False])
        )
        top_parents = agg.head(12).to_dict(orient="records")

    return {
        "event": event_name,
        "points": int(len(df)),
        "cluster_core_points": int((df["lineage_role"] == "cluster_core").sum()),
        "spawned_dense_points": int((df["lineage_role"] == "spawned_dense_peak").sum()),
        "unassigned_dense_points": int((df["lineage_role"] == "unassigned_dense_peak").sum()),
        "spawned_ratio": float((df["lineage_role"] == "spawned_dense_peak").mean()) if len(df) else 0.0,
        "unassigned_ratio": float((df["lineage_role"] == "unassigned_dense_peak").mean()) if len(df) else 0.0,
        "top_parent_clusters": json.dumps(top_parents, ensure_ascii=False),
        "out_csv": str(out_csv),
        "out_png": str(out_png),
        "out_html": str(out_html),
    }


def build_event(points_csv: Path, out_dir: Path, args) -> dict | None:
    df = load_csv(points_csv)
    if df is None or len(df) == 0:
        return None
    required = {"x12", "y12", "z_time", "time_sec", "frame_index", "freq_hz", "note_token", "relative_amp", "component_type"}
    if not required.issubset(set(df.columns)):
        return None

    event_name = points_csv.name.replace("__percussion_spiral3d_points.csv", "")
    out_df = assign_parent_clusters(
        df,
        max_frame_gap=int(args.max_frame_gap),
        max_xy_distance=float(args.max_xy_distance),
        time_weight=float(args.time_weight),
    )

    out_csv = out_dir / f"{event_name}__percussion_cluster_lineage_points.csv"
    out_png = out_dir / f"{event_name}__percussion_cluster_lineage.png"
    out_html = out_dir / f"{event_name}__percussion_cluster_lineage.html"

    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    save_png(event_name, out_df, out_png)
    save_html(event_name, out_df, out_html)
    return summarize_event(event_name, out_df, out_csv, out_png, out_html)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument_name", required=True)
    ap.add_argument("--spiral3d_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_frame_gap", type=int, default=4)
    ap.add_argument("--max_xy_distance", type=float, default=1.8)
    ap.add_argument("--time_weight", type=float, default=0.15)
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    spiral3d_dir = Path(args.spiral3d_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    summaries = []
    skipped = 0

    for points_csv in sorted(spiral3d_dir.glob("*__percussion_spiral3d_points.csv")):
        event_name = points_csv.name.replace("__percussion_spiral3d_points.csv", "")
        out_csv = out_dir / f"{event_name}__percussion_cluster_lineage_points.csv"
        if args.skip_existing and out_csv.exists():
            summary = load_csv(out_dir / f"{args.instrument_name}__cluster_lineage_summary.csv")
            skipped += 1
            continue
        summary = build_event(points_csv, out_dir, args)
        if summary is None:
            skipped += 1
        else:
            summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_csv = out_dir / f"{args.instrument_name}__cluster_lineage_summary.csv"
    summary_json = out_dir / f"{args.instrument_name}__cluster_lineage_summary.json"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    print("PERCUSSION CLUSTER LINEAGE BUILDER DONE")
    print(f"instrument_name : {args.instrument_name}")
    print(f"spiral3d_dir    : {spiral3d_dir}")
    print(f"out_dir         : {out_dir}")
    print(f"built           : {len(summaries)}")
    print(f"skipped         : {skipped}")


if __name__ == "__main__":
    main()
