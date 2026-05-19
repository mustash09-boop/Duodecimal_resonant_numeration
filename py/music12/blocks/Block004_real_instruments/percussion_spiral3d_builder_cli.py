# -*- coding: utf-8 -*-
"""
PERCUSSION SPIRAL 3D BUILDER

Строит 3D-спираль ударных событий во времени.

Вход:
  10_reports/<event>/*__percussion_dense.csv
  10_reports/<event>/*__percussion_frequency_clusters.csv

Выход:
  50_spiral3d/
    <event>__percussion_spiral3d_points.csv
    <event>__percussion_spiral3d.png
    <event>__percussion_spiral3d.html
    percussion__spiral3d_summary.csv
"""

import os
import math
import json
import argparse

import pandas as pd
import matplotlib.pyplot as plt


DIGITS12 = "123456789ABC"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_csv_safe(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def safe_name(s):
    return str(s).replace(" ", "_").replace("/", "_").replace("\\", "_")


def token_to_spiral_xy(token):
    if token is None:
        return None

    token = str(token).strip()
    clean = token.split("'")[0].replace("-", "")

    if "." not in clean:
        return None

    octave_s, degree_s = clean.split(".", 1)

    try:
        octave = int(octave_s)
        degree = DIGITS12.index(degree_s[0]) + 1
    except Exception:
        return None

    angle = (degree - 1) * (2.0 * math.pi / 12.0)
    radius = octave + degree / 12.0

    return radius * math.cos(angle), radius * math.sin(angle)


def find_file(event_dir, suffix):
    for f in os.listdir(event_dir):
        if f.endswith(suffix):
            return os.path.join(event_dir, f)
    return None


def build_event_points(event_name, event_dir, out_dir):
    dense_path = find_file(event_dir, "__percussion_dense.csv")
    cluster_path = find_file(event_dir, "__percussion_frequency_clusters.csv")

    if not dense_path:
        return None

    dense_df = load_csv_safe(dense_path)
    if dense_df is None or len(dense_df) == 0:
        return None

    required = {"time_sec", "freq_hz", "amplitude", "relative_amp", "note_token"}
    if not required.issubset(set(dense_df.columns)):
        return None

    cluster_tokens = set()

    clusters_df = load_csv_safe(cluster_path) if cluster_path else None
    if clusters_df is not None and "token" in clusters_df.columns:
        cluster_tokens = set(clusters_df["token"].astype(str).tolist())

    rows = []

    for _, r in dense_df.iterrows():
        token = str(r.get("note_token", ""))
        xy = token_to_spiral_xy(token)

        if not xy:
            continue

        is_cluster = token in cluster_tokens

        rows.append({
            "source_event": event_name,
            "time_sec": float(r.get("time_sec", 0.0)),
            "z_time": float(r.get("time_sec", 0.0)),
            "frame_index": int(r.get("frame_index", 0)),
            "x12": float(xy[0]),
            "y12": float(xy[1]),
            "freq_hz": float(r.get("freq_hz", 0.0)),
            "note_token": token,
            "amplitude": float(r.get("amplitude", 0.0)),
            "relative_amp": float(r.get("relative_amp", 0.0)),
            "component_type": "resonance_cluster" if is_cluster else "dense_peak",
            "is_cluster": int(is_cluster),
        })

    if not rows:
        return None

    out_df = pd.DataFrame(rows)

    out_csv = os.path.join(out_dir, f"{event_name}__percussion_spiral3d_points.csv")
    out_png = os.path.join(out_dir, f"{event_name}__percussion_spiral3d.png")
    out_html = os.path.join(out_dir, f"{event_name}__percussion_spiral3d.html")

    out_df.to_csv(out_csv, index=False)

    save_png(event_name, out_df, out_png)
    save_html(event_name, out_df, out_html)

    return {
        "event": event_name,
        "points": int(len(out_df)),
        "cluster_points": int(out_df["is_cluster"].sum()),
        "dense_points": int((out_df["component_type"] == "dense_peak").sum()),
        "out_csv": out_csv,
        "out_png": out_png,
        "out_html": out_html,
    }


def save_png(event_name, df, out_png):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    for component_type, label, base_size, alpha in [
        ("dense_peak", "dense peaks", 9, 0.18),
        ("resonance_cluster", "resonance clusters", 45, 0.85),
    ]:
        sub = df[df["component_type"] == component_type]

        if len(sub) == 0:
            continue

        sizes = [max(base_size, float(a) * base_size * 5.0) for a in sub["relative_amp"]]

        ax.scatter(
            sub["x12"],
            sub["y12"],
            sub["z_time"],
            s=sizes,
            alpha=alpha,
            label=label,
        )

    ax.set_title(f"Percussion 3D 12-spiral over time: {event_name}")
    ax.set_xlabel("x12")
    ax.set_ylabel("y12")
    ax.set_zlabel("time_sec")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close()


def save_html(event_name, df, out_html):
    traces = []

    for component_type, label, opacity in [
        ("dense_peak", "dense peaks", 0.25),
        ("resonance_cluster", "resonance clusters", 0.85),
    ]:
        sub = df[df["component_type"] == component_type]

        if len(sub) == 0:
            continue

        hover = []
        for _, r in sub.iterrows():
            hover.append(
                f"token={r['note_token']}<br>"
                f"hz={r['freq_hz']:.2f}<br>"
                f"time={r['time_sec']:.4f}<br>"
                f"amp={r['amplitude']:.6f}<br>"
                f"type={r['component_type']}"
            )

        traces.append({
            "type": "scatter3d",
            "mode": "markers",
            "name": label,
            "x": sub["x12"].tolist(),
            "y": sub["y12"].tolist(),
            "z": sub["z_time"].tolist(),
            "text": hover,
            "hoverinfo": "text",
            "marker": {
                "size": [
                    max(2.5, float(a) * 12.0)
                    for a in sub["relative_amp"].tolist()
                ],
                "opacity": opacity,
            },
        })

    payload = json.dumps(traces, ensure_ascii=False)

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{event_name} percussion spiral3d</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
<h2>Percussion 3D 12-spiral over time: {event_name}</h2>
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

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--instrument_name", required=True)
    ap.add_argument("--reports_root", required=True)
    ap.add_argument("--out_dir", required=True)

    args = ap.parse_args()

    ensure_dir(args.out_dir)

    summaries = []
    skipped = 0

    for d in os.listdir(args.reports_root):
        event_dir = os.path.join(args.reports_root, d)

        if not os.path.isdir(event_dir):
            continue

        result = build_event_points(
            event_name=d,
            event_dir=event_dir,
            out_dir=args.out_dir,
        )

        if result is None:
            skipped += 1
        else:
            summaries.append(result)

    summary_df = pd.DataFrame(summaries)

    summary_csv = os.path.join(args.out_dir, f"{args.instrument_name}__spiral3d_summary.csv")
    summary_json = os.path.join(args.out_dir, f"{args.instrument_name}__spiral3d_summary.json")

    summary_df.to_csv(summary_csv, index=False)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print("PERCUSSION SPIRAL 3D BUILDER DONE")
    print(f"instrument_name : {args.instrument_name}")
    print(f"reports_root    : {args.reports_root}")
    print(f"out_dir         : {args.out_dir}")
    print(f"built           : {len(summaries)}")
    print(f"skipped         : {skipped}")


if __name__ == "__main__":
    main()