# -*- coding: utf-8 -*-
"""
NOTE BOX SPIRAL 3D BUILDER

Создаёт 3D-визуализацию ноты во времени:

X = x12
Y = y12
Z = time_sec

Типы точек:
- chain_candidate / note harmonic
- note_box
- dense_other

Вход:
  10_reports/<note>/__spiral12_clean_points.csv
  10_reports/<note>/__root_consensus_summary.txt
  30_note_box_profiles/<note>__note_box_profile.csv

Выход:
  50_spiral3d/
    <note>__spiral3d_points.csv
    <note>__spiral3d.png
    <note>__spiral3d.html
    <instrument>__spiral3d_summary.csv
"""

import os
import math
import json
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_csv_safe(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def cents_diff(hz1, hz2):
    if hz1 <= 0 or hz2 <= 0:
        return 9999.0
    return 1200.0 * math.log2(hz1 / hz2)


def normalize_points_columns(df):
    ren = {}

    if "freq_hz" in df.columns and "hz" not in df.columns:
        ren["freq_hz"] = "hz"

    if "amplitude" in df.columns and "amp" not in df.columns:
        ren["amplitude"] = "amp"

    if "frame_index" in df.columns and "frame_idx" not in df.columns:
        ren["frame_index"] = "frame_idx"

    if ren:
        df = df.rename(columns=ren)

    return df


def extract_root_hz(root_path):
    if not root_path or not os.path.exists(root_path):
        return None

    with open(root_path, "r", encoding="utf-8") as f:
        for line in f:
            if "consensus_root_hz" in line:
                try:
                    return float(line.split(":")[1].strip())
                except Exception:
                    pass

    return None


def find_file(note_dir, suffix):
    for f in os.listdir(note_dir):
        if f.endswith(suffix):
            return os.path.join(note_dir, f)
    return None


def harmonic_index_for_hz(hz, root_hz, tolerance_cents, harmonic_min=1, harmonic_max=12):
    if root_hz is None:
        return None

    best_h = None
    best_delta = 9999.0

    for h in range(harmonic_min, harmonic_max + 1):
        expected = root_hz * h
        delta = abs(cents_diff(hz, expected))

        if delta < best_delta:
            best_delta = delta
            best_h = h

    if best_delta <= tolerance_cents:
        return best_h

    return None


def build_note_spiral3d_points(
    note_name,
    note_dir,
    note_box_dir,
    out_dir,
    tolerance_cents,
):
    points_path = find_file(note_dir, "__spiral12_clean_points.csv")
    root_path = find_file(note_dir, "__root_consensus_summary.txt")

    if not points_path or not root_path:
        return None

    points_df = load_csv_safe(points_path)
    if points_df is None or len(points_df) == 0:
        return None

    points_df = normalize_points_columns(points_df)

    required = {"hz", "amp", "frame_idx", "note_token", "x12", "y12"}
    if not required.issubset(set(points_df.columns)):
        return None

    if "time_sec" not in points_df.columns:
        if "frame_idx" in points_df.columns:
            points_df["time_sec"] = points_df["frame_idx"].astype(float)
        else:
            points_df["time_sec"] = range(len(points_df))

    root_hz = extract_root_hz(root_path)
    if root_hz is None:
        return None

    profile_path = os.path.join(note_box_dir, f"{note_name}__note_box_profile.csv")
    box_tokens = set()

    profile_df = load_csv_safe(profile_path)
    if profile_df is not None and "token" in profile_df.columns:
        box_tokens = set(profile_df["token"].astype(str).tolist())

    rows = []

    max_amp = float(points_df["amp"].max()) if len(points_df) else 1.0
    if max_amp <= 0:
        max_amp = 1.0

    for _, r in points_df.iterrows():
        hz = float(r["hz"])
        token = str(r["note_token"])

        h = harmonic_index_for_hz(
            hz=hz,
            root_hz=root_hz,
            tolerance_cents=tolerance_cents,
        )

        is_chain = h is not None
        is_box = token in box_tokens and not is_chain

        if is_chain:
            component_type = "chain"
        elif is_box:
            component_type = "note_box"
        else:
            component_type = "dense_other"

        amp = float(r["amp"])
        rel_amp = amp / max_amp

        rows.append(
            {
                "source_note": note_name,
                "time_sec": float(r["time_sec"]),
                "frame_idx": int(r["frame_idx"]),
                "x12": float(r["x12"]),
                "y12": float(r["y12"]),
                "z_time": float(r["time_sec"]),
                "hz": hz,
                "note_token": token,
                "amplitude": amp,
                "relative_amp": rel_amp,
                "component_type": component_type,
                "is_chain": int(is_chain),
                "is_note_box": int(is_box),
                "harmonic_index": h if h is not None else "",
                "root_hz": root_hz,
            }
        )

    out_df = pd.DataFrame(rows)

    out_csv = os.path.join(out_dir, f"{note_name}__spiral3d_points.csv")
    out_df.to_csv(out_csv, index=False)

    save_png(note_name, out_df, os.path.join(out_dir, f"{note_name}__spiral3d.png"))
    save_html(note_name, out_df, os.path.join(out_dir, f"{note_name}__spiral3d.html"))

    return {
        "note": note_name,
        "points": len(out_df),
        "chain_points": int(out_df["is_chain"].sum()),
        "note_box_points": int(out_df["is_note_box"].sum()),
        "dense_other_points": int((out_df["component_type"] == "dense_other").sum()),
        "root_hz": root_hz,
        "out_csv": out_csv,
        "out_png": os.path.join(out_dir, f"{note_name}__spiral3d.png"),
        "out_html": os.path.join(out_dir, f"{note_name}__spiral3d.html"),
    }


def save_png(note_name, df, out_png):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    types = [
        ("dense_other", "dense other", 10, 0.18),
        ("note_box", "note box", 34, 0.75),
        ("chain", "chain", 55, 0.9),
    ]

    for component_type, label, base_size, alpha in types:
        sub = df[df["component_type"] == component_type]
        if len(sub) == 0:
            continue

        sizes = [max(base_size, float(a) * base_size * 4) for a in sub["relative_amp"]]

        ax.scatter(
            sub["x12"],
            sub["y12"],
            sub["z_time"],
            s=sizes,
            alpha=alpha,
            label=label,
        )

    ax.set_title(f"3D 12-spiral over time: {note_name}")
    ax.set_xlabel("x12")
    ax.set_ylabel("y12")
    ax.set_zlabel("time_sec")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close()


def save_html(note_name, df, out_html):
    """
    Самодостаточный HTML с Plotly CDN.
    Если интернета нет, CSV и PNG всё равно остаются основными артефактами.
    """
    traces = []

    type_labels = {
        "chain": "chain",
        "note_box": "note box",
        "dense_other": "dense other",
    }

    for component_type in ["dense_other", "note_box", "chain"]:
        sub = df[df["component_type"] == component_type]
        if len(sub) == 0:
            continue

        hover = []
        for _, r in sub.iterrows():
            hover.append(
                f"token={r['note_token']}<br>"
                f"hz={r['hz']:.2f}<br>"
                f"time={r['time_sec']:.4f}<br>"
                f"amp={r['amplitude']:.6f}<br>"
                f"type={r['component_type']}<br>"
                f"h={r['harmonic_index']}"
            )

        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "name": type_labels[component_type],
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
                    "opacity": 0.75 if component_type != "dense_other" else 0.25,
                },
            }
        )

    payload = json.dumps(traces, ensure_ascii=False)

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{note_name} spiral3d</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
<h2>3D 12-spiral over time: {note_name}</h2>
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
    ap.add_argument("--note_box_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--harmonic_tolerance_cents", type=float, default=18.0)
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    summaries = []
    skipped = 0

    for d in os.listdir(args.reports_root):
        note_dir = os.path.join(args.reports_root, d)
        if not os.path.isdir(note_dir):
            continue

        result = build_note_spiral3d_points(
            note_name=d,
            note_dir=note_dir,
            note_box_dir=args.note_box_dir,
            out_dir=args.out_dir,
            tolerance_cents=args.harmonic_tolerance_cents,
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

    print("NOTE BOX SPIRAL 3D BUILDER DONE")
    print(f"instrument_name : {args.instrument_name}")
    print(f"reports_root    : {args.reports_root}")
    print(f"note_box_dir    : {args.note_box_dir}")
    print(f"out_dir         : {args.out_dir}")
    print(f"built           : {len(summaries)}")
    print(f"skipped         : {skipped}")


if __name__ == "__main__":
    main()