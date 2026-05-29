# -*- coding: utf-8 -*-
"""
HARMONIC CHAIN SPIRAL 3D BUILDER

Create a second 3D visualization layer for Block004 notes:

- which harmonic index forms the core chain
- which note-box and residual points are attracted to that harmonic chain
- how the spawned resonance lineage evolves through time

Input:
  50_spiral3d/<note>__spiral3d_points.csv

Output:
  55_harmonic_chain_spiral3d/
    <note>__harmonic_chain_spiral3d_points.csv
    <note>__harmonic_chain_spiral3d.png
    <note>__harmonic_chain_spiral3d.html
    <instrument>__harmonic_chain_spiral3d_summary.csv
    <instrument>__harmonic_chain_spiral3d_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_csv_safe(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def nearest_parent_harmonic(
    row: pd.Series,
    chain_df: pd.DataFrame,
    max_frame_gap: int,
    max_xy_distance: float,
    time_weight: float,
) -> tuple[Optional[int], Optional[float], Optional[int]]:
    frame_idx = int(row["frame_idx"])
    nearby = chain_df[
        (chain_df["frame_idx"] >= frame_idx - max_frame_gap)
        & (chain_df["frame_idx"] <= frame_idx + max_frame_gap)
    ].copy()

    if len(nearby) == 0:
        return None, None, None

    nearby["xy_distance"] = (
        (nearby["x12"].astype(float) - float(row["x12"])) ** 2
        + (nearby["y12"].astype(float) - float(row["y12"])) ** 2
    ) ** 0.5
    nearby["frame_distance"] = (nearby["frame_idx"].astype(int) - frame_idx).abs()
    nearby["parent_score"] = nearby["xy_distance"] + nearby["frame_distance"] * float(time_weight)

    nearby = nearby.sort_values(["parent_score", "xy_distance", "frame_distance"])
    best = nearby.iloc[0]

    xy_distance = float(best["xy_distance"])
    if xy_distance > float(max_xy_distance):
        return None, xy_distance, int(best["harmonic_index"])

    return int(best["harmonic_index"]), xy_distance, int(best["frame_idx"])


def annotate_lineage(
    points_df: pd.DataFrame,
    max_frame_gap: int,
    max_xy_distance: float,
    time_weight: float,
) -> pd.DataFrame:
    df = points_df.copy()
    if "frame_idx" not in df.columns and "frame_index" in df.columns:
        df = df.rename(columns={"frame_index": "frame_idx"})

    required = {"frame_idx", "x12", "y12", "time_sec", "component_type", "harmonic_index"}
    missing = required.difference(set(df.columns))
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    df["parent_harmonic_index"] = ""
    df["parent_xy_distance"] = ""
    df["parent_frame_idx"] = ""
    df["lineage_role"] = ""

    chain_df = df[df["component_type"] == "chain"].copy()
    chain_df = chain_df[pd.to_numeric(chain_df["harmonic_index"], errors="coerce").notna()].copy()
    if len(chain_df) == 0:
        return df

    chain_df["harmonic_index"] = chain_df["harmonic_index"].astype(int)

    for idx, row in df.iterrows():
        component_type = str(row["component_type"])
        if component_type == "chain":
            try:
                h = int(row["harmonic_index"])
            except Exception:
                continue
            df.at[idx, "parent_harmonic_index"] = h
            df.at[idx, "parent_xy_distance"] = 0.0
            df.at[idx, "parent_frame_idx"] = int(row["frame_idx"])
            df.at[idx, "lineage_role"] = "harmonic_core"
            continue

        parent_h, xy_distance, parent_frame_idx = nearest_parent_harmonic(
            row=row,
            chain_df=chain_df,
            max_frame_gap=max_frame_gap,
            max_xy_distance=max_xy_distance,
            time_weight=time_weight,
        )

        if parent_h is None:
            df.at[idx, "lineage_role"] = "unassigned_resonance"
            if xy_distance is not None:
                df.at[idx, "parent_xy_distance"] = xy_distance
            continue

        df.at[idx, "parent_harmonic_index"] = parent_h
        df.at[idx, "parent_xy_distance"] = xy_distance if xy_distance is not None else ""
        df.at[idx, "parent_frame_idx"] = parent_frame_idx if parent_frame_idx is not None else ""

        if component_type == "note_box":
            df.at[idx, "lineage_role"] = "spawned_note_box"
        else:
            df.at[idx, "lineage_role"] = "spawned_residual"

    return df


def harmonic_color_map(harmonic_indices: list[int]) -> dict[int, str]:
    palette = [
        "#d62839",
        "#f77f00",
        "#fcbf49",
        "#2a9d8f",
        "#1d4ed8",
        "#6a4c93",
        "#9d4edd",
        "#ef476f",
        "#118ab2",
        "#06d6a0",
        "#8d99ae",
        "#bc4749",
    ]
    out = {}
    for i, h in enumerate(sorted(set(harmonic_indices))):
        out[int(h)] = palette[i % len(palette)]
    return out


def downsample_even(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    ordered = df.sort_values("time_sec").reset_index(drop=True)
    if max_points <= 1:
        return ordered.iloc[[0]]
    indices = []
    last = len(ordered) - 1
    for i in range(max_points):
        idx = round(i * last / (max_points - 1))
        indices.append(idx)
    return ordered.iloc[sorted(set(indices))].copy()


def sampled_for_visual(
    df: pd.DataFrame,
    max_core_points_plot: int,
    max_note_box_points_plot: int,
    max_residual_points_plot: int,
    max_unassigned_points_plot: int,
) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []

    assigned = df[pd.to_numeric(df["parent_harmonic_index"], errors="coerce").notna()].copy()
    if len(assigned):
        assigned["parent_harmonic_index"] = assigned["parent_harmonic_index"].astype(int)
        for parent_h in sorted(set(assigned["parent_harmonic_index"].tolist())):
            sub = assigned[assigned["parent_harmonic_index"] == parent_h].copy()
            core = sub[sub["lineage_role"] == "harmonic_core"]
            if len(core):
                chunks.append(downsample_even(core, max_core_points_plot))
            box = sub[sub["lineage_role"] == "spawned_note_box"]
            if len(box):
                chunks.append(downsample_even(box, max_note_box_points_plot))
            residual = sub[sub["lineage_role"] == "spawned_residual"]
            if len(residual):
                chunks.append(downsample_even(residual, max_residual_points_plot))

    unassigned = df[df["lineage_role"] == "unassigned_resonance"]
    if len(unassigned):
        chunks.append(downsample_even(unassigned, max_unassigned_points_plot))

    if not chunks:
        return df.iloc[0:0].copy()
    return pd.concat(chunks, ignore_index=True)


def save_png(note_name: str, df: pd.DataFrame, out_png: str) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    assigned = df[pd.to_numeric(df["parent_harmonic_index"], errors="coerce").notna()].copy()
    assigned["parent_harmonic_index"] = assigned["parent_harmonic_index"].astype(int)
    colors = harmonic_color_map(assigned["parent_harmonic_index"].tolist())

    for harmonic_index in sorted(colors):
        color = colors[harmonic_index]
        sub = assigned[assigned["parent_harmonic_index"] == harmonic_index].copy()

        core = sub[sub["lineage_role"] == "harmonic_core"].sort_values("time_sec")
        if len(core):
            ax.plot(
                core["x12"],
                core["y12"],
                core["z_time"],
                color=color,
                linewidth=2.0,
                alpha=0.95,
                label=f"h{harmonic_index} core",
            )
            ax.scatter(
                core["x12"],
                core["y12"],
                core["z_time"],
                s=[max(22.0, float(a) * 85.0) for a in core["relative_amp"]],
                c=color,
                alpha=0.95,
                edgecolors="none",
            )

        box = sub[sub["lineage_role"] == "spawned_note_box"]
        if len(box):
            ax.scatter(
                box["x12"],
                box["y12"],
                box["z_time"],
                s=26,
                c=color,
                alpha=0.55,
                edgecolors="none",
            )

        residual = sub[sub["lineage_role"] == "spawned_residual"]
        if len(residual):
            ax.scatter(
                residual["x12"],
                residual["y12"],
                residual["z_time"],
                s=9,
                c=color,
                alpha=0.20,
                edgecolors="none",
            )

    unassigned = df[df["lineage_role"] == "unassigned_resonance"]
    if len(unassigned):
        ax.scatter(
            unassigned["x12"],
            unassigned["y12"],
            unassigned["z_time"],
            s=8,
            c="#9e9e9e",
            alpha=0.15,
            label="unassigned resonance",
            edgecolors="none",
        )

    ax.set_title(f"Harmonic lineage 3D: {note_name}")
    ax.set_xlabel("x12")
    ax.set_ylabel("y12")
    ax.set_zlabel("time_sec")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def save_html(note_name: str, df: pd.DataFrame, out_html: str) -> None:
    traces = []
    assigned = df[pd.to_numeric(df["parent_harmonic_index"], errors="coerce").notna()].copy()
    assigned["parent_harmonic_index"] = assigned["parent_harmonic_index"].astype(int)
    colors = harmonic_color_map(assigned["parent_harmonic_index"].tolist())

    for harmonic_index in sorted(colors):
        color = colors[harmonic_index]
        sub = assigned[assigned["parent_harmonic_index"] == harmonic_index].copy()

        core = sub[sub["lineage_role"] == "harmonic_core"].sort_values("time_sec")
        if len(core):
            hover = []
            for _, r in core.iterrows():
                hover.append(
                    f"h={harmonic_index}<br>"
                    f"token={r['note_token']}<br>"
                    f"time={float(r['time_sec']):.4f}<br>"
                    f"amp={float(r['amplitude']):.6f}<br>"
                    f"role={r['lineage_role']}"
                )
            traces.append(
                {
                    "type": "scatter3d",
                    "mode": "lines+markers",
                    "name": f"h{harmonic_index} core",
                    "x": core["x12"].tolist(),
                    "y": core["y12"].tolist(),
                    "z": core["z_time"].tolist(),
                    "text": hover,
                    "hoverinfo": "text",
                    "line": {"color": color, "width": 5},
                    "marker": {
                        "size": [max(4.0, float(a) * 14.0) for a in core["relative_amp"]],
                        "opacity": 0.95,
                        "color": color,
                    },
                }
            )

        for role_name, label, opacity, size in [
            ("spawned_note_box", f"h{harmonic_index} note box", 0.58, 4.0),
            ("spawned_residual", f"h{harmonic_index} residual", 0.20, 2.5),
        ]:
            part = sub[sub["lineage_role"] == role_name]
            if len(part) == 0:
                continue
            hover = []
            for _, r in part.iterrows():
                hover.append(
                    f"h={harmonic_index}<br>"
                    f"token={r['note_token']}<br>"
                    f"time={float(r['time_sec']):.4f}<br>"
                    f"role={r['lineage_role']}<br>"
                    f"parent_xy_distance={r.get('parent_xy_distance','')}"
                )
            traces.append(
                {
                    "type": "scatter3d",
                    "mode": "markers",
                    "name": label,
                    "x": part["x12"].tolist(),
                    "y": part["y12"].tolist(),
                    "z": part["z_time"].tolist(),
                    "text": hover,
                    "hoverinfo": "text",
                    "marker": {
                        "size": size,
                        "opacity": opacity,
                        "color": color,
                    },
                }
            )

    unassigned = df[df["lineage_role"] == "unassigned_resonance"]
    if len(unassigned):
        hover = []
        for _, r in unassigned.iterrows():
            hover.append(
                f"token={r['note_token']}<br>"
                f"time={float(r['time_sec']):.4f}<br>"
                f"role=unassigned_resonance"
            )
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "name": "unassigned resonance",
                "x": unassigned["x12"].tolist(),
                "y": unassigned["y12"].tolist(),
                "z": unassigned["z_time"].tolist(),
                "text": hover,
                "hoverinfo": "text",
                "marker": {
                    "size": 2.5,
                    "opacity": 0.14,
                    "color": "#9e9e9e",
                },
            }
        )

    payload = json.dumps(traces, ensure_ascii=False)
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{note_name} harmonic chain spiral3d</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
<h2>Harmonic lineage 3D: {note_name}</h2>
<p>Each color is a harmonic core chain. Fainter points show note-box and residual
resonance assigned to that harmonic lineage.</p>
<div id="plot" style="width:100%;height:920px;"></div>
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


def process_note(
    note_name: str,
    csv_path: str,
    out_dir: str,
    max_frame_gap: int,
    max_xy_distance: float,
    time_weight: float,
    max_core_points_plot: int,
    max_note_box_points_plot: int,
    max_residual_points_plot: int,
    max_unassigned_points_plot: int,
) -> Optional[dict]:
    df = load_csv_safe(csv_path)
    if df is None or len(df) == 0:
        return None

    out_df = annotate_lineage(
        points_df=df,
        max_frame_gap=max_frame_gap,
        max_xy_distance=max_xy_distance,
        time_weight=time_weight,
    )

    out_csv = os.path.join(out_dir, f"{note_name}__harmonic_chain_spiral3d_points.csv")
    out_png = os.path.join(out_dir, f"{note_name}__harmonic_chain_spiral3d.png")
    out_html = os.path.join(out_dir, f"{note_name}__harmonic_chain_spiral3d.html")

    out_df.to_csv(out_csv, index=False)
    visual_df = sampled_for_visual(
        df=out_df,
        max_core_points_plot=max_core_points_plot,
        max_note_box_points_plot=max_note_box_points_plot,
        max_residual_points_plot=max_residual_points_plot,
        max_unassigned_points_plot=max_unassigned_points_plot,
    )

    save_png(note_name, visual_df, out_png)
    save_html(note_name, visual_df, out_html)

    assigned = out_df[pd.to_numeric(out_df["parent_harmonic_index"], errors="coerce").notna()].copy()
    assigned_harmonics = (
        sorted(set(int(x) for x in assigned["parent_harmonic_index"].tolist()))
        if len(assigned)
        else []
    )

    return {
        "note": note_name,
        "points": int(len(out_df)),
        "harmonic_core_points": int((out_df["lineage_role"] == "harmonic_core").sum()),
        "spawned_note_box_points": int((out_df["lineage_role"] == "spawned_note_box").sum()),
        "spawned_residual_points": int((out_df["lineage_role"] == "spawned_residual").sum()),
        "unassigned_points": int((out_df["lineage_role"] == "unassigned_resonance").sum()),
        "visual_points": int(len(visual_df)),
        "assigned_harmonics": " ".join(str(x) for x in assigned_harmonics),
        "out_csv": out_csv,
        "out_png": out_png,
        "out_html": out_html,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument_name", required=True)
    ap.add_argument("--spiral3d_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_frame_gap", type=int, default=4)
    ap.add_argument("--max_xy_distance", type=float, default=1.8)
    ap.add_argument("--time_weight", type=float, default=0.15)
    ap.add_argument("--max_core_points_plot", type=int, default=900)
    ap.add_argument("--max_note_box_points_plot", type=int, default=240)
    ap.add_argument("--max_residual_points_plot", type=int, default=180)
    ap.add_argument("--max_unassigned_points_plot", type=int, default=220)
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    summaries = []
    skipped = 0

    for f in sorted(os.listdir(args.spiral3d_dir)):
        if not f.endswith("__spiral3d_points.csv"):
            continue
        note_name = f.replace("__spiral3d_points.csv", "")
        path = os.path.join(args.spiral3d_dir, f)
        out_csv = os.path.join(args.out_dir, f"{note_name}__harmonic_chain_spiral3d_points.csv")
        out_png = os.path.join(args.out_dir, f"{note_name}__harmonic_chain_spiral3d.png")
        out_html = os.path.join(args.out_dir, f"{note_name}__harmonic_chain_spiral3d.html")

        if args.skip_existing and os.path.exists(out_csv) and os.path.exists(out_png) and os.path.exists(out_html):
            existing_df = load_csv_safe(out_csv)
            if existing_df is not None:
                assigned = existing_df[pd.to_numeric(existing_df["parent_harmonic_index"], errors="coerce").notna()].copy()
                assigned_harmonics = (
                    sorted(set(int(x) for x in assigned["parent_harmonic_index"].tolist()))
                    if len(assigned)
                    else []
                )
                summaries.append(
                    {
                        "note": note_name,
                        "points": int(len(existing_df)),
                        "harmonic_core_points": int((existing_df["lineage_role"] == "harmonic_core").sum()),
                        "spawned_note_box_points": int((existing_df["lineage_role"] == "spawned_note_box").sum()),
                        "spawned_residual_points": int((existing_df["lineage_role"] == "spawned_residual").sum()),
                        "unassigned_points": int((existing_df["lineage_role"] == "unassigned_resonance").sum()),
                        "visual_points": None,
                        "assigned_harmonics": " ".join(str(x) for x in assigned_harmonics),
                        "out_csv": out_csv,
                        "out_png": out_png,
                        "out_html": out_html,
                        "reused_existing": True,
                    }
                )
                continue

        result = process_note(
            note_name=note_name,
            csv_path=path,
            out_dir=args.out_dir,
            max_frame_gap=args.max_frame_gap,
            max_xy_distance=args.max_xy_distance,
            time_weight=args.time_weight,
            max_core_points_plot=args.max_core_points_plot,
            max_note_box_points_plot=args.max_note_box_points_plot,
            max_residual_points_plot=args.max_residual_points_plot,
            max_unassigned_points_plot=args.max_unassigned_points_plot,
        )
        if result is None:
            skipped += 1
        else:
            result["reused_existing"] = False
            summaries.append(result)

    summary_df = pd.DataFrame(summaries)
    summary_csv = os.path.join(args.out_dir, f"{args.instrument_name}__harmonic_chain_spiral3d_summary.csv")
    summary_json = os.path.join(args.out_dir, f"{args.instrument_name}__harmonic_chain_spiral3d_summary.json")
    summary_df.to_csv(summary_csv, index=False)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print("HARMONIC CHAIN SPIRAL 3D BUILDER DONE")
    print(f"instrument_name : {args.instrument_name}")
    print(f"spiral3d_dir    : {args.spiral3d_dir}")
    print(f"out_dir         : {args.out_dir}")
    print(f"built           : {len(summaries)}")
    print(f"skipped         : {skipped}")


if __name__ == "__main__":
    main()
