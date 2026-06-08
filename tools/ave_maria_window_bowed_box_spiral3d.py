# -*- coding: ascii -*-
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from music12.core.notation12 import bij12_to_int, parse_token, step_index0
from music12.blocks.Block002_pipeline.resonance_candidate_inference_core import (
    load_coords_csv,
    load_matrix_csv_memmap,
    load_times_csv,
)


def token_to_spiral_xy(token: str) -> tuple[float, float] | None:
    tok = str(token or "").strip()
    if not tok:
        return None
    try:
        parsed = parse_token(tok)
        octave = bij12_to_int(parsed.oct)
        degree0 = step_index0(parsed.step)
    except Exception:
        return None
    angle = degree0 * (2.0 * math.pi / 12.0)
    radius = float(octave) + float(degree0 + 1) / 12.0
    return radius * math.cos(angle), radius * math.sin(angle)


def _size_from_series(series: pd.Series, lo: float, hi: float) -> pd.Series:
    if len(series) == 0:
        return pd.Series(dtype=float)
    vmax = float(series.max()) or 1.0
    vmin = float(series.min()) or 0.0
    if vmax <= vmin:
        return pd.Series([0.5 * (lo + hi)] * len(series), index=series.index, dtype=float)
    norm = (series.astype(float) - vmin) / (vmax - vmin)
    return lo + norm * (hi - lo)


def load_harmonics(path: Path, top_k_per_frame_harmonic: int = 3) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep = [
        "frame_index",
        "time_sec",
        "probe_index",
        "harmonic_index",
        "observed_note_token",
        "frequency_hz",
        "energy",
        "energy_over_frame_p95",
    ]
    if "extraction_score" in df.columns:
        keep.append("extraction_score")
    df = df[keep].copy()
    if "extraction_score" not in df.columns:
        df["extraction_score"] = df["energy_over_frame_p95"].astype(float)
    df = (
        df.sort_values(
            ["frame_index", "harmonic_index", "extraction_score", "energy_over_frame_p95", "energy"],
            ascending=[True, True, False, False, False],
        )
        .groupby(["frame_index", "harmonic_index"], as_index=False, group_keys=False)
        .head(int(top_k_per_frame_harmonic))
        .copy()
    )
    df["note_token"] = df["observed_note_token"].astype(str)
    xy = df["note_token"].apply(token_to_spiral_xy)
    df["x12"] = xy.apply(lambda p: None if p is None else p[0])
    df["y12"] = xy.apply(lambda p: None if p is None else p[1])
    df = df.dropna(subset=["x12", "y12"]).copy()
    df["component_type"] = "bowed_harmonic"
    df["marker_size"] = _size_from_series(df["energy_over_frame_p95"], 5.0, 15.0)
    df["display_name"] = df["harmonic_index"].apply(lambda h: f"harmonic h{int(h)}")
    return df


def load_box(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep = [
        "frame_index",
        "time_sec",
        "probe_index",
        "observed_micro_symbol",
        "frequency_hz",
        "energy",
        "data_grounded_support_role",
    ]
    df = df[keep].copy()
    df["note_token"] = df["observed_micro_symbol"].astype(str)
    xy = df["note_token"].apply(token_to_spiral_xy)
    df["x12"] = xy.apply(lambda p: None if p is None else p[0])
    df["y12"] = xy.apply(lambda p: None if p is None else p[1])
    df = df.dropna(subset=["x12", "y12"]).copy()
    df["component_type"] = "bowed_box"
    df["marker_size"] = _size_from_series(df["energy"], 9.0, 18.0)
    df["display_name"] = df["data_grounded_support_role"].fillna("box_body")
    return df


def load_background(path: Path, excluded_keys: set[tuple[int, int]]) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep = [
        "frame_index",
        "time_sec",
        "probe_index",
        "observed_micro_symbol",
        "frequency_hz",
        "energy",
        "owner_label",
        "owner_family",
    ]
    df = df[keep].copy()
    key = list(zip(df["frame_index"].astype(int), df["probe_index"].astype(int)))
    df = df[[k not in excluded_keys for k in key]].copy()
    df = df[df["owner_family"].astype(str) != "SECOND_LAYER_OWNER"].copy()
    df["note_token"] = df["observed_micro_symbol"].astype(str)
    xy = df["note_token"].apply(token_to_spiral_xy)
    df["x12"] = xy.apply(lambda p: None if p is None else p[0])
    df["y12"] = xy.apply(lambda p: None if p is None else p[1])
    df = df.dropna(subset=["x12", "y12"]).copy()
    df["component_type"] = df["owner_family"].astype(str).map(
        {
            "PIANOISH_OWNER": "other_pianoish",
            "UNRESOLVED_BACKBONE": "other_unresolved",
        }
    ).fillna("other_pianoish")
    df["marker_size"] = _size_from_series(df["energy"], 2.0, 5.0)
    df["display_name"] = df["owner_label"].fillna("other")
    return df


def load_raw_stream_background(
    *,
    matrix_csv: Path,
    times_csv: Path,
    coords_csv: Path,
    start_frame: int,
    end_frame: int,
    top_k_per_frame: int,
    z_scale: float,
) -> pd.DataFrame:
    matrix, _info = load_matrix_csv_memmap(matrix_csv)
    times = load_times_csv(times_csv)
    coords = load_coords_csv(coords_csv)

    rows: list[dict] = []
    frame_stop = min(int(end_frame), matrix.shape[1] - 1)
    probe_count = matrix.shape[0]

    for frame_idx in range(int(start_frame), frame_stop + 1):
        frame_values = matrix[:, frame_idx]
        if probe_count <= int(top_k_per_frame):
            top_idx = list(range(probe_count))
        else:
            part = frame_values.argpartition(-int(top_k_per_frame))[-int(top_k_per_frame):]
            top_idx = part[np.argsort(frame_values[part])[::-1]].tolist()

        for probe_idx in top_idx:
            energy = float(frame_values[int(probe_idx)])
            if energy <= 0.0:
                continue
            coord = coords[int(probe_idx)]
            xy = token_to_spiral_xy(coord.note_token)
            if xy is None:
                continue
            time_sec = float(times[frame_idx]) if frame_idx < len(times) else frame_idx / 60.0
            rows.append(
                {
                    "frame_index": int(frame_idx),
                    "time_sec": time_sec * float(z_scale),
                    "probe_index": int(probe_idx),
                    "note_token": str(coord.note_token),
                    "frequency_hz": float(coord.frequency_hz),
                    "energy": energy,
                    "component_type": "general_stream",
                    "display_name": "general stream",
                    "x12": xy[0],
                    "y12": xy[1],
                }
            )

    df = pd.DataFrame(rows)
    if len(df):
        df["marker_size"] = _size_from_series(df["energy"], 2.6, 4.8)
    else:
        df["marker_size"] = []
    return df


def save_png(title: str, points: pd.DataFrame, out_png: Path) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    styles = [
        ("general_stream", "general stream", "#c93030", 0.30),
        ("bowed_box", "bowed box/body", "#f5a623", 0.84),
        ("bowed_harmonic", "bowed harmonics", "#2e9f45", 0.96),
    ]
    for comp, label, color, alpha in styles:
        sub = points[points["component_type"] == comp]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub["x12"],
            sub["y12"],
            sub["time_sec"],
            s=sub["marker_size"],
            alpha=alpha,
            c=color,
            label=label,
            edgecolors="none",
        )
    ax.set_title(title)
    ax.set_xlabel("x12")
    ax.set_ylabel("y12")
    ax.set_zlabel("time_sec")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def save_html(title: str, points: pd.DataFrame, out_html: Path) -> None:
    traces = []
    order = [
        ("general_stream", "general stream", "#c93030", 0.30),
        ("bowed_box", "bowed box/body", "#f5a623", 0.84),
        ("bowed_harmonic", "bowed harmonics", "#2e9f45", 0.96),
    ]
    for comp, label, color, alpha in order:
        sub = points[points["component_type"] == comp]
        if len(sub) == 0:
            continue
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "name": label,
                "x": sub["x12"].tolist(),
                "y": sub["y12"].tolist(),
                "z": sub["time_sec"].tolist(),
                "text": (
                    sub["observed_token"].astype(str)
                    + " | "
                    + sub["display_name"].astype(str)
                    + " | hz="
                    + sub["frequency_hz"].map(lambda x: f"{float(x):.2f}")
                ).tolist(),
                "hovertemplate": "%{text}<extra></extra>",
                "marker": {
                    "size": sub["marker_size"].round(3).tolist(),
                    "color": color,
                    "opacity": alpha,
                },
            }
        )

    payload = json.dumps(traces, ensure_ascii=False)
    visible_all = [True] * len(traces)
    visible_bowed_only = [False] * len(traces)
    visible_general_only = [False] * len(traces)
    visible_harmonics_only = [False] * len(traces)
    for idx, trace in enumerate(traces):
        name = str(trace.get("name", ""))
        if name == "general stream":
            visible_general_only[idx] = True
        if name in {"bowed box/body", "bowed harmonics"}:
            visible_bowed_only[idx] = True
        if name == "bowed harmonics":
            visible_harmonics_only[idx] = True

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body style="margin:0;background:#ffffff;color:#1a1a1a;font-family:Arial,sans-serif;">
<div id="plot" style="width:100vw;height:100vh;"></div>
<script>
const traces = {payload};
const layout = {{
  title: {json.dumps(title)},
  paper_bgcolor: "#ffffff",
  plot_bgcolor: "#ffffff",
  font: {{color: "#1a1a1a"}},
  scene: {{
    xaxis: {{title: "x12", gridcolor: "#d8d8d8", zerolinecolor: "#d8d8d8", backgroundcolor: "#ffffff"}},
    yaxis: {{title: "y12", gridcolor: "#d8d8d8", zerolinecolor: "#d8d8d8", backgroundcolor: "#ffffff"}},
    zaxis: {{title: "time_sec", gridcolor: "#d8d8d8", zerolinecolor: "#d8d8d8", backgroundcolor: "#ffffff"}},
    camera: {{eye: {{x: 1.7, y: 1.25, z: 1.25}}}},
    aspectmode: "data"
  }},
  legend: {{bgcolor: "rgba(255,255,255,0.82)"}},
  updatemenus: [{{
    type: "buttons",
    direction: "right",
    x: 0.02,
    y: 1.12,
    showactive: true,
    buttons: [
      {{label: "All", method: "update", args: [{{visible: {json.dumps(visible_all)}}}]}},
      {{label: "Bowed Only", method: "update", args: [{{visible: {json.dumps(visible_bowed_only)}}}]}},
      {{label: "General Only", method: "update", args: [{{visible: {json.dumps(visible_general_only)}}}]}},
      {{label: "Harmonics Only", method: "update", args: [{{visible: {json.dumps(visible_harmonics_only)}}}]}}
    ]
  }}]
}};
Plotly.newPlot("plot", traces, layout, {{responsive: true, displaylogo: false}});
</script>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")


def save_summary(points: pd.DataFrame, out_txt: Path) -> None:
    counts = points.groupby("component_type").size().to_dict()
    lines = [
        "AVE MARIA WINDOW BOWED BOX SPIRAL3D",
        "=" * 72,
        f"total_points     : {len(points)}",
        f"general_stream   : {int(counts.get('general_stream', 0))}",
        f"bowed_harmonic   : {int(counts.get('bowed_harmonic', 0))}",
        f"bowed_box        : {int(counts.get('bowed_box', 0))}",
        "",
        "color_logic:",
        "  general stream   -> red",
        "  bowed harmonics -> green",
        "  bowed box/body  -> orange",
        "",
        "view_modes:",
        "  All",
        "  Bowed Only",
        "  General Only",
        "  Harmonics Only",
    ]
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    base = Path(
        r"E:\Duodecimal_resonant_numeration\Block001_data\Ave_Maria\11_reports_Ave_Maria_clean_probe_rerun_v1\window_11p703s_15p026s_backbone_second_audit_v1"
    )
    ap = argparse.ArgumentParser(description="Build a 3D bowed harmonics vs box/body vs other-scene projection for the Ave Maria window.")
    ap.add_argument("--harmonics-csv", default=str(base / "window_bowed_harmonic_owner_rows_v2.csv"))
    ap.add_argument("--box-csv", default=str(base / "window_second_sustain_data_grounded_support_rows_v1.csv"))
    ap.add_argument("--background-csv", default=str(base / "window_probe_ownership_observations_v1.csv"))
    ap.add_argument("--matrix-csv", default=str(base.parent / "ave_maria_probe_matrix_micro_full.csv"))
    ap.add_argument("--times-csv", default=str(base.parent / "ave_maria_probe_times_micro_full.csv"))
    ap.add_argument("--coords-csv", default=str(base.parent / "ave_maria_probe_coords_micro_full.csv"))
    ap.add_argument("--out-prefix", default="window_bowed_harmonics_box_context_v1")
    ap.add_argument("--out-dir", default=str(base))
    ap.add_argument("--top-k-harmonics", type=int, default=3)
    ap.add_argument("--raw-top-k-per-frame", type=int, default=900)
    ap.add_argument("--z-scale", type=float, default=8.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    harmonics = load_harmonics(Path(args.harmonics_csv), top_k_per_frame_harmonic=int(args.top_k_harmonics))
    harmonics["time_sec"] = harmonics["time_sec"].astype(float) * float(args.z_scale)
    box = load_box(Path(args.box_csv))
    box["time_sec"] = box["time_sec"].astype(float) * float(args.z_scale)
    excluded = set(
        zip(harmonics["frame_index"].astype(int), harmonics["probe_index"].astype(int))
    ) | set(
        zip(box["frame_index"].astype(int), box["probe_index"].astype(int))
    )
    raw_stream = load_raw_stream_background(
        matrix_csv=Path(args.matrix_csv),
        times_csv=Path(args.times_csv),
        coords_csv=Path(args.coords_csv),
        start_frame=int(min(harmonics["frame_index"].min(), box["frame_index"].min())),
        end_frame=int(max(harmonics["frame_index"].max(), box["frame_index"].max())),
        top_k_per_frame=int(args.raw_top_k_per_frame),
        z_scale=float(args.z_scale),
    )

    points = pd.concat([raw_stream, box, harmonics], ignore_index=True)
    points = points[
        [
            "frame_index",
            "time_sec",
            "probe_index",
            "note_token",
            "frequency_hz",
            "energy",
            "component_type",
            "display_name",
            "x12",
            "y12",
            "marker_size",
        ]
    ].copy()
    points = points.rename(columns={"note_token": "observed_token"})

    out_csv = out_dir / f"{args.out_prefix}.csv"
    out_png = out_dir / f"{args.out_prefix}.png"
    out_html = out_dir / f"{args.out_prefix}.html"
    out_txt = out_dir / f"{args.out_prefix}_summary.txt"

    points.to_csv(out_csv, index=False, encoding="utf-8-sig")
    title = "Ave Maria 11.703-15.026: bowed instrument inside the full stream"
    save_png(title, points, out_png)
    save_html(title, points, out_html)
    save_summary(points, out_txt)

    print(out_csv)
    print(out_png)
    print(out_html)
    print(out_txt)


if __name__ == "__main__":
    main()
