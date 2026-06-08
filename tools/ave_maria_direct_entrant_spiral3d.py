# -*- coding: ascii -*-
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from music12.core.notation12 import bij12_to_int, parse_token, step_index0


REPORTS = Path(
    r"E:\Duodecimal_resonant_numeration\Block001_data\Ave_Maria\10_reports_Ave_Maria"
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def load_probe_row(matrix_csv: Path, probe_index: int) -> dict[str, float]:
    with matrix_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["probe_index"]) == int(probe_index):
                return {k: float(v) for k, v in row.items() if k != "probe_index"}
    raise KeyError(f"probe_index not found: {probe_index}")


def load_direct_targets(detector_csv: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with detector_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "label": str(row["label"]).strip(),
                    "track_name": str(row["track_name"]).strip(),
                    "reference_note_token": str(row["reference_note_token"]).strip(),
                    "exact_probe_index": int(row["exact_probe_index"]),
                    "exact_probe_band_token": str(row["exact_probe_band_token"]).strip(),
                    "exact_probe_hz": float(row["exact_probe_hz"]),
                    "exact_first_hit_frame": int(row["exact_first_hit_frame"] or 0),
                    "exact_best_run_start_frame": int(row["exact_best_run_start_frame"] or 0),
                    "exact_best_run_end_frame": int(row["exact_best_run_end_frame"] or 0),
                    "exact_threshold": float(row["exact_threshold"]),
                }
            )
    return rows


def load_fragment_points(points_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(points_csv)
    if "note_token" in df.columns:
        df = df.rename(columns={"note_token": "display_note_token"})
    keep = ["frame_index", "time_sec", "display_note_token", "x12", "y12", "z_time", "component_type", "marker_size", "relative_amp", "phase"]
    cols = [c for c in keep if c in df.columns]
    return df[cols].copy()


def build_probe_points(
    detector_rows: list[dict[str, object]],
    matrix_csv: Path,
    start_frame: int,
    stop_frame: int,
) -> pd.DataFrame:
    out_rows: list[dict[str, object]] = []
    for row in detector_rows:
        token = str(row["exact_probe_band_token"])
        xy = token_to_spiral_xy(token)
        if xy is None:
            continue
        x12, y12 = xy
        probe_values = load_probe_row(matrix_csv, int(row["exact_probe_index"]))
        max_window_amp = 0.0
        for frame in range(start_frame, stop_frame + 1):
            max_window_amp = max(max_window_amp, float(probe_values.get(f"frame_{frame}", 0.0)))
        max_window_amp = max(max_window_amp, 1e-9)
        for frame in range(start_frame, stop_frame + 1):
            amp = float(probe_values.get(f"frame_{frame}", 0.0))
            rel = amp / max_window_amp
            run_start = int(row["exact_best_run_start_frame"])
            run_end = int(row["exact_best_run_end_frame"])
            is_best_run = run_start > 0 and run_start <= frame <= run_end
            is_first_hit = int(row["exact_first_hit_frame"]) == frame
            component_type = "direct_probe_band"
            if is_best_run:
                component_type = "direct_probe_band_best_run"
            elif is_first_hit:
                component_type = "direct_probe_band_first_hit"
            out_rows.append(
                {
                    "frame_index": frame,
                    "time_sec": frame / 60.0,
                    "display_note_token": token,
                    "x12": x12,
                    "y12": y12,
                    "z_time": frame / 60.0,
                    "component_type": component_type,
                    "marker_size": max(8.0, rel * 44.0),
                    "relative_amp": rel,
                    "phase": row["track_name"],
                    "probe_label": row["label"],
                    "probe_amp": amp,
                    "probe_threshold": float(row["exact_threshold"]),
                    "probe_hz": float(row["exact_probe_hz"]),
                }
            )
    return pd.DataFrame(out_rows)


def save_png(title: str, points: pd.DataFrame, out_png: Path) -> None:
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    style = [
        ("residual_resonance", "residual resonance", "#9a9a9a", 0.16),
        ("note_box", "note box", "#2a6fdb", 0.50),
        ("note_candidate", "presumed note", "#d62839", 0.78),
        ("direct_probe_band", "direct probe band", "#111111", 0.18),
        ("direct_probe_band_first_hit", "first direct hit", "#f77f00", 0.95),
        ("direct_probe_band_best_run", "direct best run", "#0a9396", 0.95),
    ]
    for comp, label, color, alpha in style:
        sub = points[points["component_type"] == comp]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub["x12"],
            sub["y12"],
            sub["z_time"],
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
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def save_html(title: str, points: pd.DataFrame, out_html: Path) -> None:
    styles = {
        "note_candidate": ("presumed note", "#d62839", 0.78),
        "note_box": ("note box", "#2a6fdb", 0.50),
        "residual_resonance": ("residual resonance", "#9a9a9a", 0.16),
        "direct_probe_band": ("direct probe band", "#111111", 0.18),
        "direct_probe_band_first_hit": ("first direct hit", "#f77f00", 0.95),
        "direct_probe_band_best_run": ("direct best run", "#0a9396", 0.95),
    }
    traces: list[dict[str, object]] = []
    for comp in [
        "residual_resonance",
        "note_box",
        "note_candidate",
        "direct_probe_band",
        "direct_probe_band_first_hit",
        "direct_probe_band_best_run",
    ]:
        sub = points[points["component_type"] == comp]
        if len(sub) == 0:
            continue
        label, color, opacity = styles[comp]
        hover = []
        for _, r in sub.iterrows():
            extra = ""
            if "probe_label" in r and isinstance(r.get("probe_label"), str):
                extra = (
                    f"<br>probe={r.get('probe_label','')}"
                    f"<br>amp={float(r.get('probe_amp', 0.0)):.6f}"
                    f"<br>thr={float(r.get('probe_threshold', 0.0)):.6f}"
                )
            hover.append(
                f"token={r['display_note_token']}<br>"
                f"time={float(r['time_sec']):.4f}<br>"
                f"type={comp}{extra}<br>"
                f"size={float(r['marker_size']):.2f}<br>"
                f"phase={r.get('phase','')}"
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
                    "size": sub["marker_size"].astype(float).tolist(),
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
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
<h2>{title}</h2>
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
    out_html.write_text(html, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build an interactive 3D view of the direct sustained entrant over the shared Ave Maria field."
    )
    ap.add_argument("--fragment-points-csv", default=str(REPORTS / "90_spiral3d_fragments" / "ave_maria_fragment_11p95s_12p25s__spiral3d_points_v1.csv"))
    ap.add_argument("--detector-csv", default=str(REPORTS / "ave_maria_11p95s_12p25s_direct_sustained_entrant_fundamental_v1.csv"))
    ap.add_argument("--probe-matrix-csv", default=str(REPORTS / "ave_maria_probe_matrix_micro_full.csv"))
    ap.add_argument("--start-frame", type=int, default=717)
    ap.add_argument("--stop-frame", type=int, default=735)
    ap.add_argument("--out-dir", default=str(REPORTS / "90_spiral3d_fragments"))
    ap.add_argument("--out-prefix", default="ave_maria_direct_entrant_fragment_11p95s_12p25s")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    fragment_points = load_fragment_points(Path(args.fragment_points_csv))
    detector_rows = load_direct_targets(Path(args.detector_csv))
    probe_points = build_probe_points(
        detector_rows=detector_rows,
        matrix_csv=Path(args.probe_matrix_csv),
        start_frame=int(args.start_frame),
        stop_frame=int(args.stop_frame),
    )
    all_points = pd.concat([fragment_points, probe_points], ignore_index=True, sort=False)
    all_points = all_points.sort_values(["time_sec", "component_type", "display_note_token"]).reset_index(drop=True)

    title = "Ave Maria direct entrant 3D fragment 11.95s -> 12.25s"
    out_csv = out_dir / f"{args.out_prefix}__spiral3d_points_v1.csv"
    out_png = out_dir / f"{args.out_prefix}__spiral3d.png"
    out_html = out_dir / f"{args.out_prefix}__spiral3d.html"
    out_summary = out_dir / f"{args.out_prefix}__summary.txt"
    out_meta = out_dir / f"{args.out_prefix}__meta_v1.json"

    all_points.to_csv(out_csv, index=False)
    save_png(title, all_points, out_png)
    save_html(title, all_points, out_html)

    direct_rows = [r for r in detector_rows if r["track_name"] in {"Cello", "Violin"}]
    summary_lines = [
        "AVE MARIA DIRECT ENTRANT SPIRAL 3D",
        "=" * 72,
        f"window_frames60: {int(args.start_frame)} -> {int(args.stop_frame)}",
        "",
        "direct_targets:",
    ]
    for row in detector_rows:
        summary_lines.append(
            f"  {row['label']}: first_hit={row['exact_first_hit_frame']} best_run={row['exact_best_run_start_frame']}->{row['exact_best_run_end_frame']}"
        )
    summary_lines.extend(
        [
            "",
            "reading_hint:",
            "  - orange points mark first direct probe-band hit",
            "  - teal points mark stable best-run direct probe support",
            "  - black points show weaker probe-band trajectory before stable support",
            "  - red/blue/grey points show the shared note/box/residual field",
            "",
            "main_interpretation:",
            "  - if orange/teal direct points appear before the field fully stabilizes, the entrant exists at raw probe-band level before later shared collapse",
        ]
    )
    out_summary.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    meta = {
        "title": title,
        "window_frames60": [int(args.start_frame), int(args.stop_frame)],
        "direct_targets": direct_rows,
        "out_csv": str(out_csv),
        "out_png": str(out_png),
        "out_html": str(out_html),
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("AVE MARIA DIRECT ENTRANT SPIRAL 3D DONE")
    print(f"out_csv     : {out_csv}")
    print(f"out_png     : {out_png}")
    print(f"out_html    : {out_html}")
    print(f"out_summary : {out_summary}")


if __name__ == "__main__":
    main()
