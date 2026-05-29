# -*- coding: ascii -*-
"""
MULTI INSTRUMENT HARMONIC AMPLITUDE COMPARE

Compare the same canonical note across instruments by harmonic amplitude
behavior, not only by token presence.

Outputs:
- raw chain points CSV
- per-harmonic summary CSV
- 2D PNG (mean relative amplitude vs harmonic index)
- 3D HTML / PNG (harmonic index, time, relative amplitude)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_note(note: str) -> str:
    return str(note or "").strip().replace("'", "")


def safe_name(s: str) -> str:
    return (
        str(s)
        .replace("'", "")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def load_points_for_note(index_csv: Path, note: str, instruments: list[str]) -> pd.DataFrame:
    target = normalize_note(note)
    idx = pd.read_csv(index_csv)
    idx["canonical_norm"] = idx["canonical_note12"].astype(str).apply(normalize_note)
    sub = idx[idx["canonical_norm"] == target]
    if instruments:
        sub = sub[sub["instrument"].isin(set(instruments))]

    rows = []
    for _, r in sub.iterrows():
        csv_path = Path(str(r["spiral3d_csv"]))
        if not csv_path.exists():
            continue
        pts = pd.read_csv(csv_path)
        if len(pts) == 0:
            continue
        pts = pts[pts["component_type"] == "chain"].copy()
        if len(pts) == 0:
            continue
        pts["instrument"] = str(r["instrument"])
        pts["canonical_note12"] = str(r["canonical_note12"])
        pts["source_note_name"] = str(r.get("source_note_name", ""))
        rows.append(pts)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["harmonic_index"] = pd.to_numeric(out["harmonic_index"], errors="coerce")
    out = out.dropna(subset=["harmonic_index"])
    out["harmonic_index"] = out["harmonic_index"].astype(int)
    rel_times = []
    for (_, source_name), g in out.groupby(["instrument", "source_note_name"]):
        t0 = float(g["time_sec"].min())
        t1 = float(g["time_sec"].max())
        span = max(1e-9, t1 - t0)
        rel = (g["time_sec"] - t0) / span
        rel_times.append(rel)
    if rel_times:
        out["relative_time01"] = pd.concat(rel_times).sort_index()
    else:
        out["relative_time01"] = 0.0
    bins = []
    for x in out["relative_time01"].tolist():
        if x <= 0.2:
            bins.append("attack")
        elif x <= 0.6:
            bins.append("sustain")
        else:
            bins.append("late")
    out["phase_bin"] = bins
    return out


def save_profile_png(summary_df: pd.DataFrame, note: str, out_png: Path) -> None:
    plt.figure(figsize=(10, 6))
    attack_df = summary_df[summary_df["phase_bin"] == "attack"].copy()
    for instrument, g in attack_df.groupby("instrument"):
        g = g.sort_values("harmonic_index")
        plt.plot(
            g["harmonic_index"],
            g["mean_relative_amp"],
            marker="o",
            linewidth=2,
            label=instrument,
        )
    plt.title(f"Harmonic amplitude profile (attack): {note}")
    plt.xlabel("harmonic_index")
    plt.ylabel("mean_relative_amp")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def save_harmonic_3d_png(points_df: pd.DataFrame, note: str, out_png: Path) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = {
        "RealPiano_1_1": "#d62839",
        "cello": "#1d4ed8",
        "violin": "#2a9d8f",
    }
    for instrument, g in points_df.groupby("instrument"):
        ax.scatter(
            g["harmonic_index"],
            g["relative_time01"],
            g["relative_amp"],
            s=[max(10.0, float(x) * 55.0) for x in g["relative_amp"]],
            alpha=0.78,
            c=colors.get(instrument, "#444444"),
            label=instrument,
            edgecolors="none",
        )

    ax.set_title(f"Harmonic amplitude 3D: {note}")
    ax.set_xlabel("harmonic_index")
    ax.set_ylabel("relative_time01")
    ax.set_zlabel("relative_amp")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def save_harmonic_3d_html(points_df: pd.DataFrame, note: str, out_html: Path) -> None:
    colors = {
        "RealPiano_1_1": "#d62839",
        "cello": "#1d4ed8",
        "violin": "#2a9d8f",
    }
    traces = []
    for instrument, g in points_df.groupby("instrument"):
        hover = []
        for _, r in g.iterrows():
            hover.append(
                f"instrument={instrument}<br>"
                f"note={note}<br>"
                f"harmonic={int(r['harmonic_index'])}<br>"
                f"relative_time01={float(r['relative_time01']):.4f}<br>"
                f"phase={r.get('phase_bin','')}<br>"
                f"relative_amp={float(r['relative_amp']):.6f}<br>"
                f"token={r.get('note_token','')}<br>"
                f"hz={float(r.get('hz', 0.0)):.2f}"
            )
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "name": instrument,
                "x": g["harmonic_index"].astype(int).tolist(),
                "y": g["relative_time01"].astype(float).tolist(),
                "z": g["relative_amp"].astype(float).tolist(),
                "text": hover,
                "hoverinfo": "text",
                "marker": {
                    "size": [max(3.0, float(x) * 12.0) for x in g["relative_amp"]],
                    "opacity": 0.8,
                    "color": colors.get(instrument, "#444444"),
                },
            }
        )

    payload = json.dumps(traces, ensure_ascii=False)
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Harmonic amplitude 3D: {note}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
<h2>Harmonic amplitude 3D: {note}</h2>
<p>x = harmonic index, y = normalized note time, z = relative harmonic amplitude</p>
<div id="plot" style="width:100%;height:900px;"></div>
<script>
const traces = {payload};
const layout = {{
  scene: {{
    xaxis: {{title: "harmonic_index"}},
    yaxis: {{title: "relative_time01"}},
    zaxis: {{title: "relative_amp"}}
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--note_index_csv", required=True)
    ap.add_argument("--note", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--instruments", nargs="*", default=["RealPiano_1_1", "cello", "violin"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    note = normalize_note(args.note)
    points = load_points_for_note(Path(args.note_index_csv), note, args.instruments)
    if len(points) == 0:
        raise RuntimeError(f"No chain points found for note {note}")

    summary = (
        points.groupby(["instrument", "phase_bin", "harmonic_index"], as_index=False)
        .agg(
            point_count=("relative_amp", "count"),
            mean_relative_amp=("relative_amp", "mean"),
            median_relative_amp=("relative_amp", "median"),
            max_relative_amp=("relative_amp", "max"),
            mean_time=("relative_time01", "mean"),
            min_time=("relative_time01", "min"),
            max_time=("relative_time01", "max"),
            mean_hz=("hz", "mean"),
        )
        .sort_values(["phase_bin", "harmonic_index", "instrument"])
    )

    base = out_dir / f"harmonic_amplitude_compare__{safe_name(note)}"
    raw_csv = Path(str(base) + "__raw_points.csv")
    summary_csv = Path(str(base) + "__summary.csv")
    png_2d = Path(str(base) + "__profile.png")
    png_3d = Path(str(base) + "__3d.png")
    html_3d = Path(str(base) + "__3d.html")
    md_summary = Path(str(base) + "__summary.md")

    points.to_csv(raw_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    save_profile_png(summary, note, png_2d)
    save_harmonic_3d_png(points, note, png_3d)
    save_harmonic_3d_html(points, note, html_3d)

    top = (
        summary[summary["phase_bin"] == "attack"]
        .sort_values(["harmonic_index", "mean_relative_amp"], ascending=[True, False])
        .groupby("harmonic_index")
        .first()
        .reset_index()
    )
    lines = [
        f"# Harmonic amplitude comparison: {note}",
        "",
        "This comparison focuses on chain points only.",
        "",
        "## Instrument set",
        "",
        *[f"- `{x}`" for x in sorted(points['instrument'].unique())],
        "",
        "## Files",
        "",
        f"- raw points: `{raw_csv.name}`",
        f"- summary csv: `{summary_csv.name}`",
        f"- 2D profile png: `{png_2d.name}`",
        f"- 3D amplitude png: `{png_3d.name}`",
        f"- 3D amplitude html: `{html_3d.name}`",
        "",
        "## Strongest instrument by harmonic index in attack phase",
        "",
    ]
    for _, row in top.iterrows():
        lines.append(
            f"- harmonic `{int(row['harmonic_index'])}`: `{row['instrument']}` "
            f"(mean_relative_amp={float(row['mean_relative_amp']):.4f})"
        )
    md_summary.write_text("\n".join(lines), encoding="utf-8")

    print("MULTI INSTRUMENT HARMONIC AMPLITUDE COMPARE DONE")
    print(summary_csv)
    print(png_2d)
    print(png_3d)
    print(html_3d)


if __name__ == "__main__":
    main()
