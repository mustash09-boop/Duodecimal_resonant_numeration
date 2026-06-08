from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(r"E:\Duodecimal_resonant_numeration")
POINTS_CSV = PROJECT_ROOT / "Block004_data" / "violin2" / "50_spiral3d" / "015_9.A-_violin2_4string__spiral3d_points.csv"
OUT_HTML = PROJECT_ROOT / "Block004_data" / "violin2" / "50_spiral3d" / "015_9.A-_violin2_4string__spiral3d_motion_presentation.html"
OUT_SUMMARY = PROJECT_ROOT / "Block004_data" / "violin2" / "50_spiral3d" / "015_9.A-_violin2_4string__spiral3d_motion_presentation_summary.txt"


def build_trace(df: pd.DataFrame, *, name: str, color: str, opacity: float, size_base: float, size_mul: float) -> dict:
    hover = []
    sizes = []
    for _, row in df.iterrows():
        rel_amp = float(row.get("relative_amp", 0.0))
        sizes.append(max(size_base, size_base + rel_amp * size_mul))
        hover.append(
            f"type={row.get('component_type','')}<br>"
            f"token={row.get('note_token','')}<br>"
            f"hz={float(row.get('hz', 0.0)):.2f}<br>"
            f"time={float(row.get('time_sec', 0.0)):.4f}<br>"
            f"amp={float(row.get('amplitude', 0.0)):.6f}<br>"
            f"h={row.get('harmonic_index','')}"
        )
    return {
        "type": "scatter3d",
        "mode": "markers",
        "name": name,
        "x": df["x12"].astype(float).tolist(),
        "y": df["y12"].astype(float).tolist(),
        "z": df["z_time"].astype(float).tolist(),
        "text": hover,
        "hoverinfo": "text",
        "marker": {
            "size": sizes,
            "opacity": opacity,
            "color": color,
        },
    }


def main() -> None:
    df = pd.read_csv(POINTS_CSV)

    chain = df[df["component_type"].astype(str) == "chain"].copy()
    note_box = df[df["component_type"].astype(str) == "note_box"].copy()
    secondary = df[df["component_type"].astype(str) == "dense_other"].copy()

    traces = [
        build_trace(
            chain,
            name="violin2 / chain core",
            color="#22c55e",
            opacity=0.84,
            size_base=3.8,
            size_mul=20.0,
        ),
        build_trace(
            note_box,
            name="violin2 / note_box body",
            color="#f59e0b",
            opacity=0.72,
            size_base=3.2,
            size_mul=12.0,
        ),
        build_trace(
            secondary,
            name="violin2 / secondary harmonics",
            color="#8b5cf6",
            opacity=0.58,
            size_base=2.9,
            size_mul=8.0,
        ),
    ]

    payload = json.dumps(traces, ensure_ascii=False)
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>015_9.A-_violin2_4string spiral3d motion presentation</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
<h2>3D 12-spiral over time: 015_9.A-_violin2_4string</h2>
<p>Presentation split for Motion: chain core, note box body, secondary harmonics.</p>
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
    OUT_HTML.write_text(html, encoding="utf-8")

    lines = [
        "VIOLIN2 9.A- SPIRAL3D MOTION PRESENTATION",
        "=" * 80,
        f"source_csv             : {POINTS_CSV}",
        f"out_html               : {OUT_HTML}",
        "",
        f"chain_points           : {len(chain)}",
        f"note_box_points        : {len(note_box)}",
        f"secondary_points       : {len(secondary)}",
        "",
        "trace 1: violin2 / chain core",
        "trace 2: violin2 / note_box body",
        "trace 3: violin2 / secondary harmonics",
    ]
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(OUT_HTML)
    print(OUT_SUMMARY)


if __name__ == "__main__":
    main()
