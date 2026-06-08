# -*- coding: ascii -*-
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(r"E:\Duodecimal_resonant_numeration")
THEORY_CSV = PROJECT_ROOT / "docs" / "theoretical_harmonics_9A_corrected.csv"
OBSERVED_CSV = (
    PROJECT_ROOT
    / "Block004_data"
    / "violin2"
    / "55_harmonic_chain_spiral3d"
    / "015_9.A-_violin2_4string__harmonic_chain_spiral3d_points.csv"
)
OUT_DIR = (
    PROJECT_ROOT
    / "Block004_data"
    / "violin2"
    / "55_harmonic_chain_spiral3d"
)
STEM = "015_9.A-_violin2_4string__theory_hchain_bridge"


def read_theory() -> dict[int, dict[str, float | str]]:
    rows: dict[int, dict[str, float | str]] = {}
    with THEORY_CSV.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            h = int(row["harmonic_order"])
            rows[h] = {
                "token": row["token"],
                "micro_shift": float(row["micro_shift"]),
                "relative_step": float(row["relative_step"]),
                "radius": float(row["radius"]),
                "angle_deg": float(row["angle_deg"]),
            }
    return rows


def read_observed_chain() -> tuple[dict[int, dict[str, float | str]], list[dict[str, float | str]]]:
    grouped: dict[int, list[dict[str, float | str]]] = defaultdict(list)
    chain_rows: list[dict[str, float | str]] = []
    with OBSERVED_CSV.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("is_chain") != "1":
                continue
            h = int(float(row["harmonic_index"]))
            grouped[h].append(row)
            chain_rows.append(row)

    means: dict[int, dict[str, float | str]] = {}
    for h, rows in grouped.items():
        mean_x = sum(float(r["x12"]) for r in rows) / len(rows)
        mean_y = sum(float(r["y12"]) for r in rows) / len(rows)
        mean_z = sum(float(r["z_time"]) for r in rows) / len(rows)
        mean_hz = sum(float(r["hz"]) for r in rows) / len(rows)
        mean_amp = sum(float(r["amplitude"]) for r in rows) / len(rows)
        mean_r = sum(math.hypot(float(r["x12"]), float(r["y12"])) for r in rows) / len(rows)
        means[h] = {
            "obs_token": str(rows[0]["note_token"]),
            "mean_x": mean_x,
            "mean_y": mean_y,
            "mean_z": mean_z,
            "mean_hz": mean_hz,
            "mean_amp": mean_amp,
            "mean_radius": mean_r,
            "frames": len(rows),
        }
    return means, chain_rows


def norm_angle(deg: float) -> float:
    return deg % 360.0


def polar_xy(radius: float, angle_deg: float) -> tuple[float, float]:
    rad = math.radians(angle_deg)
    return radius * math.cos(rad), radius * math.sin(rad)


def sample_segment(
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    samples: int,
) -> list[tuple[float, float, float]]:
    out: list[tuple[float, float, float]] = []
    for i in range(samples):
        t = i / (samples - 1) if samples > 1 else 0.0
        out.append(
            (
                p0[0] + (p1[0] - p0[0]) * t,
                p0[1] + (p1[1] - p0[1]) * t,
                p0[2] + (p1[2] - p0[2]) * t,
            )
        )
    return out


def build_bridge() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    theory = read_theory()
    observed_means, chain_rows = read_observed_chain()

    shared = sorted(set(theory) & set(observed_means))
    if not shared:
        raise SystemExit("No shared harmonic orders between theory and observed chain.")

    bridge_rows: list[dict[str, float | str]] = []

    theory_trace_x: list[float] = []
    theory_trace_y: list[float] = []
    theory_trace_z: list[float] = []
    theory_trace_text: list[str] = []
    theory_trace_size: list[float] = []

    floor_trace_x: list[float] = []
    floor_trace_y: list[float] = []
    floor_trace_z: list[float] = []
    floor_trace_text: list[str] = []
    floor_trace_size: list[float] = []

    lift_trace_x: list[float] = []
    lift_trace_y: list[float] = []
    lift_trace_z: list[float] = []
    lift_trace_text: list[str] = []
    lift_trace_size: list[float] = []

    theory_to_floor_x: list[float] = []
    theory_to_floor_y: list[float] = []
    theory_to_floor_z: list[float] = []
    floor_to_lift_x: list[float] = []
    floor_to_lift_y: list[float] = []
    floor_to_lift_z: list[float] = []

    summary_rows: list[dict[str, float | str]] = []

    for h in shared:
        trow = theory[h]
        orow = observed_means[h]
        theory_x, theory_y = polar_xy(float(orow["mean_radius"]), norm_angle(float(trow["angle_deg"])))
        floor_x = float(orow["mean_x"])
        floor_y = float(orow["mean_y"])
        lift_x = floor_x
        lift_y = floor_y
        lift_z = float(orow["mean_z"])

        theory_trace_x.append(theory_x)
        theory_trace_y.append(theory_y)
        theory_trace_z.append(0.0)
        theory_trace_text.append(
            f"h{h}<br>theory={trow['token']}<br>micro={float(trow['micro_shift']):+.3f}<br>scaled_r={float(orow['mean_radius']):.3f}"
        )
        theory_trace_size.append(10.0)

        floor_trace_x.append(floor_x)
        floor_trace_y.append(floor_y)
        floor_trace_z.append(0.0)
        floor_trace_text.append(
            f"h{h}<br>observed_mean_xy<br>token={orow['obs_token']}<br>frames={int(orow['frames'])}"
        )
        floor_trace_size.append(8.0)

        lift_trace_x.append(lift_x)
        lift_trace_y.append(lift_y)
        lift_trace_z.append(lift_z)
        lift_trace_text.append(
            f"h{h}<br>observed_mean_3d<br>token={orow['obs_token']}<br>z={lift_z:.3f}"
        )
        lift_trace_size.append(8.0)

        for x, y, z in sample_segment((theory_x, theory_y, 0.0), (floor_x, floor_y, 0.0), 16):
            theory_to_floor_x.append(x)
            theory_to_floor_y.append(y)
            theory_to_floor_z.append(z)

        for x, y, z in sample_segment((floor_x, floor_y, 0.0), (lift_x, lift_y, lift_z), 16):
            floor_to_lift_x.append(x)
            floor_to_lift_y.append(y)
            floor_to_lift_z.append(z)

        delta_xy = math.hypot(floor_x - theory_x, floor_y - theory_y)
        summary_rows.append(
            {
                "harmonic_order": h,
                "theory_token": trow["token"],
                "observed_token": orow["obs_token"],
                "theory_angle_deg_mod": norm_angle(float(trow["angle_deg"])),
                "observed_xy_angle_deg_mod": norm_angle(math.degrees(math.atan2(floor_y, floor_x))),
                "delta_xy": delta_xy,
                "observed_mean_radius": orow["mean_radius"],
                "observed_mean_z": orow["mean_z"],
                "observed_frames": orow["frames"],
            }
        )

        bridge_rows.extend(
            [
                {
                    "harmonic_order": h,
                    "dataset_role": "theory_scaled_2d",
                    "token": trow["token"],
                    "x12": theory_x,
                    "y12": theory_y,
                    "z_time": 0.0,
                    "marker_size": 10.0,
                },
                {
                    "harmonic_order": h,
                    "dataset_role": "observed_mean_floor",
                    "token": orow["obs_token"],
                    "x12": floor_x,
                    "y12": floor_y,
                    "z_time": 0.0,
                    "marker_size": 8.0,
                },
                {
                    "harmonic_order": h,
                    "dataset_role": "observed_mean_3d",
                    "token": orow["obs_token"],
                    "x12": lift_x,
                    "y12": lift_y,
                    "z_time": lift_z,
                    "marker_size": 8.0,
                },
            ]
        )

    chain_x = [float(r["x12"]) for r in chain_rows]
    chain_y = [float(r["y12"]) for r in chain_rows]
    chain_z = [float(r["z_time"]) for r in chain_rows]
    chain_text = [
        f"h{int(float(r['harmonic_index']))}<br>{r['note_token']}<br>time={float(r['z_time']):.4f}<br>amp={float(r['amplitude']):.4f}"
        for r in chain_rows
    ]
    chain_size = [4.0 + min(8.0, math.sqrt(max(float(r["relative_amp"]), 0.0)) * 24.0) for r in chain_rows]

    traces = [
        {
            "type": "scatter3d",
            "mode": "lines+markers",
            "name": "theory 2D scaled",
            "x": theory_trace_x,
            "y": theory_trace_y,
            "z": theory_trace_z,
            "text": theory_trace_text,
            "hoverinfo": "text",
            "line": {"color": "#7f3fbf", "width": 6},
            "marker": {"size": theory_trace_size, "color": "#7f3fbf", "opacity": 0.98},
        },
        {
            "type": "scatter3d",
            "mode": "markers",
            "name": "bridge theory to observed floor",
            "x": theory_to_floor_x,
            "y": theory_to_floor_y,
            "z": theory_to_floor_z,
            "hoverinfo": "skip",
            "marker": {"size": 2.8, "color": "#c7a0ff", "opacity": 0.55},
        },
        {
            "type": "scatter3d",
            "mode": "lines+markers",
            "name": "observed harmonic means floor",
            "x": floor_trace_x,
            "y": floor_trace_y,
            "z": floor_trace_z,
            "text": floor_trace_text,
            "hoverinfo": "text",
            "line": {"color": "#0c98d6", "width": 5},
            "marker": {"size": floor_trace_size, "color": "#0c98d6", "opacity": 0.96},
        },
        {
            "type": "scatter3d",
            "mode": "markers",
            "name": "bridge floor to lifted means",
            "x": floor_to_lift_x,
            "y": floor_to_lift_y,
            "z": floor_to_lift_z,
            "hoverinfo": "skip",
            "marker": {"size": 2.8, "color": "#ffb347", "opacity": 0.55},
        },
        {
            "type": "scatter3d",
            "mode": "lines+markers",
            "name": "observed harmonic means lifted",
            "x": lift_trace_x,
            "y": lift_trace_y,
            "z": lift_trace_z,
            "text": lift_trace_text,
            "hoverinfo": "text",
            "line": {"color": "#1f9d55", "width": 5},
            "marker": {"size": lift_trace_size, "color": "#1f9d55", "opacity": 0.96},
        },
        {
            "type": "scatter3d",
            "mode": "markers",
            "name": "violin2 harmonic chain 3D",
            "x": chain_x,
            "y": chain_y,
            "z": chain_z,
            "text": chain_text,
            "hoverinfo": "text",
            "marker": {"size": chain_size, "color": "#d62839", "opacity": 0.22},
        },
    ]

    html_path = OUT_DIR / f"{STEM}.html"
    title = "Theory to Harmonic Chain Bridge: violin2 015 9.A- 4th string"
    subtitle = (
        "Purple = corrected theory scaled to observed harmonic radii; "
        "blue = observed mean XY anchors; green = lifted observed means; "
        "red cloud = full harmonic chain in time."
    )
    buttons = [
        {
            "label": "All",
            "method": "update",
            "args": [{"visible": [True, True, True, True, True, True]}],
        },
        {
            "label": "Theory Only",
            "method": "update",
            "args": [{"visible": [True, False, False, False, False, False]}],
        },
        {
            "label": "Mean Bridge",
            "method": "update",
            "args": [{"visible": [True, True, True, True, True, False]}],
        },
        {
            "label": "3D Chain",
            "method": "update",
            "args": [{"visible": [False, False, False, False, False, True]}],
        },
    ]
    layout = {
        "scene": {
            "xaxis": {"title": "x12"},
            "yaxis": {"title": "y12"},
            "zaxis": {"title": "time_sec"},
        },
        "margin": {"l": 0, "r": 0, "b": 0, "t": 40},
        "legend": {"orientation": "h"},
        "updatemenus": [
            {
                "type": "buttons",
                "direction": "right",
                "x": 0.02,
                "y": 1.08,
                "showactive": True,
                "buttons": buttons,
            }
        ],
    }
    html_text = (
        "<!doctype html>\n"
        "<html>\n<head>\n<meta charset=\"utf-8\">\n"
        f"<title>{title}</title>\n"
        "<script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>\n"
        "</head>\n<body>\n"
        f"<h2>{title}</h2>\n"
        f"<p>{subtitle}</p>\n"
        "<div id=\"plot\" style=\"width:100%;height:920px;\"></div>\n"
        "<script>\n"
        f"const traces = {json.dumps(traces, ensure_ascii=False)};\n"
        f"const layout = {json.dumps(layout, ensure_ascii=False)};\n"
        "Plotly.newPlot(\"plot\", traces, layout);\n"
        "</script>\n</body>\n</html>\n"
    )
    html_path.write_text(html_text, encoding="utf-8")

    points_csv = OUT_DIR / f"{STEM}_points.csv"
    with points_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        fieldnames = ["harmonic_order", "dataset_role", "token", "x12", "y12", "z_time", "marker_size"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(bridge_rows)

    compare_csv = OUT_DIR / f"{STEM}_harmonic_compare.csv"
    with compare_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        fieldnames = list(summary_rows[0].keys())
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_txt = OUT_DIR / f"{STEM}_summary.txt"
    summary_lines = [
        "THEORY TO HARMONIC CHAIN BRIDGE",
        "=" * 72,
        f"theory_csv                : {THEORY_CSV}",
        f"observed_chain_csv        : {OBSERVED_CSV}",
        f"out_html                  : {html_path}",
        f"out_points_csv            : {points_csv}",
        f"out_harmonic_compare_csv  : {compare_csv}",
        f"shared_harmonics          : {len(shared)}",
        "",
        "bridge_logic:",
        "  1. Theoretical harmonics stay on the corrected 12-tone resonance geometry.",
        "  2. Their radii are scaled to the observed mean radii of violin2 h1..h12.",
        "  3. Observed mean XY anchors show how real harmonic centers drift from theory.",
        "  4. Full 3D chain shows the time-life of the same harmonic family.",
        "",
        "harmonic_summary:",
    ]
    for row in summary_rows:
        summary_lines.append(
            f"  h{row['harmonic_order']}: theory={row['theory_token']} -> observed={row['observed_token']}; "
            f"delta_xy={float(row['delta_xy']):.4f}; frames={int(row['observed_frames'])}"
        )
    summary_txt.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(html_path)
    print(points_csv)
    print(compare_csv)
    print(summary_txt)


if __name__ == "__main__":
    build_bridge()
