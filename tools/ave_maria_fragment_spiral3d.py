# -*- coding: ascii -*-
"""
AVE MARIA FRAGMENT SPIRAL 3D

Build a compact 3D spiral-time projection for a short ensemble fragment.

Layers:
- note_candidate      : from controlled sustain frames
- note_box            : field points with stronger anchor coupling
- residual_resonance  : field points with weaker anchor coupling

Special rule for this experiment:
- note point size depends on amplitude proxy
- note_box and residual_resonance keep fixed marker size
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import wave
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from music12.core.notation12 import bij12_to_int, parse_token, step_index0


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def coarse_token(token: str) -> str:
    tok = str(token or "").strip()
    return tok.split("'")[0]


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


def cut_audio_fragment(src_wav: Path, out_wav: Path, start_sec: float, stop_sec: float) -> None:
    with wave.open(str(src_wav), "rb") as r:
        fr = r.getframerate()
        s0 = int(start_sec * fr)
        s1 = int(stop_sec * fr)
        r.setpos(s0)
        data = r.readframes(max(0, s1 - s0))
        with wave.open(str(out_wav), "wb") as w:
            w.setnchannels(r.getnchannels())
            w.setsampwidth(r.getsampwidth())
            w.setframerate(fr)
            w.writeframes(data)


def load_family_energy_index(families_csv: Path, start_frame: int, stop_frame: int) -> tuple[dict, dict]:
    exact_idx: dict[tuple[int, str], float] = {}
    coarse_idx: dict[tuple[int, str], float] = {}
    usecols = [
        "frame_index",
        "family_root_note_micro",
        "family_root_note_coarse",
        "root_cluster_energy",
        "family_score",
    ]
    df = pd.read_csv(families_csv, usecols=usecols)
    df = df[(df["frame_index"] >= start_frame) & (df["frame_index"] < stop_frame)]
    for _, row in df.iterrows():
        frame = int(row["frame_index"])
        micro = str(row.get("family_root_note_micro", "") or "")
        coarse = str(row.get("family_root_note_coarse", "") or "")
        amp = float(row.get("root_cluster_energy", 0.0) or 0.0)
        if amp <= 0:
            amp = float(row.get("family_score", 0.0) or 0.0)
        if micro:
            exact_idx[(frame, micro)] = max(exact_idx.get((frame, micro), 0.0), amp)
        if coarse:
            coarse_idx[(frame, coarse)] = max(coarse_idx.get((frame, coarse), 0.0), amp)
    return exact_idx, coarse_idx


def build_note_points(
    sustain_csv: Path,
    exact_idx: dict,
    coarse_idx: dict,
    start_frame: int,
    stop_frame: int,
) -> pd.DataFrame:
    usecols = [
        "frame_index",
        "selected_note_token",
        "coarse_note",
        "phase",
        "selection_reason",
        "phase_score",
        "proto_exciter_id",
    ]
    df = pd.read_csv(sustain_csv, usecols=usecols)
    df = df[(df["frame_index"] >= start_frame) & (df["frame_index"] < stop_frame)]

    rows: list[dict] = []
    grouped: dict[tuple[int, str], dict] = {}
    for _, row in df.iterrows():
        frame = int(row["frame_index"])
        token = str(row["selected_note_token"] or "")
        if not token:
            continue
        amp = exact_idx.get((frame, token), 0.0)
        if amp <= 0:
            amp = coarse_idx.get((frame, str(row["coarse_note"] or coarse_token(token))), 0.0)
        if amp <= 0:
            amp = float(row.get("phase_score", 0.0) or 0.0)
        key = (frame, token)
        current = grouped.get(key)
        proto_id = int(row.get("proto_exciter_id", 0) or 0)
        phase = str(row.get("phase", "") or "")
        reason = str(row.get("selection_reason", "") or "")
        if current is None or amp > current["amplitude"]:
            grouped[key] = {
                "frame_index": frame,
                "time_sec": frame / 60.0,
                "note_token": token,
                "x12": None,
                "y12": None,
                "z_time": frame / 60.0,
                "amplitude": amp,
                "component_type": "note_candidate",
                "phase": phase,
                "selection_reason": reason,
                "proto_exciter_id": proto_id,
            }
    for item in grouped.values():
        xy = token_to_spiral_xy(item["note_token"])
        if xy is None:
            continue
        item["x12"], item["y12"] = xy
        rows.append(item)
    out = pd.DataFrame(rows)
    if len(out):
        max_amp = float(out["amplitude"].max()) or 1.0
        out["relative_amp"] = out["amplitude"] / max_amp
        out["marker_size"] = out["relative_amp"].apply(lambda x: max(8.0, float(x) * 42.0))
    else:
        out["relative_amp"] = []
        out["marker_size"] = []
    return out


def build_field_points(field_csv: Path, start_frame: int, stop_frame: int) -> pd.DataFrame:
    usecols = [
        "frame_index",
        "phase",
        "dominant_note_token",
        "anchor_note_token",
        "field_strength",
        "dominant_score",
        "field_diversity",
        "anchor_match_ratio",
        "selected_family_count",
    ]
    df = pd.read_csv(field_csv, usecols=usecols)
    df = df[(df["frame_index"] >= start_frame) & (df["frame_index"] < stop_frame)]

    grouped: dict[tuple[int, str, str], dict] = {}
    for _, row in df.iterrows():
        token = str(row.get("dominant_note_token", "") or "")
        if not token:
            continue
        frame = int(row["frame_index"])
        anchor_match = float(row.get("anchor_match_ratio", 0.0) or 0.0)
        component_type = "note_box" if anchor_match >= 0.5 else "residual_resonance"
        strength = float(row.get("field_strength", 0.0) or 0.0)
        key = (frame, token, component_type)
        current = grouped.get(key)
        if current is None or strength > current["field_strength"]:
            grouped[key] = {
                "frame_index": frame,
                "time_sec": frame / 60.0,
                "note_token": token,
                "x12": None,
                "y12": None,
                "z_time": frame / 60.0,
                "field_strength": strength,
                "relative_amp": 1.0,
                "component_type": component_type,
                "phase": str(row.get("phase", "") or ""),
                "anchor_note_token": str(row.get("anchor_note_token", "") or ""),
                "anchor_match_ratio": anchor_match,
                "field_diversity": int(row.get("field_diversity", 0) or 0),
                "selected_family_count": int(row.get("selected_family_count", 0) or 0),
                "marker_size": 24.0 if component_type == "note_box" else 10.0,
            }
    rows = []
    for item in grouped.values():
        xy = token_to_spiral_xy(item["note_token"])
        if xy is None:
            continue
        item["x12"], item["y12"] = xy
        rows.append(item)
    return pd.DataFrame(rows)


def save_png(title: str, points: pd.DataFrame, out_png: Path) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    style = [
        ("residual_resonance", "residual resonance", "#8f8f8f", 0.18),
        ("note_box", "note box", "#2a6fdb", 0.72),
        ("note_candidate", "presumed note", "#d62839", 0.92),
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
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def save_html(title: str, points: pd.DataFrame, out_html: Path) -> None:
    traces = []
    styles = {
        "note_candidate": ("presumed note", "#d62839", 0.92),
        "note_box": ("note box", "#2a6fdb", 0.72),
        "residual_resonance": ("residual resonance", "#8f8f8f", 0.22),
    }
    for comp in ["residual_resonance", "note_box", "note_candidate"]:
        sub = points[points["component_type"] == comp]
        if len(sub) == 0:
            continue
        label, color, opacity = styles[comp]
        hover = []
        for _, r in sub.iterrows():
            hover.append(
                f"token={r['note_token']}<br>"
                f"time={float(r['time_sec']):.4f}<br>"
                f"type={comp}<br>"
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


def subset_csv(input_csv: Path, out_csv: Path, start_sec: float, stop_sec: float) -> int:
    with input_csv.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if start_sec <= float(row["start_sec"]) < stop_sec]
        with out_csv.open("w", encoding="utf-8", newline="") as g:
            writer = csv.DictWriter(g, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return len(rows)


def role_summary(role_csv: Path, layered_csv: Path, start_frame: int, stop_frame: int) -> tuple[Counter, Counter]:
    role_counts: Counter = Counter()
    dominant_counts: Counter = Counter()
    with role_csv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            frame = int(row["birth_frame"])
            if start_frame <= frame < stop_frame:
                role_counts[row["role_pattern"]] += 1
    with layered_csv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            frame = int(row["birth_frame"])
            if start_frame <= frame < stop_frame:
                dominant_counts[row["dominant_instrument"]] += 1
    return role_counts, dominant_counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report_dir", required=True)
    ap.add_argument("--src_wav", required=True)
    ap.add_argument("--midi_events_csv", required=True)
    ap.add_argument("--start_sec", type=float, default=8.0)
    ap.add_argument("--duration_sec", type=float, default=1.0)
    ap.add_argument("--out_prefix", default="ave_maria_fragment_8p0s_9p0s")
    args = ap.parse_args()

    report_dir = Path(args.report_dir)
    out_dir = report_dir / "90_spiral3d_fragments"
    ensure_dir(out_dir)

    start_sec = float(args.start_sec)
    stop_sec = start_sec + float(args.duration_sec)
    start_frame = int(round(start_sec * 60.0))
    stop_frame = int(round(stop_sec * 60.0))

    sustain_csv = report_dir / "ave_maria_controlled_sustain_frames_v1.csv"
    families_csv = report_dir / "ave_maria_micro_families_v1.csv"
    field_csv = report_dir / "ave_maria_event_field_frames_v1.csv"
    role_csv = report_dir / "ave_maria_instrument_role_behavior_map_v1.csv"
    layered_csv = report_dir / "ave_maria_multi_instrument_layered_assignment_v1.csv"

    exact_idx, coarse_idx = load_family_energy_index(families_csv, start_frame, stop_frame)
    note_df = build_note_points(sustain_csv, exact_idx, coarse_idx, start_frame, stop_frame)
    field_df = build_field_points(field_csv, start_frame, stop_frame)

    keep_cols = [
        "frame_index",
        "time_sec",
        "note_token",
        "x12",
        "y12",
        "z_time",
        "component_type",
        "marker_size",
        "relative_amp",
        "phase",
    ]
    note_extra = ["amplitude", "selection_reason", "proto_exciter_id"]
    field_extra = ["field_strength", "anchor_note_token", "anchor_match_ratio", "field_diversity", "selected_family_count"]
    note_points = note_df[keep_cols + note_extra] if len(note_df) else pd.DataFrame(columns=keep_cols + note_extra)
    field_points = field_df[keep_cols + field_extra] if len(field_df) else pd.DataFrame(columns=keep_cols + field_extra)
    all_points = pd.concat([note_points, field_points], ignore_index=True, sort=False)
    all_points = all_points.sort_values(["time_sec", "component_type", "note_token"]).reset_index(drop=True)

    title = f"Ave Maria 3D spiral fragment {start_sec:.1f}s -> {stop_sec:.1f}s"
    out_csv = out_dir / f"{args.out_prefix}__spiral3d_points_v1.csv"
    out_png = out_dir / f"{args.out_prefix}__spiral3d.png"
    out_html = out_dir / f"{args.out_prefix}__spiral3d.html"
    out_wav = out_dir / f"{args.out_prefix}__audio.wav"
    out_midi_csv = out_dir / f"{args.out_prefix}__midi_events.csv"
    out_summary = out_dir / f"{args.out_prefix}__summary.txt"

    all_points.to_csv(out_csv, index=False)
    save_png(title, all_points, out_png)
    save_html(title, all_points, out_html)
    cut_audio_fragment(Path(args.src_wav), out_wav, start_sec, stop_sec)
    midi_rows = subset_csv(Path(args.midi_events_csv), out_midi_csv, start_sec, stop_sec)
    role_counts, dominant_counts = role_summary(role_csv, layered_csv, start_frame, stop_frame)

    component_counts = Counter(all_points["component_type"].tolist())
    top_note_tokens = Counter(note_df["note_token"].tolist()).most_common(10) if len(note_df) else []
    top_box_tokens = Counter(field_df[field_df["component_type"] == "note_box"]["note_token"].tolist()).most_common(10) if len(field_df) else []
    top_residual_tokens = Counter(field_df[field_df["component_type"] == "residual_resonance"]["note_token"].tolist()).most_common(10) if len(field_df) else []

    summary_lines = [
        "AVE MARIA FRAGMENT SPIRAL 3D",
        "=" * 72,
        f"window_sec: {start_sec:.3f} -> {stop_sec:.3f}",
        f"window_frames60: {start_frame} -> {stop_frame}",
        "",
        "point_counts:",
        *[f"  {k}: {v}" for k, v in component_counts.items()],
        "",
        f"midi_events_in_window: {midi_rows}",
        "dominant_instrument_counts:",
        *[f"  {k}: {v}" for k, v in dominant_counts.most_common()],
        "",
        "role_pattern_counts:",
        *[f"  {k}: {v}" for k, v in role_counts.most_common()],
        "",
        "top_note_tokens:",
        *[f"  {k}: {v}" for k, v in top_note_tokens],
        "",
        "top_note_box_tokens:",
        *[f"  {k}: {v}" for k, v in top_box_tokens],
        "",
        "top_residual_tokens:",
        *[f"  {k}: {v}" for k, v in top_residual_tokens],
        "",
        "assumptions:",
        "  note_candidate size follows amplitude proxy from micro family root energy",
        "  note_box size is fixed",
        "  residual_resonance size is fixed",
        "  note_box vs residual_resonance split uses anchor_match_ratio >= 0.5",
    ]
    out_summary.write_text("\n".join(summary_lines), encoding="utf-8")

    meta = {
        "title": title,
        "window_sec": [start_sec, stop_sec],
        "window_frames60": [start_frame, stop_frame],
        "component_counts": dict(component_counts),
        "midi_events_in_window": midi_rows,
        "dominant_instrument_counts": dict(dominant_counts),
        "role_pattern_counts": dict(role_counts),
        "out_csv": str(out_csv),
        "out_png": str(out_png),
        "out_html": str(out_html),
        "out_wav": str(out_wav),
        "out_midi_csv": str(out_midi_csv),
    }
    (out_dir / f"{args.out_prefix}__meta_v1.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("AVE MARIA FRAGMENT SPIRAL 3D DONE")
    print(f"out_dir     : {out_dir}")
    print(f"out_csv     : {out_csv}")
    print(f"out_png     : {out_png}")
    print(f"out_html    : {out_html}")
    print(f"out_summary : {out_summary}")


if __name__ == "__main__":
    main()
