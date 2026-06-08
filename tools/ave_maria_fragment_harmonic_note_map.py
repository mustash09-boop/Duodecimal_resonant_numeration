# -*- coding: ascii -*-
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(r"E:\Duodecimal_resonant_numeration")
AVE_ROOT = PROJECT_ROOT / "Block001_data" / "Ave_Maria"
REPORTS_ROOT = AVE_ROOT / "10_reports_Ave_Maria"
MIDI_CSV = AVE_ROOT / "00_sources" / "midi" / "ave_maria_gounod_midi_events_with_parts_v1.csv"

_DEGREE_TO_INDEX = {
    "1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5,
    "7": 6, "8": 7, "9": 8, "A": 9, "B": 10, "C": 11,
}
_TOKEN_RE = re.compile(r"^\s*([1-9A-C]+)\.([1-9A-C])(?:'(.*))?\s*$")
_HARMONIC_OFFSETS_12 = {
    2: 12,
    3: 19,
    4: 24,
    5: 28,
    6: 31,
    7: 34,
    8: 36,
}
_BLOCK4_WEIGHTS = {
    2: 0.10,
    3: 0.15,
    4: 0.08,
    5: 0.26,
    6: 0.10,
    7: 0.24,
    8: 0.07,
}


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _parse_token(token: str) -> tuple[int, int, str] | None:
    m = _TOKEN_RE.match(str(token or "").strip())
    if not m:
        return None
    octave_s, degree_s, micro = m.groups()
    if degree_s not in _DEGREE_TO_INDEX:
        return None
    octave_value = 0
    for ch in octave_s:
        if ch not in _DEGREE_TO_INDEX:
            return None
        octave_value = octave_value * 12 + (_DEGREE_TO_INDEX[ch] + 1)
    return octave_value, _DEGREE_TO_INDEX[degree_s], micro or "-"


def _pitch_index12(token: str) -> int | None:
    p = _parse_token(token)
    if not p:
        return None
    octave, degree, _micro = p
    return octave * 12 + degree


def _expected_harmonic_degrees(root_token: str) -> dict[int, int]:
    root_idx = _pitch_index12(root_token)
    if root_idx is None:
        return {}
    return {h: (root_idx + offset) % 12 for h, offset in _HARMONIC_OFFSETS_12.items()}


def _register_compensation(root_token: str, weighted_score: float, h57_score: float) -> tuple[float, str]:
    parsed = _parse_token(root_token)
    if not parsed:
        return 0.0, "unknown_register"
    octave, _degree, _micro = parsed
    if octave <= 7:
        return 0.18 * weighted_score + 0.12 * h57_score, "low_register_compensation"
    if octave >= 11:
        return 0.10 * weighted_score + 0.20 * h57_score, "high_register_compensation"
    return 0.12 * weighted_score + 0.16 * h57_score, "mid_register_compensation"


def load_fragment_tokens(points_csv: Path, start_frame: int, end_frame: int) -> dict[int, set[str]]:
    observed: dict[int, set[str]] = defaultdict(set)
    with points_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        token_col = "display_note_token" if "display_note_token" in reader.fieldnames else "note_token"
        for row in reader:
            frame = _safe_int(row.get("frame_index"), -1)
            if frame < start_frame or frame > end_frame:
                continue
            token = str(row.get(token_col, "")).strip()
            if token:
                observed[frame].add(token)
    return observed


def load_midi_unique_roots(midi_csv: Path, start_sec: float, end_sec: float) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    with midi_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            start = _safe_float(row.get("start_sec"), 0.0)
            end = _safe_float(row.get("end_sec"), 0.0)
            if start >= end_sec or end < start_sec:
                continue
            track = str(row.get("track_name", "")).strip()
            note = str(row.get("note12", "")).strip()
            if not track or not note:
                continue
            key = (track, note)
            if key not in grouped:
                grouped[key] = {
                    "track_name": track,
                    "note_token": note,
                    "freq_hz": _safe_float(row.get("freq_hz"), 0.0),
                    "first_start_sec": start,
                    "first_start_frame60": _safe_int(row.get("start_frame60"), 0),
                    "occurrences": 1,
                }
            else:
                grouped[key]["occurrences"] += 1
                grouped[key]["first_start_sec"] = min(grouped[key]["first_start_sec"], start)
                grouped[key]["first_start_frame60"] = min(grouped[key]["first_start_frame60"], _safe_int(row.get("start_frame60"), 0))
    rows = list(grouped.values())
    rows.sort(key=lambda r: (r["first_start_sec"], r["track_name"], r["freq_hz"]))
    for i, row in enumerate(rows, start=1):
        row["label"] = f"{row['track_name']}::{row['note_token']}"
        row["row_order"] = i
    return rows


def support_for_root(root_token: str, observed_tokens: set[str]) -> dict[str, Any]:
    expected = _expected_harmonic_degrees(root_token)
    if not expected:
        return {
            "equal_support_score": 0.0,
            "weighted_support_score": 0.0,
            "harmonic_5_7_score": 0.0,
            "block4_register_score": 0.0,
            "register_basis": "unknown_register",
            "present_harmonics": "",
            "missing_harmonics": "",
        }

    observed_by_degree: dict[int, list[str]] = defaultdict(list)
    for token in observed_tokens:
        parsed = _parse_token(token)
        if not parsed:
            continue
        _oct, degree, _micro = parsed
        observed_by_degree[degree].append(token)

    present: list[str] = []
    missing: list[str] = []
    unweighted_hits = 0
    weighted_hits = 0.0
    total_weight = sum(_BLOCK4_WEIGHTS.values())
    h57_hits = 0.0
    h57_total = _BLOCK4_WEIGHTS[5] + _BLOCK4_WEIGHTS[7]

    for h, degree in expected.items():
        toks = observed_by_degree.get(degree, [])
        if toks:
            present.append(str(h))
            unweighted_hits += 1
            weighted_hits += _BLOCK4_WEIGHTS.get(h, 0.0)
            if h in (5, 7):
                h57_hits += _BLOCK4_WEIGHTS.get(h, 0.0)
        else:
            missing.append(str(h))

    equal_support = unweighted_hits / max(len(expected), 1)
    weighted_support = weighted_hits / max(total_weight, 1e-9)
    h57_score = h57_hits / max(h57_total, 1e-9)
    register_score, reg_basis = _register_compensation(root_token, weighted_support, h57_score)

    return {
        "equal_support_score": equal_support,
        "weighted_support_score": weighted_support,
        "harmonic_5_7_score": h57_score,
        "block4_register_score": register_score,
        "register_basis": reg_basis,
        "present_harmonics": " ".join(present),
        "missing_harmonics": " ".join(missing),
    }


def save_heatmap_png(data: np.ndarray, x_labels: list[int], y_labels: list[str], title: str, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, max(6, len(y_labels) * 0.42)))
    im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap="magma", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("frame60")
    ax.set_ylabel("track::note")
    x_tick_idx = np.linspace(0, len(x_labels) - 1, min(12, len(x_labels)), dtype=int) if x_labels else np.array([], dtype=int)
    ax.set_xticks(x_tick_idx)
    ax.set_xticklabels([str(x_labels[i]) for i in x_tick_idx], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.0195)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def save_heatmap_html(
    x_labels: list[int],
    y_labels: list[str],
    equal_map: np.ndarray,
    weighted_map: np.ndarray,
    h57_map: np.ndarray,
    out_html: Path,
) -> None:
    payload = {
        "equal": equal_map.tolist(),
        "weighted": weighted_map.tolist(),
        "h57": h57_map.tolist(),
        "x": x_labels,
        "y": y_labels,
    }
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Ave Maria fragment harmonic note map</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
<h2>Ave Maria fragment harmonic note map</h2>
<p>Three synchronized heatmaps: equal harmonic support, Block4 weighted support, and dedicated 5/7 support.</p>
<div id="equal" style="width:100%;height:520px;"></div>
<div id="weighted" style="width:100%;height:520px;"></div>
<div id="h57" style="width:100%;height:520px;"></div>
<script>
const payload = {json.dumps(payload, ensure_ascii=False)};
function draw(divId, zData, title) {{
  Plotly.newPlot(divId, [{{
    type: "heatmap",
    z: zData,
    x: payload.x,
    y: payload.y,
    colorscale: "Magma",
    zmin: 0,
    zmax: 1
  }}], {{
    title: title,
    xaxis: {{title: "frame60"}},
    yaxis: {{title: "track::note", automargin: true}},
    margin: {{l: 180, r: 20, t: 50, b: 60}}
  }});
}}
draw("equal", payload.equal, "Equal harmonic support");
draw("weighted", payload.weighted, "Block4-weighted harmonic support");
draw("h57", payload.h57, "5/7 harmonic support");
</script>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a harmonic note map for an Ave Maria fragment using all harmonics and Block4 5/7 emphasis.")
    ap.add_argument("--start-sec", type=float, required=True)
    ap.add_argument("--end-sec", type=float, required=True)
    ap.add_argument("--fragment-points-csv", required=True)
    ap.add_argument("--midi-csv", default=str(MIDI_CSV))
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    start_frame = int(round(float(args.start_sec) * 60.0))
    end_frame = int(round(float(args.end_sec) * 60.0))
    observed_by_frame = load_fragment_tokens(Path(args.fragment_points_csv), start_frame, end_frame)
    roots = load_midi_unique_roots(Path(args.midi_csv), float(args.start_sec), float(args.end_sec))

    frames = list(range(start_frame, end_frame + 1))
    frame_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    equal_map = np.zeros((len(roots), len(frames)), dtype=float)
    weighted_map = np.zeros((len(roots), len(frames)), dtype=float)
    h57_map = np.zeros((len(roots), len(frames)), dtype=float)

    for row_idx, root in enumerate(roots):
        equal_vals: list[float] = []
        weighted_vals: list[float] = []
        h57_vals: list[float] = []
        reg_vals: list[float] = []
        active_frames: list[int] = []
        best_present = ""
        best_weight = -1.0

        for col_idx, frame in enumerate(frames):
            support = support_for_root(root["note_token"], observed_by_frame.get(frame, set()))
            equal = float(support["equal_support_score"])
            weighted = float(support["weighted_support_score"])
            h57 = float(support["harmonic_5_7_score"])
            reg = float(support["block4_register_score"])

            equal_map[row_idx, col_idx] = equal
            weighted_map[row_idx, col_idx] = weighted
            h57_map[row_idx, col_idx] = h57
            equal_vals.append(equal)
            weighted_vals.append(weighted)
            h57_vals.append(h57)
            reg_vals.append(reg)
            if weighted >= 0.40 or h57 >= 0.50:
                active_frames.append(frame)
            if weighted > best_weight:
                best_weight = weighted
                best_present = str(support["present_harmonics"])

            frame_rows.append(
                {
                    "frame60": frame,
                    "time_sec": frame / 60.0,
                    "track_name": root["track_name"],
                    "reference_note_token": root["note_token"],
                    "label": root["label"],
                    "freq_hz": f"{float(root['freq_hz']):.6f}",
                    "equal_support_score": f"{equal:.9f}",
                    "weighted_support_score": f"{weighted:.9f}",
                    "harmonic_5_7_score": f"{h57:.9f}",
                    "block4_register_score": f"{reg:.9f}",
                    "present_harmonics": support["present_harmonics"],
                    "missing_harmonics": support["missing_harmonics"],
                    "register_basis": support["register_basis"],
                }
            )

        summary_rows.append(
            {
                "track_name": root["track_name"],
                "reference_note_token": root["note_token"],
                "label": root["label"],
                "freq_hz": f"{float(root['freq_hz']):.6f}",
                "occurrences_in_midi_window": int(root["occurrences"]),
                "mean_equal_support_score": f"{float(np.mean(equal_vals)):.9f}",
                "max_equal_support_score": f"{float(np.max(equal_vals)):.9f}",
                "mean_weighted_support_score": f"{float(np.mean(weighted_vals)):.9f}",
                "max_weighted_support_score": f"{float(np.max(weighted_vals)):.9f}",
                "mean_harmonic_5_7_score": f"{float(np.mean(h57_vals)):.9f}",
                "max_harmonic_5_7_score": f"{float(np.max(h57_vals)):.9f}",
                "mean_block4_register_score": f"{float(np.mean(reg_vals)):.9f}",
                "max_block4_register_score": f"{float(np.max(reg_vals)):.9f}",
                "active_frame_count": len(active_frames),
                "first_active_frame": active_frames[0] if active_frames else "",
                "last_active_frame": active_frames[-1] if active_frames else "",
                "best_present_harmonics": best_present,
            }
        )

    frame_df = pd.DataFrame(frame_rows)
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(["max_weighted_support_score", "max_harmonic_5_7_score", "freq_hz"], ascending=[False, False, True]).reset_index(drop=True)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_frame_csv = Path(f"{args.out_prefix}__frame_scores.csv")
    out_summary_csv = Path(f"{args.out_prefix}__summary.csv")
    out_summary_txt = Path(f"{args.out_prefix}__summary.txt")
    out_html = Path(f"{args.out_prefix}__heatmap.html")
    out_png_equal = Path(f"{args.out_prefix}__equal_support.png")
    out_png_weighted = Path(f"{args.out_prefix}__weighted_support.png")
    out_png_h57 = Path(f"{args.out_prefix}__h57_support.png")
    out_meta = Path(f"{args.out_prefix}__meta.json")

    frame_df.to_csv(out_frame_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_summary_csv, index=False, encoding="utf-8-sig")

    y_labels = [str(r["label"]) for r in roots]
    save_heatmap_png(equal_map, frames, y_labels, "Ave Maria fragment - equal harmonic support", out_png_equal)
    save_heatmap_png(weighted_map, frames, y_labels, "Ave Maria fragment - Block4 weighted support", out_png_weighted)
    save_heatmap_png(h57_map, frames, y_labels, "Ave Maria fragment - 5/7 harmonic support", out_png_h57)
    save_heatmap_html(frames, y_labels, equal_map, weighted_map, h57_map, out_html)

    lines = [
        "AVE MARIA FRAGMENT HARMONIC NOTE MAP",
        "=" * 72,
        f"window_sec: {float(args.start_sec):.3f} -> {float(args.end_sec):.3f}",
        f"window_frames60: {start_frame} -> {end_frame}",
        f"candidate_unique_midi_roots: {len(roots)}",
        "",
        "top_by_weighted_support:",
    ]
    for _, row in summary_df.head(12).iterrows():
        lines.append(
            f"  {row['label']}: max_weighted={row['max_weighted_support_score']}  max_h57={row['max_harmonic_5_7_score']}  active_frames={row['active_frame_count']}  best_harmonics={row['best_present_harmonics']}"
        )
    lines.append("")
    lines.append("notes:")
    lines.append("  - equal_support_score: fraction of harmonics 2..8 present with equal weight")
    lines.append("  - weighted_support_score: Block4-style score with strong emphasis on harmonics 5 and 7")
    lines.append("  - harmonic_5_7_score: isolated score for harmonics 5 and 7 only")
    lines.append("  - block4_register_score: register compensation built on weighted + 5/7 support")
    out_summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    out_meta.write_text(
        json.dumps(
            {
                "window_sec": [float(args.start_sec), float(args.end_sec)],
                "window_frames60": [start_frame, end_frame],
                "candidate_unique_midi_roots": len(roots),
                "frame_scores_csv": str(out_frame_csv),
                "summary_csv": str(out_summary_csv),
                "heatmap_html": str(out_html),
                "png_equal": str(out_png_equal),
                "png_weighted": str(out_png_weighted),
                "png_h57": str(out_png_h57),
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print(f"WROTE {out_frame_csv}")
    print(f"WROTE {out_summary_csv}")
    print(f"WROTE {out_summary_txt}")
    print(f"WROTE {out_html}")


if __name__ == "__main__":
    main()
