from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


# ============================================================
# DUODECIMAL HELPERS
# ============================================================

DUO_MAP = {
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
    "7": 7, "8": 8, "9": 9, "A": 10, "B": 11, "C": 12,
}


def duo_digit_to_int(ch: str) -> int:
    ch = ch.strip().upper()
    if ch not in DUO_MAP:
        raise ValueError(f"Unsupported duodecimal digit: {ch}")
    return DUO_MAP[ch]


def duo_str_to_int(s: str) -> int:
    s = s.strip().upper()
    if not s:
        raise ValueError("Empty duodecimal string")
    value = 0
    for ch in s:
        value = value * 12 + duo_digit_to_int(ch)
    return value


def parse_note_token_sort_key(token: str) -> tuple[int, int, str]:
    token = (token or "").strip().replace(" ", "")
    if "." not in token:
        return (999999, 999999, token)

    left, right = token.split(".", 1)

    octave_part = ""
    degree_part = ""

    for ch in left:
        if ch.upper() in DUO_MAP:
            octave_part += ch.upper()
        else:
            break

    for ch in right:
        if ch.upper() in DUO_MAP:
            degree_part += ch.upper()
        else:
            break

    if not octave_part or not degree_part:
        return (999999, 999999, token)

    try:
        octave = duo_str_to_int(octave_part)
        degree = duo_str_to_int(degree_part)
        return (octave, degree, token)
    except Exception:
        return (999999, 999999, token)


# ============================================================
# HELPERS
# ============================================================

def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def normalize_series(values: list[float]) -> list[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-12:
        return [1.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class Row:
    note: str
    target_zone_ratio: float
    core_ratio: float
    mean_target_convergence_score: float

    note_index: int = 0
    norm_target_zone_ratio: float = 0.0
    norm_core_ratio: float = 0.0
    norm_mean_target_convergence_score: float = 0.0
    composite_stability_score: float = 0.0


# ============================================================
# LOAD
# ============================================================

def load_rows(path: Path) -> list[Row]:
    rows: list[Row] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                Row(
                    note=(r.get("note", "") or "").strip(),
                    target_zone_ratio=safe_float(r.get("target_zone_ratio", 0.0)),
                    core_ratio=safe_float(r.get("core_ratio", 0.0)),
                    mean_target_convergence_score=safe_float(
                        r.get("mean_target_convergence_score", 0.0)
                    ),
                )
            )

    rows.sort(key=lambda x: parse_note_token_sort_key(x.note))
    for i, row in enumerate(rows, start=1):
        row.note_index = i

    return rows


# ============================================================
# ANALYSIS
# ============================================================

def enrich_rows(rows: list[Row]) -> None:
    tz = [r.target_zone_ratio for r in rows]
    cr = [r.core_ratio for r in rows]
    mc = [r.mean_target_convergence_score for r in rows]

    tz_n = normalize_series(tz)
    cr_n = normalize_series(cr)
    mc_n = normalize_series(mc)

    for row, a, b, c in zip(rows, tz_n, cr_n, mc_n):
        row.norm_target_zone_ratio = a
        row.norm_core_ratio = b
        row.norm_mean_target_convergence_score = c
        row.composite_stability_score = (a + b + c) / 3.0


def weighted_center_index(rows: list[Row], attr: str) -> float:
    weights = [max(0.0, getattr(r, attr)) for r in rows]
    total = sum(weights)
    if total <= 1e-12:
        return 0.0
    return sum(r.note_index * w for r, w in zip(rows, weights)) / total


def nearest_row_to_index(rows: list[Row], x: float) -> Row:
    return min(rows, key=lambda r: abs(r.note_index - x))


def build_summary(rows: list[Row]) -> str:
    if not rows:
        return "No rows loaded."

    peak_core = max(rows, key=lambda r: r.core_ratio)
    peak_conv = max(rows, key=lambda r: r.mean_target_convergence_score)
    peak_target = max(rows, key=lambda r: r.target_zone_ratio)
    peak_composite = max(rows, key=lambda r: r.composite_stability_score)

    center_target_idx = weighted_center_index(rows, "target_zone_ratio")
    center_core_idx = weighted_center_index(rows, "core_ratio")
    center_conv_idx = weighted_center_index(rows, "mean_target_convergence_score")
    center_comp_idx = weighted_center_index(rows, "composite_stability_score")

    center_target_row = nearest_row_to_index(rows, center_target_idx)
    center_core_row = nearest_row_to_index(rows, center_core_idx)
    center_conv_row = nearest_row_to_index(rows, center_conv_idx)
    center_comp_row = nearest_row_to_index(rows, center_comp_idx)

    lines: list[str] = []
    lines.append("RESONANCE CENTER ANALYSIS")
    lines.append("=" * 80)
    lines.append(f"note_count: {len(rows)}")
    lines.append("")
    lines.append("LOCAL PEAKS")
    lines.append(f"  peak_target_zone_ratio: {peak_target.note} -> {peak_target.target_zone_ratio:.6f}")
    lines.append(f"  peak_core_ratio: {peak_core.note} -> {peak_core.core_ratio:.6f}")
    lines.append(f"  peak_mean_target_convergence_score: {peak_conv.note} -> {peak_conv.mean_target_convergence_score:.6f}")
    lines.append(f"  peak_composite_stability_score: {peak_composite.note} -> {peak_composite.composite_stability_score:.6f}")
    lines.append("")
    lines.append("WEIGHTED CENTERS")
    lines.append(f"  target_zone_center_index: {center_target_idx:.6f} -> nearest_note={center_target_row.note}")
    lines.append(f"  core_ratio_center_index: {center_core_idx:.6f} -> nearest_note={center_core_row.note}")
    lines.append(f"  mean_convergence_center_index: {center_conv_idx:.6f} -> nearest_note={center_conv_row.note}")
    lines.append(f"  composite_stability_center_index: {center_comp_idx:.6f} -> nearest_note={center_comp_row.note}")
    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("  - local peak = strongest point-like manifestation of the system.")
    lines.append("  - weighted center = center of gravity of the whole stability zone.")
    lines.append("  - divergence between local peak and weighted center indicates that stability is a plateau, not a single point.")
    lines.append("")
    lines.append("RECOMMENDED PROJECT READING")
    lines.append(f"  - sharpest organization peak: {peak_composite.note}")
    lines.append(f"  - integrated resonance center: {center_comp_row.note}")
    return "\n".join(lines)


# ============================================================
# CSV / META
# ============================================================

def write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = [
        "note",
        "note_index",
        "target_zone_ratio",
        "core_ratio",
        "mean_target_convergence_score",
        "norm_target_zone_ratio",
        "norm_core_ratio",
        "norm_mean_target_convergence_score",
        "composite_stability_score",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "note": r.note,
                    "note_index": r.note_index,
                    "target_zone_ratio": r.target_zone_ratio,
                    "core_ratio": r.core_ratio,
                    "mean_target_convergence_score": r.mean_target_convergence_score,
                    "norm_target_zone_ratio": r.norm_target_zone_ratio,
                    "norm_core_ratio": r.norm_core_ratio,
                    "norm_mean_target_convergence_score": r.norm_mean_target_convergence_score,
                    "composite_stability_score": r.composite_stability_score,
                }
            )


def write_txt(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_meta_json(path: Path, *, input_csv: Path, outputs: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "inputs": {
            "comparative_csv": str(input_csv),
        },
        "outputs": outputs,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# PLOTS
# ============================================================

def plot_composite(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    xs = [r.note_index for r in rows]
    ys = [r.composite_stability_score for r in rows]
    labels = [r.note for r in rows]

    fig = plt.figure(figsize=(16, 5))
    ax = fig.add_subplot(111)
    ax.plot(xs, ys)
    ax.set_title("Composite stability score across full note range")
    ax.set_xlabel("note_index")
    ax.set_ylabel("composite_stability_score")
    ax.grid(True, alpha=0.3)

    tick_step = max(1, len(xs) // 16)
    ax.set_xticks(xs[::tick_step])
    ax.set_xticklabels(labels[::tick_step], rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_normalized_overlay(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    xs = [r.note_index for r in rows]
    labels = [r.note for r in rows]

    y1 = [r.norm_target_zone_ratio for r in rows]
    y2 = [r.norm_core_ratio for r in rows]
    y3 = [r.norm_mean_target_convergence_score for r in rows]

    fig = plt.figure(figsize=(16, 5))
    ax = fig.add_subplot(111)
    ax.plot(xs, y1, label="norm_target_zone_ratio")
    ax.plot(xs, y2, label="norm_core_ratio")
    ax.plot(xs, y3, label="norm_mean_target_convergence_score")
    ax.set_title("Normalized stability metrics across full note range")
    ax.set_xlabel("note_index")
    ax.set_ylabel("normalized_value")
    ax.grid(True, alpha=0.3)
    ax.legend()

    tick_step = max(1, len(xs) // 16)
    ax.set_xticks(xs[::tick_step])
    ax.set_xticklabels(labels[::tick_step], rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build resonance center analysis from comparative range analysis. "
            "This layer estimates both local peak and weighted resonance center."
        )
    )
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_txt", required=True)
    ap.add_argument("--out_plot_composite_png", required=True)
    ap.add_argument("--out_plot_overlay_png", required=True)
    ap.add_argument("--out_meta_json", required=True)
    args = ap.parse_args()

    input_csv = Path(args.input_csv).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_txt = Path(args.out_txt).resolve()
    out_plot_composite_png = Path(args.out_plot_composite_png).resolve()
    out_plot_overlay_png = Path(args.out_plot_overlay_png).resolve()
    out_meta_json = Path(args.out_meta_json).resolve()

    rows = load_rows(input_csv)
    enrich_rows(rows)

    write_csv(out_csv, rows)
    write_txt(out_txt, build_summary(rows))
    plot_composite(out_plot_composite_png, rows)
    plot_normalized_overlay(out_plot_overlay_png, rows)

    write_meta_json(
        out_meta_json,
        input_csv=input_csv,
        outputs={
            "resonance_center_csv": str(out_csv),
            "resonance_center_txt": str(out_txt),
            "plot_composite_png": str(out_plot_composite_png),
            "plot_overlay_png": str(out_plot_overlay_png),
        },
    )

    print("resonance center analysis build complete")
    print(json.dumps(
        {
            "row_count": len(rows),
            "out_csv": str(out_csv),
            "out_txt": str(out_txt),
            "out_plot_composite_png": str(out_plot_composite_png),
            "out_plot_overlay_png": str(out_plot_overlay_png),
            "out_meta_json": str(out_meta_json),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()