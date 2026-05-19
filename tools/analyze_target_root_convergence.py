from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt


# ============================================================
# DUODECIMAL / NOTE PARSING
# ============================================================

def duodecimal_digit_to_int(ch: str) -> int:
    ch = ch.strip().upper()
    mapping = {
        "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
        "7": 7, "8": 8, "9": 9, "A": 10, "B": 11, "C": 12,
    }
    if ch not in mapping:
        raise ValueError(f"Unsupported duodecimal digit: {ch}")
    return mapping[ch]


def duodecimal_str_to_int(s: str) -> int:
    s = s.strip().upper()
    if not s:
        raise ValueError("Empty duodecimal string")

    value = 0
    for ch in s:
        value = value * 12 + duodecimal_digit_to_int(ch)
    return value


@dataclass(frozen=True)
class NoteGeometry:
    token: str
    octave: int
    degree: int
    phase_deg: float
    radial_level: float


def parse_note_geometry(token: str) -> Optional[NoteGeometry]:
    if token is None:
        return None

    token = str(token).strip().replace(" ", "")
    if not token:
        return None

    m = re.match(r"^([1-9ABC]+)\.([1-9ABC]+)", token, flags=re.IGNORECASE)
    if not m:
        return None

    octave = duodecimal_str_to_int(m.group(1))
    degree = duodecimal_str_to_int(m.group(2))
    phase_deg = float(degree) * 30.0
    radial_level = float(octave) + float(degree) / 12.0

    return NoteGeometry(
        token=token,
        octave=octave,
        degree=degree,
        phase_deg=phase_deg,
        radial_level=radial_level,
    )


def safe_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def safe_int(v: str, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


# ============================================================
# DATA MODEL
# ============================================================

@dataclass(frozen=True)
class TargetAnalysisRow:
    segment_index: int
    chosen_time_sec: float
    representative_rc_note: str
    representative_rc_hz: float
    strongest_peak_note: str
    strongest_peak_hz: float

    best_theoretical_root_token: str
    best_theoretical_root_score: float
    best_theoretical_chain_string: str

    matched_harmonics_same_frame: str
    matched_harmonics_window: str
    missing_harmonics_window: str
    extra_tokens_window: str

    phase_deg: float
    radial_level: float
    phase_delta: Optional[float]
    radial_delta: Optional[float]

    phase_consistency_score: float
    radial_consistency_score: float
    window_chain_match_score: float
    theoretical_chain_verdict: str

    stabilization_role: str
    stabilization_reason: str
    stabilization_score: float

    target_phase_distance: float
    target_radial_distance: float
    target_is_best_root: bool
    target_convergence_score: float
    target_state: str


# ============================================================
# CORE ANALYSIS
# ============================================================

def classify_target_state(
    *,
    target_is_best_root: bool,
    target_phase_distance: float,
    target_radial_distance: float,
    window_chain_match_score: float,
    phase_consistency_score: float,
    radial_consistency_score: float,
    theoretical_chain_verdict: str,
) -> str:
    if target_is_best_root:
        if theoretical_chain_verdict == "CHAIN_PHASE_CONFIRMED":
            return "TARGET_CONVERGENCE_NODE"
        if theoretical_chain_verdict == "CHAIN_CONFIRMED":
            return "TARGET_HOLD"
        if phase_consistency_score >= 0.25 or radial_consistency_score >= 0.25:
            return "PHASE_LOCK_TO_TARGET"

    if target_phase_distance <= 30.0 and target_radial_distance <= 1.0:
        if window_chain_match_score > -2.0:
            return "APPROACHING_TARGET"

    if target_phase_distance <= 30.0:
        return "TARGET_PHASE_NEAR"

    if target_radial_distance <= 1.0:
        return "TARGET_RADIAL_NEAR"

    if target_is_best_root:
        return "TARGET_WEAK"

    return "COMPETING_ROOT"


def build_target_convergence_score(
    *,
    target_is_best_root: bool,
    target_phase_distance: float,
    target_radial_distance: float,
    window_chain_match_score: float,
    phase_consistency_score: float,
    radial_consistency_score: float,
    stabilization_score: float,
) -> float:
    root_bonus = 2.0 if target_is_best_root else 0.0
    phase_bonus = max(0.0, 1.0 - target_phase_distance / 180.0) * 3.0
    radial_bonus = max(0.0, 1.0 - min(target_radial_distance, 4.0) / 4.0) * 3.0

    return (
        root_bonus
        + phase_bonus
        + radial_bonus
        + window_chain_match_score * 0.6
        + phase_consistency_score * 3.0
        + radial_consistency_score * 2.0
        + stabilization_score * 0.05
    )


def load_and_analyze(
    input_csv: Path,
    target_root_token: str,
) -> tuple[list[TargetAnalysisRow], NoteGeometry]:
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        raw_rows = list(csv.DictReader(f))

    target_geo = parse_note_geometry(target_root_token)
    if target_geo is None:
        raise ValueError(f"Cannot parse target_root_token: {target_root_token}")

    analyzed: list[TargetAnalysisRow] = []

    for r in raw_rows:
        rc_note = (r.get("representative_rc_note", "") or "").strip()
        rc_geo = parse_note_geometry(rc_note)

        phase_deg = safe_float(r.get("phase_deg", ""), 0.0)
        radial_level = safe_float(r.get("radial_level", ""), 0.0)

        if rc_geo is not None:
            phase_deg = rc_geo.phase_deg
            radial_level = rc_geo.radial_level

        target_phase_distance = abs(phase_deg - target_geo.phase_deg)
        target_phase_distance = min(target_phase_distance, 360.0 - target_phase_distance)

        target_radial_distance = abs(radial_level - target_geo.radial_level)

        best_root = (r.get("best_theoretical_root_token", "") or "").strip()
        target_is_best_root = best_root == target_root_token

        phase_consistency_score = safe_float(r.get("phase_consistency_score", ""), 0.0)
        radial_consistency_score = safe_float(r.get("radial_consistency_score", ""), 0.0)
        window_chain_match_score = safe_float(r.get("window_chain_match_score", ""), 0.0)
        stabilization_score = safe_float(r.get("stabilization_score", ""), 0.0)
        theoretical_chain_verdict = (r.get("theoretical_chain_verdict", "") or "").strip()

        target_convergence_score = build_target_convergence_score(
            target_is_best_root=target_is_best_root,
            target_phase_distance=target_phase_distance,
            target_radial_distance=target_radial_distance,
            window_chain_match_score=window_chain_match_score,
            phase_consistency_score=phase_consistency_score,
            radial_consistency_score=radial_consistency_score,
            stabilization_score=stabilization_score,
        )

        target_state = classify_target_state(
            target_is_best_root=target_is_best_root,
            target_phase_distance=target_phase_distance,
            target_radial_distance=target_radial_distance,
            window_chain_match_score=window_chain_match_score,
            phase_consistency_score=phase_consistency_score,
            radial_consistency_score=radial_consistency_score,
            theoretical_chain_verdict=theoretical_chain_verdict,
        )

        analyzed.append(
            TargetAnalysisRow(
                segment_index=safe_int(r.get("segment_index", ""), 0),
                chosen_time_sec=safe_float(r.get("chosen_time_sec", ""), 0.0),
                representative_rc_note=rc_note,
                representative_rc_hz=safe_float(r.get("representative_rc_hz", ""), 0.0),
                strongest_peak_note=(r.get("strongest_peak_note", "") or "").strip(),
                strongest_peak_hz=safe_float(r.get("strongest_peak_hz", ""), 0.0),

                best_theoretical_root_token=best_root,
                best_theoretical_root_score=safe_float(r.get("best_theoretical_root_score", ""), 0.0),
                best_theoretical_chain_string=(r.get("best_theoretical_chain_string", "") or "").strip(),

                matched_harmonics_same_frame=(r.get("matched_harmonics_same_frame", "") or "").strip(),
                matched_harmonics_window=(r.get("matched_harmonics_window", "") or "").strip(),
                missing_harmonics_window=(r.get("missing_harmonics_window", "") or "").strip(),
                extra_tokens_window=(r.get("extra_tokens_window", "") or "").strip(),

                phase_deg=phase_deg,
                radial_level=radial_level,
                phase_delta=safe_float(r.get("phase_delta", ""), 0.0) if str(r.get("phase_delta", "")).strip() else None,
                radial_delta=safe_float(r.get("radial_delta", ""), 0.0) if str(r.get("radial_delta", "")).strip() else None,

                phase_consistency_score=phase_consistency_score,
                radial_consistency_score=radial_consistency_score,
                window_chain_match_score=window_chain_match_score,
                theoretical_chain_verdict=theoretical_chain_verdict,

                stabilization_role=(r.get("stabilization_role", "") or "").strip(),
                stabilization_reason=(r.get("stabilization_reason", "") or "").strip(),
                stabilization_score=stabilization_score,

                target_phase_distance=target_phase_distance,
                target_radial_distance=target_radial_distance,
                target_is_best_root=target_is_best_root,
                target_convergence_score=target_convergence_score,
                target_state=target_state,
            )
        )

    return analyzed, target_geo


# ============================================================
# OUTPUTS
# ============================================================

def write_analysis_csv(path: Path, rows: list[TargetAnalysisRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].__dict__.keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def write_analysis_txt(path: Path, rows: list[TargetAnalysisRow], target_geo: NoteGeometry) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    state_counts = Counter(r.target_state for r in rows)
    best_root_counts = Counter(r.best_theoretical_root_token for r in rows if r.best_theoretical_root_token)
    target_hits = [r for r in rows if r.target_is_best_root]

    mean_target_score = sum(r.target_convergence_score for r in rows) / len(rows) if rows else 0.0
    mean_phase_dist = sum(r.target_phase_distance for r in rows) / len(rows) if rows else 0.0
    mean_radial_dist = sum(r.target_radial_distance for r in rows) / len(rows) if rows else 0.0

    with path.open("w", encoding="utf-8") as f:
        f.write("TARGET ROOT CONVERGENCE ANALYSIS\n")
        f.write("=" * 80 + "\n")
        f.write(f"target_root_token: {target_geo.token}\n")
        f.write(f"target_phase_deg: {target_geo.phase_deg:.6f}\n")
        f.write(f"target_radial_level: {target_geo.radial_level:.6f}\n")
        f.write(f"row_count: {len(rows)}\n")
        f.write(f"target_best_root_count: {len(target_hits)}\n")
        f.write(f"mean_target_convergence_score: {mean_target_score:.6f}\n")
        f.write(f"mean_target_phase_distance: {mean_phase_dist:.6f}\n")
        f.write(f"mean_target_radial_distance: {mean_radial_dist:.6f}\n")
        f.write("\nSTATE COUNTS\n")
        for k, v in state_counts.most_common():
            f.write(f"  {k}: {v}\n")
        f.write("\nBEST ROOT COUNTS\n")
        for k, v in best_root_counts.most_common(20):
            f.write(f"  {k}: {v}\n")

        top_hits = sorted(rows, key=lambda r: r.target_convergence_score, reverse=True)[:20]
        f.write("\nTOP TARGET CONVERGENCE SEGMENTS\n")
        for r in top_hits:
            f.write(
                f"  segment={r.segment_index} time={r.chosen_time_sec:.6f} "
                f"best_root={r.best_theoretical_root_token} "
                f"score={r.target_convergence_score:.6f} "
                f"state={r.target_state} "
                f"phase_dist={r.target_phase_distance:.6f} "
                f"radial_dist={r.target_radial_distance:.6f}\n"
            )


def plot_convergence_score(path: Path, rows: list[TargetAnalysisRow], target_token: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    xs = [r.segment_index for r in rows]
    ys = [r.target_convergence_score for r in rows]

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    ax.plot(xs, ys)
    ax.set_title(f"Target convergence score to {target_token}")
    ax.set_xlabel("segment_index")
    ax.set_ylabel("target_convergence_score")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_spiral_3d(path: Path, rows: list[TargetAnalysisRow], target_geo: NoteGeometry) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    xs = []
    ys = []
    zs = []
    colors = []

    for r in rows:
        angle = math.radians(r.phase_deg)
        radius = r.radial_level
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = r.segment_index

        xs.append(x)
        ys.append(y)
        zs.append(z)
        colors.append(r.representative_rc_hz if r.representative_rc_hz > 0 else 0.0)

    target_angle = math.radians(target_geo.phase_deg)
    tx = target_geo.radial_level * math.cos(target_angle)
    ty = target_geo.radial_level * math.sin(target_angle)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(xs, ys, zs, c=colors, cmap="rainbow", s=22)
    ax.plot(xs, ys, zs, alpha=0.35)

    ax.scatter([tx], [ty], [0], marker="x", s=120)
    ax.set_title(f"Spiral convergence trajectory to {target_geo.token}")
    ax.set_xlabel("spiral_x")
    ax.set_ylabel("spiral_y")
    ax.set_zlabel("segment_index")
    fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.1, label="representative_rc_hz")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_meta_json(path: Path, input_csv: Path, target_token: str, csv_out: Path, txt_out: Path, plot_score: Path, plot_spiral: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "input_csv": str(input_csv),
        "target_root_token": target_token,
        "outputs": {
            "analysis_csv": str(csv_out),
            "analysis_txt": str(txt_out),
            "plot_convergence_score_png": str(plot_score),
            "plot_spiral_3d_png": str(plot_spiral),
        },
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Analyze convergence of stabilized rc data toward a given target root token. "
            "This is a laboratory analysis tool, not a recognition algorithm."
        )
    )
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--target_root_token", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_txt", required=True)
    ap.add_argument("--out_plot_score_png", required=True)
    ap.add_argument("--out_plot_spiral_png", required=True)
    ap.add_argument("--out_meta_json", required=True)
    args = ap.parse_args()

    input_csv = Path(args.input_csv).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_txt = Path(args.out_txt).resolve()
    out_plot_score_png = Path(args.out_plot_score_png).resolve()
    out_plot_spiral_png = Path(args.out_plot_spiral_png).resolve()
    out_meta_json = Path(args.out_meta_json).resolve()

    rows, target_geo = load_and_analyze(input_csv, args.target_root_token)

    write_analysis_csv(out_csv, rows)
    write_analysis_txt(out_txt, rows, target_geo)
    plot_convergence_score(out_plot_score_png, rows, target_geo.token)
    plot_spiral_3d(out_plot_spiral_png, rows, target_geo)
    write_meta_json(
        out_meta_json,
        input_csv=input_csv,
        target_token=target_geo.token,
        csv_out=out_csv,
        txt_out=out_txt,
        plot_score=out_plot_score_png,
        plot_spiral=out_plot_spiral_png,
    )

    print("target root convergence analysis complete")
    print(json.dumps(
        {
            "row_count": len(rows),
            "target_root_token": target_geo.token,
            "out_csv": str(out_csv),
            "out_txt": str(out_txt),
            "out_plot_score_png": str(out_plot_score_png),
            "out_plot_spiral_png": str(out_plot_spiral_png),
            "out_meta_json": str(out_meta_json),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()