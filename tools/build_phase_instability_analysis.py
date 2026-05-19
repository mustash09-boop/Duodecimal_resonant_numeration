from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
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


def safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def pstdev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    return (sum((x - m) ** 2 for x in values) / len(values)) ** 0.5


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class PhaseRow:
    note: str
    note_index: int = 0

    segments: int = 0

    mean_target_phase_distance: float = 0.0
    phase_distance_std: float = 0.0
    max_target_phase_distance: float = 0.0

    mean_target_radial_distance: float = 0.0
    radial_distance_std: float = 0.0

    phase_lock_ratio: float = 0.0
    approaching_ratio: float = 0.0
    target_phase_near_ratio: float = 0.0
    target_radial_near_ratio: float = 0.0
    competing_root_ratio: float = 0.0
    target_hold_ratio: float = 0.0
    target_convergence_node_ratio: float = 0.0

    mean_target_convergence_score: float = 0.0

    phase_instability_score: float = 0.0
    phase_coherence_score: float = 0.0


# ============================================================
# LOAD
# ============================================================

def load_one_csv(path: Path) -> PhaseRow:
    with path.open("r", encoding="utf-8", newline="") as f:
        data = list(csv.DictReader(f))

    prefix = path.name.replace("__target_root_convergence.csv", "")
    note = prefix.split("__")[2] if "__" in prefix else prefix

    if not data:
        return PhaseRow(note=note, segments=0)

    phase = [safe_float(r.get("target_phase_distance", 0.0)) for r in data]
    radial = [safe_float(r.get("target_radial_distance", 0.0)) for r in data]
    conv = [safe_float(r.get("target_convergence_score", 0.0)) for r in data]
    states = [(r.get("target_state", "") or "").strip() for r in data]
    c = Counter(states)

    n = len(data)

    phase_std = pstdev(phase)
    mean_phase = mean(phase)
    mean_radial = mean(radial)

    phase_lock_ratio = c.get("PHASE_LOCK_TO_TARGET", 0) / n
    approaching_ratio = c.get("APPROACHING_TARGET", 0) / n
    target_phase_near_ratio = c.get("TARGET_PHASE_NEAR", 0) / n
    target_radial_near_ratio = c.get("TARGET_RADIAL_NEAR", 0) / n
    competing_root_ratio = c.get("COMPETING_ROOT", 0) / n
    target_hold_ratio = c.get("TARGET_HOLD", 0) / n
    target_convergence_node_ratio = c.get("TARGET_CONVERGENCE_NODE", 0) / n

    # Чем выше std и COMPETING_ROOT, тем хуже.
    # Чем выше PHASE_LOCK, тем лучше.
    phase_instability_score = (
        phase_std
        + competing_root_ratio * 30.0
        - phase_lock_ratio * 20.0
    )

    # Простая фазовая когерентность для встраивания в Block002.
    phase_coherence_score = 1.0 / (1.0 + phase_std)

    return PhaseRow(
        note=note,
        segments=n,

        mean_target_phase_distance=mean_phase,
        phase_distance_std=phase_std,
        max_target_phase_distance=max(phase) if phase else 0.0,

        mean_target_radial_distance=mean_radial,
        radial_distance_std=pstdev(radial),

        phase_lock_ratio=phase_lock_ratio,
        approaching_ratio=approaching_ratio,
        target_phase_near_ratio=target_phase_near_ratio,
        target_radial_near_ratio=target_radial_near_ratio,
        competing_root_ratio=competing_root_ratio,
        target_hold_ratio=target_hold_ratio,
        target_convergence_node_ratio=target_convergence_node_ratio,

        mean_target_convergence_score=mean(conv),

        phase_instability_score=phase_instability_score,
        phase_coherence_score=phase_coherence_score,
    )


def load_all(convergence_dir: Path) -> list[PhaseRow]:
    files = sorted(
        convergence_dir.glob("*__target_root_convergence.csv"),
        key=lambda p: parse_note_token_sort_key(
            p.name.replace("__target_root_convergence.csv", "").split("__")[2]
            if "__" in p.name else p.name
        ),
    )

    rows = [load_one_csv(p) for p in files]
    for i, row in enumerate(rows, start=1):
        row.note_index = i
    return rows


# ============================================================
# WRITE CSV / TXT / META
# ============================================================

def write_csv(path: Path, rows: list[PhaseRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = [
        "note",
        "note_index",
        "segments",
        "mean_target_phase_distance",
        "phase_distance_std",
        "max_target_phase_distance",
        "mean_target_radial_distance",
        "radial_distance_std",
        "phase_lock_ratio",
        "approaching_ratio",
        "target_phase_near_ratio",
        "target_radial_near_ratio",
        "competing_root_ratio",
        "target_hold_ratio",
        "target_convergence_node_ratio",
        "mean_target_convergence_score",
        "phase_instability_score",
        "phase_coherence_score",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: getattr(r, k) for k in fieldnames})


def build_summary(rows: list[PhaseRow]) -> str:
    if not rows:
        return "No rows loaded."

    min_std = min(rows, key=lambda r: r.phase_distance_std)
    max_std = max(rows, key=lambda r: r.phase_distance_std)
    max_lock = max(rows, key=lambda r: r.phase_lock_ratio)
    max_competing = max(rows, key=lambda r: r.competing_root_ratio)
    min_instability = min(rows, key=lambda r: r.phase_instability_score)
    max_instability = max(rows, key=lambda r: r.phase_instability_score)

    low_zone = rows[: max(1, len(rows) // 3)]
    mid_zone = rows[max(1, len(rows) // 3): max(2, 2 * len(rows) // 3)]
    high_zone = rows[max(2, 2 * len(rows) // 3):]

    def zone_mean(zone: list[PhaseRow], attr: str) -> float:
        if not zone:
            return 0.0
        return mean([getattr(r, attr) for r in zone])

    lines: list[str] = []
    lines.append("PHASE INSTABILITY ANALYSIS")
    lines.append("=" * 80)
    lines.append(f"note_count: {len(rows)}")
    lines.append("")
    lines.append("EXTREMA")
    lines.append(f"  min_phase_distance_std: {min_std.note} -> {min_std.phase_distance_std:.6f}")
    lines.append(f"  max_phase_distance_std: {max_std.note} -> {max_std.phase_distance_std:.6f}")
    lines.append(f"  max_phase_lock_ratio: {max_lock.note} -> {max_lock.phase_lock_ratio:.6f}")
    lines.append(f"  max_competing_root_ratio: {max_competing.note} -> {max_competing.competing_root_ratio:.6f}")
    lines.append(f"  min_phase_instability_score: {min_instability.note} -> {min_instability.phase_instability_score:.6f}")
    lines.append(f"  max_phase_instability_score: {max_instability.note} -> {max_instability.phase_instability_score:.6f}")
    lines.append("")
    lines.append("ZONE MEANS")
    lines.append(f"  low_zone_mean_phase_std: {zone_mean(low_zone, 'phase_distance_std'):.6f}")
    lines.append(f"  mid_zone_mean_phase_std: {zone_mean(mid_zone, 'phase_distance_std'):.6f}")
    lines.append(f"  high_zone_mean_phase_std: {zone_mean(high_zone, 'phase_distance_std'):.6f}")
    lines.append("")
    lines.append(f"  low_zone_mean_phase_lock_ratio: {zone_mean(low_zone, 'phase_lock_ratio'):.6f}")
    lines.append(f"  mid_zone_mean_phase_lock_ratio: {zone_mean(mid_zone, 'phase_lock_ratio'):.6f}")
    lines.append(f"  high_zone_mean_phase_lock_ratio: {zone_mean(high_zone, 'phase_lock_ratio'):.6f}")
    lines.append("")
    lines.append(f"  low_zone_mean_competing_root_ratio: {zone_mean(low_zone, 'competing_root_ratio'):.6f}")
    lines.append(f"  mid_zone_mean_competing_root_ratio: {zone_mean(mid_zone, 'competing_root_ratio'):.6f}")
    lines.append(f"  high_zone_mean_competing_root_ratio: {zone_mean(high_zone, 'competing_root_ratio'):.6f}")
    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("  - low phase_distance_std = phase stability.")
    lines.append("  - high phase_lock_ratio = harmonics stay in the same phase-organized note center.")
    lines.append("  - high competing_root_ratio = note center becomes ambiguous.")
    lines.append("  - in composition analysis, phase instability should reduce note confidence and help separate overlapping note clusters.")
    return "\n".join(lines)


def write_txt(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_meta_json(path: Path, *, convergence_dir: Path, outputs: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "inputs": {
            "convergence_dir": str(convergence_dir),
        },
        "outputs": outputs,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# PLOTS
# ============================================================

def plot_metric(path: Path, rows: list[PhaseRow], attr: str, title: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    xs = [r.note_index for r in rows]
    ys = [getattr(r, attr) for r in rows]
    labels = [r.note for r in rows]

    fig = plt.figure(figsize=(16, 5))
    ax = fig.add_subplot(111)
    ax.plot(xs, ys)
    ax.set_title(title)
    ax.set_xlabel("note_index")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    tick_step = max(1, len(xs) // 16)
    ax.set_xticks(xs[::tick_step])
    ax.set_xticklabels(labels[::tick_step], rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_phase_bundle(path: Path, rows: list[PhaseRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    xs = [r.note_index for r in rows]
    labels = [r.note for r in rows]

    y1 = [r.phase_distance_std for r in rows]
    y2 = [r.phase_lock_ratio for r in rows]
    y3 = [r.competing_root_ratio for r in rows]

    fig = plt.figure(figsize=(16, 5))
    ax = fig.add_subplot(111)
    ax.plot(xs, y1, label="phase_distance_std")
    ax.plot(xs, y2, label="phase_lock_ratio")
    ax.plot(xs, y3, label="competing_root_ratio")
    ax.set_title("Phase instability bundle across full note range")
    ax.set_xlabel("note_index")
    ax.set_ylabel("value")
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
            "Build phase instability analysis from target_root_convergence CSV files. "
            "This layer estimates phase stability and its impact on note confidence."
        )
    )
    ap.add_argument("--convergence_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_txt", required=True)
    ap.add_argument("--out_plot_phase_std_png", required=True)
    ap.add_argument("--out_plot_phase_lock_png", required=True)
    ap.add_argument("--out_plot_phase_bundle_png", required=True)
    ap.add_argument("--out_meta_json", required=True)
    args = ap.parse_args()

    convergence_dir = Path(args.convergence_dir).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_txt = Path(args.out_txt).resolve()
    out_plot_phase_std_png = Path(args.out_plot_phase_std_png).resolve()
    out_plot_phase_lock_png = Path(args.out_plot_phase_lock_png).resolve()
    out_plot_phase_bundle_png = Path(args.out_plot_phase_bundle_png).resolve()
    out_meta_json = Path(args.out_meta_json).resolve()

    rows = load_all(convergence_dir)

    write_csv(out_csv, rows)
    write_txt(out_txt, build_summary(rows))

    plot_metric(
        out_plot_phase_std_png,
        rows,
        "phase_distance_std",
        "Phase distance std across full note range",
        "phase_distance_std",
    )

    plot_metric(
        out_plot_phase_lock_png,
        rows,
        "phase_lock_ratio",
        "Phase-lock ratio across full note range",
        "phase_lock_ratio",
    )

    plot_phase_bundle(
        out_plot_phase_bundle_png,
        rows,
    )

    write_meta_json(
        out_meta_json,
        convergence_dir=convergence_dir,
        outputs={
            "phase_instability_csv": str(out_csv),
            "phase_instability_txt": str(out_txt),
            "plot_phase_std_png": str(out_plot_phase_std_png),
            "plot_phase_lock_png": str(out_plot_phase_lock_png),
            "plot_phase_bundle_png": str(out_plot_phase_bundle_png),
        },
    )

    print("phase instability analysis build complete")
    print(json.dumps(
        {
            "row_count": len(rows),
            "out_csv": str(out_csv),
            "out_txt": str(out_txt),
            "out_plot_phase_std_png": str(out_plot_phase_std_png),
            "out_plot_phase_lock_png": str(out_plot_phase_lock_png),
            "out_plot_phase_bundle_png": str(out_plot_phase_bundle_png),
            "out_meta_json": str(out_meta_json),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()