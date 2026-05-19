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
# DUODECIMAL / TOKEN HELPERS
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
    """
    Supports tokens like:
      5.A'-
      11.1'-
      C.C'-
      8.6-
    Sort by octave, then degree.
    Keep original token as tie-breaker.
    """
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
# DATA MODEL
# ============================================================

@dataclass(frozen=True)
class ComparativeRow:
    note: str

    target_zone_ratio: float
    core_ratio: float
    mean_target_convergence_score: float
    total_segments: int
    target_zone_count: int
    core_count: int
    target_best_root_ratio: float

    dominant_total: list[str]
    dominant_target_zone: list[str]
    dominant_core: list[str]

    has_h1_total: int
    has_h1_target_zone: int
    has_h1_core: int

    has_h2_total: int
    has_h2_target_zone: int
    has_h2_core: int

    dominant_total_top3: str
    dominant_target_top3: str
    dominant_core_top3: str

    strongest_prefix: str
    strongest_sequence: str

    max_cluster_h1: int
    max_cluster_h2: int
    max_cluster_h5: int
    max_cluster_h7: int

    state_count_target_phase_near: int
    state_count_target_radial_near: int
    state_count_approaching_target: int
    state_count_phase_lock_to_target: int
    state_count_competing_root: int


# ============================================================
# LOAD
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


def load_signature_json(path: Path) -> ComparativeRow:
    data = json.loads(path.read_text(encoding="utf-8"))

    note = str(data.get("target_root_token", "")).strip()

    seg = data.get("segment_statistics", {}) or {}
    harm = data.get("harmonic_frequency", {}) or {}
    arr = data.get("harmonic_arrival", {}) or {}
    clusters = data.get("harmonic_clusters", {}) or {}

    dominant_total = list(harm.get("dominant_total", []) or [])
    dominant_target = list(harm.get("dominant_target_zone", []) or [])
    dominant_core = list(harm.get("dominant_core", []) or [])

    strongest_prefixes = list(arr.get("strongest_prefixes", []) or [])
    strongest_sequences = list(arr.get("strongest_sequences", []) or [])

    state_counts = dict(seg.get("state_counts", {}) or {})

    def cluster_max(h: str) -> int:
        if h not in clusters:
            return 0
        return safe_int(clusters[h].get("max_cluster_length", 0), 0)

    return ComparativeRow(
        note=note,

        target_zone_ratio=safe_float(seg.get("target_zone_ratio", 0.0)),
        core_ratio=safe_float(seg.get("core_ratio", 0.0)),
        mean_target_convergence_score=safe_float(seg.get("mean_target_convergence_score", 0.0)),
        total_segments=safe_int(seg.get("total_segments", 0)),
        target_zone_count=safe_int(seg.get("target_zone_count", 0)),
        core_count=safe_int(seg.get("core_count", 0)),
        target_best_root_ratio=safe_float(seg.get("target_best_root_ratio", 0.0)),

        dominant_total=dominant_total,
        dominant_target_zone=dominant_target,
        dominant_core=dominant_core,

        has_h1_total=1 if "h1" in dominant_total else 0,
        has_h1_target_zone=1 if "h1" in dominant_target else 0,
        has_h1_core=1 if "h1" in dominant_core else 0,

        has_h2_total=1 if "h2" in dominant_total else 0,
        has_h2_target_zone=1 if "h2" in dominant_target else 0,
        has_h2_core=1 if "h2" in dominant_core else 0,

        dominant_total_top3=" ".join(dominant_total[:3]),
        dominant_target_top3=" ".join(dominant_target[:3]),
        dominant_core_top3=" ".join(dominant_core[:3]),

        strongest_prefix=str(strongest_prefixes[0][0]).strip() if strongest_prefixes else "",
        strongest_sequence=str(strongest_sequences[0][0]).strip() if strongest_sequences else "",

        max_cluster_h1=cluster_max("h1"),
        max_cluster_h2=cluster_max("h2"),
        max_cluster_h5=cluster_max("h5"),
        max_cluster_h7=cluster_max("h7"),

        state_count_target_phase_near=safe_int(state_counts.get("TARGET_PHASE_NEAR", 0)),
        state_count_target_radial_near=safe_int(state_counts.get("TARGET_RADIAL_NEAR", 0)),
        state_count_approaching_target=safe_int(state_counts.get("APPROACHING_TARGET", 0)),
        state_count_phase_lock_to_target=safe_int(state_counts.get("PHASE_LOCK_TO_TARGET", 0)),
        state_count_competing_root=safe_int(state_counts.get("COMPETING_ROOT", 0)),
    )


# ============================================================
# BUILD
# ============================================================

def build_rows(passports_dir: Path) -> list[ComparativeRow]:
    files = sorted(
        passports_dir.glob("*__note_emergence_signature.json"),
        key=lambda p: parse_note_token_sort_key(
            json.loads(p.read_text(encoding="utf-8")).get("target_root_token", "")
        ),
    )
    return [load_signature_json(p) for p in files]


# ============================================================
# CSV
# ============================================================

def write_comparative_csv(path: Path, rows: list[ComparativeRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = [
        "note",
        "target_zone_ratio",
        "core_ratio",
        "mean_target_convergence_score",
        "total_segments",
        "target_zone_count",
        "core_count",
        "target_best_root_ratio",
        "has_h1_total",
        "has_h1_target_zone",
        "has_h1_core",
        "has_h2_total",
        "has_h2_target_zone",
        "has_h2_core",
        "dominant_total_top3",
        "dominant_target_top3",
        "dominant_core_top3",
        "strongest_prefix",
        "strongest_sequence",
        "max_cluster_h1",
        "max_cluster_h2",
        "max_cluster_h5",
        "max_cluster_h7",
        "state_count_target_phase_near",
        "state_count_target_radial_near",
        "state_count_approaching_target",
        "state_count_phase_lock_to_target",
        "state_count_competing_root",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "note": r.note,
                "target_zone_ratio": r.target_zone_ratio,
                "core_ratio": r.core_ratio,
                "mean_target_convergence_score": r.mean_target_convergence_score,
                "total_segments": r.total_segments,
                "target_zone_count": r.target_zone_count,
                "core_count": r.core_count,
                "target_best_root_ratio": r.target_best_root_ratio,
                "has_h1_total": r.has_h1_total,
                "has_h1_target_zone": r.has_h1_target_zone,
                "has_h1_core": r.has_h1_core,
                "has_h2_total": r.has_h2_total,
                "has_h2_target_zone": r.has_h2_target_zone,
                "has_h2_core": r.has_h2_core,
                "dominant_total_top3": r.dominant_total_top3,
                "dominant_target_top3": r.dominant_target_top3,
                "dominant_core_top3": r.dominant_core_top3,
                "strongest_prefix": r.strongest_prefix,
                "strongest_sequence": r.strongest_sequence,
                "max_cluster_h1": r.max_cluster_h1,
                "max_cluster_h2": r.max_cluster_h2,
                "max_cluster_h5": r.max_cluster_h5,
                "max_cluster_h7": r.max_cluster_h7,
                "state_count_target_phase_near": r.state_count_target_phase_near,
                "state_count_target_radial_near": r.state_count_target_radial_near,
                "state_count_approaching_target": r.state_count_approaching_target,
                "state_count_phase_lock_to_target": r.state_count_phase_lock_to_target,
                "state_count_competing_root": r.state_count_competing_root,
            })


# ============================================================
# PLOTS
# ============================================================

def plot_metric(path: Path, rows: list[ComparativeRow], attr: str, title: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    xs = list(range(1, len(rows) + 1))
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


def plot_binary_pair(path: Path, rows: list[ComparativeRow], attr1: str, attr2: str, title: str, label1: str, label2: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    xs = list(range(1, len(rows) + 1))
    y1 = [getattr(r, attr1) for r in rows]
    y2 = [getattr(r, attr2) for r in rows]
    labels = [r.note for r in rows]

    fig = plt.figure(figsize=(16, 5))
    ax = fig.add_subplot(111)
    ax.plot(xs, y1, label=label1)
    ax.plot(xs, y2, label=label2)
    ax.set_title(title)
    ax.set_xlabel("note_index")
    ax.set_ylabel("presence")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    tick_step = max(1, len(xs) // 16)
    ax.set_xticks(xs[::tick_step])
    ax.set_xticklabels(labels[::tick_step], rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ============================================================
# NOTES / SUMMARY
# ============================================================

def build_summary_text(rows: list[ComparativeRow]) -> str:
    if not rows:
        return "No rows loaded."

    max_target_zone = max(rows, key=lambda r: r.target_zone_ratio)
    max_core = max(rows, key=lambda r: r.core_ratio)
    max_conv = max(rows, key=lambda r: r.mean_target_convergence_score)

    h1_core_count = sum(r.has_h1_core for r in rows)
    h2_core_count = sum(r.has_h2_core for r in rows)

    # transition estimate: first note where h1 disappears from core but h2 remains
    transition_note = ""
    for r in rows:
        if r.has_h1_core == 0 and r.has_h2_core == 1:
            transition_note = r.note
            break

    # coarse zones
    low_zone = rows[: max(1, len(rows) // 3)]
    mid_zone = rows[max(1, len(rows) // 3): max(2, 2 * len(rows) // 3)]
    high_zone = rows[max(2, 2 * len(rows) // 3):]

    def mean_of(zone: list[ComparativeRow], attr: str) -> float:
        if not zone:
            return 0.0
        return sum(getattr(r, attr) for r in zone) / len(zone)

    lines: list[str] = []
    lines.append("COMPARATIVE RANGE ANALYSIS")
    lines.append("=" * 80)
    lines.append(f"note_count: {len(rows)}")
    lines.append("")
    lines.append("GLOBAL EXTREMA")
    lines.append(f"  max_target_zone_ratio: {max_target_zone.note} -> {max_target_zone.target_zone_ratio:.6f}")
    lines.append(f"  max_core_ratio: {max_core.note} -> {max_core.core_ratio:.6f}")
    lines.append(f"  max_mean_target_convergence_score: {max_conv.note} -> {max_conv.mean_target_convergence_score:.6f}")
    lines.append("")
    lines.append("CORE BASIS")
    lines.append(f"  notes_with_h1_in_core: {h1_core_count}")
    lines.append(f"  notes_with_h2_in_core: {h2_core_count}")
    lines.append(f"  first_h1_loss_h2_keep_note: {transition_note or '(not found)'}")
    lines.append("")
    lines.append("ZONE MEANS")
    lines.append(f"  low_zone_mean_target_zone_ratio: {mean_of(low_zone, 'target_zone_ratio'):.6f}")
    lines.append(f"  low_zone_mean_core_ratio: {mean_of(low_zone, 'core_ratio'):.6f}")
    lines.append(f"  low_zone_mean_convergence: {mean_of(low_zone, 'mean_target_convergence_score'):.6f}")
    lines.append("")
    lines.append(f"  mid_zone_mean_target_zone_ratio: {mean_of(mid_zone, 'target_zone_ratio'):.6f}")
    lines.append(f"  mid_zone_mean_core_ratio: {mean_of(mid_zone, 'core_ratio'):.6f}")
    lines.append(f"  mid_zone_mean_convergence: {mean_of(mid_zone, 'mean_target_convergence_score'):.6f}")
    lines.append("")
    lines.append(f"  high_zone_mean_target_zone_ratio: {mean_of(high_zone, 'target_zone_ratio'):.6f}")
    lines.append(f"  high_zone_mean_core_ratio: {mean_of(high_zone, 'core_ratio'):.6f}")
    lines.append(f"  high_zone_mean_convergence: {mean_of(high_zone, 'mean_target_convergence_score'):.6f}")
    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("  - low range: note tends to hold a more classical harmonic basis.")
    lines.append("  - middle range: note tends to show strongest organized convergence.")
    lines.append("  - high range: note tends to lose h1 from core and reorganize around higher harmonics.")
    lines.append("  - transition note marks the beginning of basis reorganization, not an instant switch.")
    return "\n".join(lines)


def write_summary_txt(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_meta_json(path: Path, *, passports_dir: Path, outputs: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "inputs": {
            "passports_dir": str(passports_dir),
        },
        "outputs": outputs,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build comparative range analysis from all note emergence signatures. "
            "This is the macro comparative layer over the whole note range."
        )
    )
    ap.add_argument("--passports_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_notes_txt", required=True)
    ap.add_argument("--out_plot_target_zone_ratio", required=True)
    ap.add_argument("--out_plot_core_ratio", required=True)
    ap.add_argument("--out_plot_mean_convergence", required=True)
    ap.add_argument("--out_plot_h1_h2_core", required=True)
    ap.add_argument("--out_meta_json", required=True)
    args = ap.parse_args()

    passports_dir = Path(args.passports_dir).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_notes_txt = Path(args.out_notes_txt).resolve()
    out_plot_target_zone_ratio = Path(args.out_plot_target_zone_ratio).resolve()
    out_plot_core_ratio = Path(args.out_plot_core_ratio).resolve()
    out_plot_mean_convergence = Path(args.out_plot_mean_convergence).resolve()
    out_plot_h1_h2_core = Path(args.out_plot_h1_h2_core).resolve()
    out_meta_json = Path(args.out_meta_json).resolve()

    rows = build_rows(passports_dir)

    write_comparative_csv(out_csv, rows)

    plot_metric(
        out_plot_target_zone_ratio,
        rows,
        "target_zone_ratio",
        "Target-zone ratio across full note range",
        "target_zone_ratio",
    )

    plot_metric(
        out_plot_core_ratio,
        rows,
        "core_ratio",
        "Core ratio across full note range",
        "core_ratio",
    )

    plot_metric(
        out_plot_mean_convergence,
        rows,
        "mean_target_convergence_score",
        "Mean target convergence score across full note range",
        "mean_target_convergence_score",
    )

    plot_binary_pair(
        out_plot_h1_h2_core,
        rows,
        "has_h1_core",
        "has_h2_core",
        "Core basis shift: h1 vs h2 across full note range",
        "h1 in core",
        "h2 in core",
    )

    summary_text = build_summary_text(rows)
    write_summary_txt(out_notes_txt, summary_text)

    write_meta_json(
        out_meta_json,
        passports_dir=passports_dir,
        outputs={
            "comparative_csv": str(out_csv),
            "notes_txt": str(out_notes_txt),
            "plot_target_zone_ratio": str(out_plot_target_zone_ratio),
            "plot_core_ratio": str(out_plot_core_ratio),
            "plot_mean_convergence": str(out_plot_mean_convergence),
            "plot_h1_h2_core": str(out_plot_h1_h2_core),
        },
    )

    print("comparative range analysis build complete")
    print(json.dumps(
        {
            "row_count": len(rows),
            "out_csv": str(out_csv),
            "out_notes_txt": str(out_notes_txt),
            "out_plot_target_zone_ratio": str(out_plot_target_zone_ratio),
            "out_plot_core_ratio": str(out_plot_core_ratio),
            "out_plot_mean_convergence": str(out_plot_mean_convergence),
            "out_plot_h1_h2_core": str(out_plot_h1_h2_core),
            "out_meta_json": str(out_meta_json),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()