from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt


# ============================================================
# DUODECIMAL HELPERS
# ============================================================

DUO_MAP = {
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
    "7": 7, "8": 8, "9": 9, "A": 10, "B": 11, "C": 12,
}


def duo_str_to_int(s: str) -> int:
    s = (s or "").strip().upper()
    if not s:
        raise ValueError("Empty duodecimal string")
    value = 0
    for ch in s:
        if ch not in DUO_MAP:
            raise ValueError(f"Unsupported duodecimal digit: {ch}")
        value = value * 12 + DUO_MAP[ch]
    return value


def parse_note_token_sort_key(token: str) -> tuple[int, int, str]:
    token = (token or "").strip().replace(" ", "")
    if "." not in token:
        return (999999, 999999, token)

    left, right = token.split(".", 1)

    octave_part = ""
    degree_part = ""

    for ch in left:
        ch = ch.upper()
        if ch in DUO_MAP:
            octave_part += ch
        else:
            break

    for ch in right:
        ch = ch.upper()
        if ch in DUO_MAP:
            degree_part += ch
        else:
            break

    if not octave_part or not degree_part:
        return (999999, 999999, token)

    return (duo_str_to_int(octave_part), duo_str_to_int(degree_part), token)


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


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-12:
        return [0.5 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class RegimeRow:
    note: str
    note_index: int = 0

    target_zone_ratio: float = 0.0
    core_ratio: float = 0.0
    mean_target_convergence_score: float = 0.0

    phase_lock_ratio: float = 0.0
    competing_root_ratio: float = 0.0
    phase_distance_std: float = 0.0
    phase_instability_score: float = 0.0

    has_h1_core: int = 0
    has_h2_core: int = 0

    # normalized features
    n_target_zone_ratio: float = 0.0
    n_core_ratio: float = 0.0
    n_mean_target_convergence_score: float = 0.0
    n_phase_lock_ratio: float = 0.0
    n_competing_root_ratio: float = 0.0
    n_phase_distance_std: float = 0.0
    n_phase_instability_score: float = 0.0

    # derived regime logic
    stability_score: float = 0.0
    ambiguity_score: float = 0.0
    transition_score: float = 0.0

    regime_id: int = 0
    regime_name: str = ""
    regime_confidence: float = 0.0


# ============================================================
# LOADERS
# ============================================================

def load_csv_by_note(path: Path) -> Dict[str, dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out: Dict[str, dict] = {}
    for r in rows:
        note = (r.get("note", "") or "").strip()
        if note:
            out[note] = r
    return out


def build_rows(
    comparative_csv: Path,
    harmonic_loss_csv: Path,
    phase_instability_csv: Path,
) -> List[RegimeRow]:
    comp = load_csv_by_note(comparative_csv)
    harm = load_csv_by_note(harmonic_loss_csv)
    phase = load_csv_by_note(phase_instability_csv)

    all_notes = sorted(
        set(comp.keys()) | set(harm.keys()) | set(phase.keys()),
        key=parse_note_token_sort_key,
    )

    rows: List[RegimeRow] = []

    for i, note in enumerate(all_notes, start=1):
        c = comp.get(note, {})
        h = harm.get(note, {})
        p = phase.get(note, {})

        row = RegimeRow(
            note=note,
            note_index=i,

            target_zone_ratio=safe_float(c.get("target_zone_ratio", 0.0)),
            core_ratio=safe_float(c.get("core_ratio", 0.0)),
            mean_target_convergence_score=safe_float(c.get("mean_target_convergence_score", 0.0)),

            phase_lock_ratio=safe_float(p.get("phase_lock_ratio", 0.0)),
            competing_root_ratio=safe_float(p.get("competing_root_ratio", 0.0)),
            phase_distance_std=safe_float(p.get("phase_distance_std", 0.0)),
            phase_instability_score=safe_float(p.get("phase_instability_score", 0.0)),

            has_h1_core=safe_int(h.get("core_has_h1", 0)),
            has_h2_core=safe_int(h.get("core_has_h2", 0)),
        )
        rows.append(row)

    return rows


# ============================================================
# FEATURE ENRICHMENT
# ============================================================

def enrich_normalized(rows: List[RegimeRow]) -> None:
    tz = normalize([r.target_zone_ratio for r in rows])
    cr = normalize([r.core_ratio for r in rows])
    cv = normalize([r.mean_target_convergence_score for r in rows])
    pl = normalize([r.phase_lock_ratio for r in rows])
    cp = normalize([r.competing_root_ratio for r in rows])
    ps = normalize([r.phase_distance_std for r in rows])
    pi = normalize([r.phase_instability_score for r in rows])

    for r, a, b, c, d, e, f, g in zip(rows, tz, cr, cv, pl, cp, ps, pi):
        r.n_target_zone_ratio = a
        r.n_core_ratio = b
        r.n_mean_target_convergence_score = c
        r.n_phase_lock_ratio = d
        r.n_competing_root_ratio = e
        r.n_phase_distance_std = f
        r.n_phase_instability_score = g


def enrich_regime_scores(rows: List[RegimeRow]) -> None:
    for r in rows:
        # Stable center = target zone + core + convergence + phase lock
        r.stability_score = (
            0.28 * r.n_target_zone_ratio
            + 0.28 * r.n_core_ratio
            + 0.24 * r.n_mean_target_convergence_score
            + 0.20 * r.n_phase_lock_ratio
        )

        # Ambiguity = competing root + phase instability + phase std
        r.ambiguity_score = (
            0.45 * r.n_competing_root_ratio
            + 0.35 * r.n_phase_instability_score
            + 0.20 * r.n_phase_distance_std
        )

    # transition score = local change intensity across neighbors
    for i, r in enumerate(rows):
        if i == 0 or i == len(rows) - 1:
            r.transition_score = 0.0
            continue

        prev_r = rows[i - 1]
        next_r = rows[i + 1]

        ds = abs(next_r.stability_score - prev_r.stability_score)
        da = abs(next_r.ambiguity_score - prev_r.ambiguity_score)
        dc = abs(next_r.n_core_ratio - prev_r.n_core_ratio)
        dt = abs(next_r.n_target_zone_ratio - prev_r.n_target_zone_ratio)
        dp = abs(next_r.n_phase_lock_ratio - prev_r.n_phase_lock_ratio)

        r.transition_score = (ds + da + dc + dt + dp) / 5.0


# ============================================================
# REGIME ASSIGNMENT
# ============================================================

def assign_regimes(rows: List[RegimeRow]) -> None:
    """
    Data-driven but still interpretable.
    We use stability / ambiguity / transition jointly.

    regime 1: low spiral anchoring zone
    regime 2: rising organized zone
    regime 3: resonance plateau / central stable zone
    regime 4: transitional upper reorganization zone
    regime 5: high unstable / decentered zone
    """
    if not rows:
        return

    for r in rows:
        s = r.stability_score
        a = r.ambiguity_score
        t = r.transition_score

        if s >= 0.72 and a <= 0.30:
            r.regime_id = 3
            r.regime_name = "central_resonance_plateau"
            r.regime_confidence = clamp01(0.65 + 0.35 * s - 0.20 * a)

        elif s >= 0.55 and a <= 0.45:
            r.regime_id = 2
            r.regime_name = "rising_stable_zone"
            r.regime_confidence = clamp01(0.55 + 0.30 * s - 0.15 * a)

        elif s <= 0.22 and a >= 0.62:
            r.regime_id = 5
            r.regime_name = "high_decentered_zone"
            r.regime_confidence = clamp01(0.60 + 0.30 * a - 0.15 * s)

        elif t >= 0.18 or (0.35 <= s <= 0.65 and 0.35 <= a <= 0.65):
            r.regime_id = 4
            r.regime_name = "transition_reorganization_zone"
            r.regime_confidence = clamp01(0.45 + 0.35 * t + 0.10 * a)

        else:
            r.regime_id = 1
            r.regime_name = "low_anchor_zone"
            r.regime_confidence = clamp01(0.50 + 0.20 * s + 0.10 * (1.0 - a))

    # smooth tiny one-note outliers
    for i in range(1, len(rows) - 1):
        a = rows[i - 1].regime_id
        b = rows[i].regime_id
        c = rows[i + 1].regime_id
        if a == c and b != a and rows[i].regime_confidence < 0.62:
            rows[i].regime_id = a
            rows[i].regime_name = rows[i - 1].regime_name
            rows[i].regime_confidence = min(0.75, rows[i].regime_confidence + 0.08)


# ============================================================
# WRITE CSV
# ============================================================

def write_csv_out(path: Path, rows: List[RegimeRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "note",
        "note_index",
        "target_zone_ratio",
        "core_ratio",
        "mean_target_convergence_score",
        "phase_lock_ratio",
        "competing_root_ratio",
        "phase_distance_std",
        "phase_instability_score",
        "has_h1_core",
        "has_h2_core",
        "stability_score",
        "ambiguity_score",
        "transition_score",
        "regime_id",
        "regime_name",
        "regime_confidence",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: getattr(r, k) for k in fieldnames})


# ============================================================
# WRITE TXT
# ============================================================

def write_txt_out(path: Path, rows: List[RegimeRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    regime_counts: Dict[str, int] = {}
    for r in rows:
        regime_counts[r.regime_name] = regime_counts.get(r.regime_name, 0) + 1

    strongest_stability = max(rows, key=lambda r: r.stability_score)
    strongest_transition = max(rows, key=lambda r: r.transition_score)
    strongest_ambiguity = max(rows, key=lambda r: r.ambiguity_score)

    with path.open("w", encoding="utf-8") as f:
        f.write("RANGE REGIME ANALYSIS\n")
        f.write("=" * 80 + "\n")
        f.write(f"note_count: {len(rows)}\n\n")

        f.write("REGIME COUNTS\n")
        for k, v in sorted(regime_counts.items()):
            f.write(f"  {k}: {v}\n")

        f.write("\nGLOBAL EXTREMA\n")
        f.write(f"  strongest_stability: {strongest_stability.note} -> {strongest_stability.stability_score:.6f}\n")
        f.write(f"  strongest_transition: {strongest_transition.note} -> {strongest_transition.transition_score:.6f}\n")
        f.write(f"  strongest_ambiguity: {strongest_ambiguity.note} -> {strongest_ambiguity.ambiguity_score:.6f}\n")

        f.write("\nREGIME SPANS\n")
        start = 0
        while start < len(rows):
            rid = rows[start].regime_id
            rname = rows[start].regime_name
            end = start
            while end + 1 < len(rows) and rows[end + 1].regime_id == rid:
                end += 1
            f.write(
                f"  regime {rid} ({rname}): "
                f"{rows[start].note} -> {rows[end].note} "
                f"[count={end - start + 1}]\n"
            )
            start = end + 1

        f.write("\nINTERPRETATION\n")
        f.write("  - range is not one law but several spiral regimes.\n")
        f.write("  - transition_score marks not a point switch but coupled parameter reorganization.\n")
        f.write("  - root selection and note confidence should later depend on regime_id.\n")
        f.write("  - for piano this gives regime rules before moving to other instruments.\n")


# ============================================================
# PLOTS
# ============================================================

def plot_regime_scores(path: Path, rows: List[RegimeRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    xs = [r.note_index for r in rows]
    labels = [r.note for r in rows]
    y1 = [r.stability_score for r in rows]
    y2 = [r.ambiguity_score for r in rows]
    y3 = [r.transition_score for r in rows]

    fig = plt.figure(figsize=(16, 5))
    ax = fig.add_subplot(111)
    ax.plot(xs, y1, label="stability_score")
    ax.plot(xs, y2, label="ambiguity_score")
    ax.plot(xs, y3, label="transition_score")
    ax.set_title("Range regime scores across full note range")
    ax.set_xlabel("note_index")
    ax.set_ylabel("score")
    ax.grid(True, alpha=0.3)
    ax.legend()

    tick_step = max(1, len(xs) // 16)
    ax.set_xticks(xs[::tick_step])
    ax.set_xticklabels(labels[::tick_step], rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_regime_ids(path: Path, rows: List[RegimeRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    xs = [r.note_index for r in rows]
    labels = [r.note for r in rows]
    ys = [r.regime_id for r in rows]

    fig = plt.figure(figsize=(16, 5))
    ax = fig.add_subplot(111)
    ax.step(xs, ys, where="mid")
    ax.set_title("Range regimes across full note range")
    ax.set_xlabel("note_index")
    ax.set_ylabel("regime_id")
    ax.grid(True, alpha=0.3)

    tick_step = max(1, len(xs) // 16)
    ax.set_xticks(xs[::tick_step])
    ax.set_xticklabels(labels[::tick_step], rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ============================================================
# META
# ============================================================

def write_meta_json(path: Path, *, inputs: dict, outputs: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "inputs": inputs,
                "outputs": outputs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build range regime analysis from comparative / harmonic / phase layers. "
            "Detects multiple spiral regimes and transition zones across the note range."
        )
    )
    ap.add_argument("--comparative_csv", required=True)
    ap.add_argument("--harmonic_loss_csv", required=True)
    ap.add_argument("--phase_instability_csv", required=True)

    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_txt", required=True)
    ap.add_argument("--out_plot_scores_png", required=True)
    ap.add_argument("--out_plot_regimes_png", required=True)
    ap.add_argument("--out_meta_json", required=True)
    args = ap.parse_args()

    comparative_csv = Path(args.comparative_csv).resolve()
    harmonic_loss_csv = Path(args.harmonic_loss_csv).resolve()
    phase_instability_csv = Path(args.phase_instability_csv).resolve()

    out_csv = Path(args.out_csv).resolve()
    out_txt = Path(args.out_txt).resolve()
    out_plot_scores_png = Path(args.out_plot_scores_png).resolve()
    out_plot_regimes_png = Path(args.out_plot_regimes_png).resolve()
    out_meta_json = Path(args.out_meta_json).resolve()

    rows = build_rows(
        comparative_csv=comparative_csv,
        harmonic_loss_csv=harmonic_loss_csv,
        phase_instability_csv=phase_instability_csv,
    )
    enrich_normalized(rows)
    enrich_regime_scores(rows)
    assign_regimes(rows)

    write_csv_out(out_csv, rows)
    write_txt_out(out_txt, rows)
    plot_regime_scores(out_plot_scores_png, rows)
    plot_regime_ids(out_plot_regimes_png, rows)

    write_meta_json(
        out_meta_json,
        inputs={
            "comparative_csv": str(comparative_csv),
            "harmonic_loss_csv": str(harmonic_loss_csv),
            "phase_instability_csv": str(phase_instability_csv),
        },
        outputs={
            "range_regime_csv": str(out_csv),
            "range_regime_txt": str(out_txt),
            "plot_scores_png": str(out_plot_scores_png),
            "plot_regimes_png": str(out_plot_regimes_png),
        },
    )

    print("range regime analysis build complete")
    print(json.dumps(
        {
            "row_count": len(rows),
            "out_csv": str(out_csv),
            "out_txt": str(out_txt),
            "out_plot_scores_png": str(out_plot_scores_png),
            "out_plot_regimes_png": str(out_plot_regimes_png),
            "out_meta_json": str(out_meta_json),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()