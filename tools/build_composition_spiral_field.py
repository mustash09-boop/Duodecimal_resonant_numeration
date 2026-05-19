from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


def load_csv_by_note(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, dict[str, str]] = {}
    for r in rows:
        note = (r.get("note", "") or r.get("note_token", "") or "").strip()
        if note:
            out[note] = r
    return out


def load_passports(passports_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not passports_dir.exists():
        return out

    for p in passports_dir.glob("*__note_emergence_signature.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            note = str(data.get("target_root_token", "")).strip()
            if note:
                out[note] = data
        except Exception:
            continue
    return out


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class CompositionCandidateRow:
    prefix: str
    source_note: str

    rank: int
    note_token: str
    phase_coherence_score: float
    note_confidence: float
    dominant_harmonics: str
    mean_phase_deg: float
    mean_radial_level: float
    time_span_start_60: int
    time_span_end_60: int
    source_cluster_id: int

    # attached analytical context
    regime_id: int = 0
    regime_name: str = ""
    regime_confidence: float = 0.0

    target_zone_ratio: float = 0.0
    core_ratio: float = 0.0
    mean_target_convergence_score: float = 0.0

    phase_lock_ratio: float = 0.0
    competing_root_ratio: float = 0.0
    phase_distance_std: float = 0.0

    # passport / Block004-like template hints
    passport_core_top3: str = ""
    passport_target_top3: str = ""
    passport_total_top3: str = ""
    strongest_prefix: str = ""
    strongest_sequence: str = ""

    template_match_score: float = 0.0
    composition_field_score: float = 0.0
    composition_role: str = ""


# ============================================================
# LOAD PHASE CANDIDATES
# ============================================================

def load_phase_note_candidates(phase_clusters_dir: Path) -> list[CompositionCandidateRow]:
    rows: list[CompositionCandidateRow] = []

    for p in sorted(phase_clusters_dir.glob("*__phase_note_candidates.csv")):
        prefix = p.name.replace("__phase_note_candidates.csv", "")
        source_note = prefix.split("__")[2] if "__" in prefix else prefix

        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(
                    CompositionCandidateRow(
                        prefix=prefix,
                        source_note=source_note,

                        rank=safe_int(r.get("rank", 0)),
                        note_token=(r.get("note_token", "") or "").strip(),
                        phase_coherence_score=safe_float(r.get("phase_coherence_score", 0.0)),
                        note_confidence=safe_float(r.get("note_confidence", 0.0)),
                        dominant_harmonics=(r.get("dominant_harmonics", "") or "").strip(),
                        mean_phase_deg=safe_float(r.get("mean_phase_deg", 0.0)),
                        mean_radial_level=safe_float(r.get("mean_radial_level", 0.0)),
                        time_span_start_60=safe_int(r.get("time_span_start_60", 0)),
                        time_span_end_60=safe_int(r.get("time_span_end_60", 0)),
                        source_cluster_id=safe_int(r.get("source_cluster_id", 0)),
                    )
                )

    return rows


# ============================================================
# ENRICHMENT
# ============================================================

def enrich_with_regime_and_metrics(
    rows: list[CompositionCandidateRow],
    *,
    range_regime_by_note: dict[str, dict[str, str]],
    comparative_by_note: dict[str, dict[str, str]],
    phase_instability_by_note: dict[str, dict[str, str]],
    passports_by_note: dict[str, dict],
) -> None:
    for row in rows:
        note = row.note_token

        rg = range_regime_by_note.get(note, {})
        cp = comparative_by_note.get(note, {})
        ph = phase_instability_by_note.get(note, {})
        ps = passports_by_note.get(note, {})

        row.regime_id = safe_int(rg.get("regime_id", 0))
        row.regime_name = (rg.get("regime_name", "") or "").strip()
        row.regime_confidence = safe_float(rg.get("regime_confidence", 0.0))

        row.target_zone_ratio = safe_float(cp.get("target_zone_ratio", 0.0))
        row.core_ratio = safe_float(cp.get("core_ratio", 0.0))
        row.mean_target_convergence_score = safe_float(cp.get("mean_target_convergence_score", 0.0))

        row.phase_lock_ratio = safe_float(ph.get("phase_lock_ratio", 0.0))
        row.competing_root_ratio = safe_float(ph.get("competing_root_ratio", 0.0))
        row.phase_distance_std = safe_float(ph.get("phase_distance_std", 0.0))

        harm = ps.get("harmonic_frequency", {}) if isinstance(ps, dict) else {}
        arr = ps.get("harmonic_arrival", {}) if isinstance(ps, dict) else {}

        dom_core = list(harm.get("dominant_core", []) or [])
        dom_target = list(harm.get("dominant_target_zone", []) or [])
        dom_total = list(harm.get("dominant_total", []) or [])
        strongest_prefixes = list(arr.get("strongest_prefixes", []) or [])
        strongest_sequences = list(arr.get("strongest_sequences", []) or [])

        row.passport_core_top3 = " ".join(dom_core[:3])
        row.passport_target_top3 = " ".join(dom_target[:3])
        row.passport_total_top3 = " ".join(dom_total[:3])
        row.strongest_prefix = str(strongest_prefixes[0][0]).strip() if strongest_prefixes else ""
        row.strongest_sequence = str(strongest_sequences[0][0]).strip() if strongest_sequences else ""


def compute_template_match_score(row: CompositionCandidateRow) -> float:
    """
    Comparison with Block004 / Laboratory_research pattern.
    Not a hard truth, only a possible template.
    """
    cluster_h = set((row.dominant_harmonics or "").split())
    passport_h = set((row.passport_core_top3 or row.passport_target_top3 or row.passport_total_top3).split())

    overlap = 0.0
    if passport_h:
        overlap = len(cluster_h.intersection(passport_h)) / len(passport_h)

    score = (
        0.40 * clamp01(row.phase_coherence_score)
        + 0.20 * clamp01(row.regime_confidence)
        + 0.20 * overlap
        + 0.20 * clamp01(row.core_ratio)
    )
    return clamp01(score)


def compute_composition_field_score(row: CompositionCandidateRow) -> float:
    """
    Global composition score:
    - local phase candidate score
    - regime suitability
    - convergence quality
    - anti-ambiguity
    - template similarity to Block004 / laboratory passports
    """
    score = (
        0.25 * clamp01(row.phase_coherence_score)
        + 0.20 * clamp01(row.note_confidence / 10.0)
        + 0.15 * clamp01(row.mean_target_convergence_score / 15.0)
        + 0.10 * clamp01(row.target_zone_ratio)
        + 0.10 * clamp01(row.core_ratio)
        + 0.10 * (1.0 - clamp01(row.competing_root_ratio))
        + 0.10 * clamp01(row.template_match_score)
    )
    return clamp01(score)


def assign_composition_role(row: CompositionCandidateRow) -> str:
    if row.rank == 1 and row.composition_field_score >= 0.70:
        return "local_primary"
    if row.composition_field_score >= 0.55:
        return "strong_context_candidate"
    if row.composition_field_score >= 0.35:
        return "secondary_context_candidate"
    return "weak_context_candidate"


def finalize_rows(rows: list[CompositionCandidateRow]) -> None:
    for row in rows:
        row.template_match_score = compute_template_match_score(row)
        row.composition_field_score = compute_composition_field_score(row)
        row.composition_role = assign_composition_role(row)


# ============================================================
# CSV
# ============================================================

def write_csv_out(path: Path, rows: list[CompositionCandidateRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "prefix",
        "source_note",
        "rank",
        "note_token",
        "phase_coherence_score",
        "note_confidence",
        "dominant_harmonics",
        "mean_phase_deg",
        "mean_radial_level",
        "time_span_start_60",
        "time_span_end_60",
        "source_cluster_id",
        "regime_id",
        "regime_name",
        "regime_confidence",
        "target_zone_ratio",
        "core_ratio",
        "mean_target_convergence_score",
        "phase_lock_ratio",
        "competing_root_ratio",
        "phase_distance_std",
        "passport_core_top3",
        "passport_target_top3",
        "passport_total_top3",
        "strongest_prefix",
        "strongest_sequence",
        "template_match_score",
        "composition_field_score",
        "composition_role",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: getattr(r, k) for k in fieldnames})


# ============================================================
# TXT SUMMARY
# ============================================================

def write_txt_out(path: Path, rows: list[CompositionCandidateRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    by_source: dict[str, list[CompositionCandidateRow]] = defaultdict(list)
    for r in rows:
        by_source[r.source_note].append(r)

    role_counts = Counter(r.composition_role for r in rows)
    regime_counts = Counter(r.regime_name for r in rows if r.regime_name)

    strongest_global = sorted(rows, key=lambda r: r.composition_field_score, reverse=True)[:20]

    with path.open("w", encoding="utf-8") as f:
        f.write("COMPOSITION SPIRAL FIELD ANALYSIS\n")
        f.write("=" * 80 + "\n")
        f.write(f"candidate_row_count: {len(rows)}\n")
        f.write(f"source_note_count: {len(by_source)}\n\n")

        f.write("COMPOSITION ROLE COUNTS\n")
        for k, v in role_counts.most_common():
            f.write(f"  {k}: {v}\n")

        f.write("\nREGIME COUNTS AMONG CANDIDATES\n")
        for k, v in regime_counts.most_common():
            f.write(f"  {k}: {v}\n")

        f.write("\nTOP GLOBAL FIELD CANDIDATES\n")
        for r in strongest_global:
            f.write(
                f"  src={r.source_note} "
                f"rank={r.rank} "
                f"cand={r.note_token} "
                f"field_score={r.composition_field_score:.6f} "
                f"phase_coh={r.phase_coherence_score:.6f} "
                f"regime={r.regime_name} "
                f"template={r.template_match_score:.6f} "
                f"harm={r.dominant_harmonics}\n"
            )

        f.write("\nINTERPRETATION\n")
        f.write("  - local phase clusters are re-read inside one composition field.\n")
        f.write("  - comparative / regime / phase layers act as context, not as rigid truth.\n")
        f.write("  - Block004 / Laboratory_research signatures are used as possible templates only.\n")
        f.write("  - the same principle may appear with different numeric values in real compositions.\n")


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
            "Build composition spiral field from local phase clusters and compare them "
            "with comparative/range/phase layers and Block004-like laboratory passports."
        )
    )
    ap.add_argument("--phase_clusters_dir", required=True)
    ap.add_argument("--range_regime_csv", required=True)
    ap.add_argument("--comparative_csv", required=True)
    ap.add_argument("--phase_instability_csv", required=True)
    ap.add_argument("--passports_dir", required=True)

    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_txt", required=True)
    ap.add_argument("--out_meta_json", required=True)
    args = ap.parse_args()

    phase_clusters_dir = Path(args.phase_clusters_dir).resolve()
    range_regime_csv = Path(args.range_regime_csv).resolve()
    comparative_csv = Path(args.comparative_csv).resolve()
    phase_instability_csv = Path(args.phase_instability_csv).resolve()
    passports_dir = Path(args.passports_dir).resolve()

    out_csv = Path(args.out_csv).resolve()
    out_txt = Path(args.out_txt).resolve()
    out_meta_json = Path(args.out_meta_json).resolve()

    rows = load_phase_note_candidates(phase_clusters_dir)

    enrich_with_regime_and_metrics(
        rows,
        range_regime_by_note=load_csv_by_note(range_regime_csv),
        comparative_by_note=load_csv_by_note(comparative_csv),
        phase_instability_by_note=load_csv_by_note(phase_instability_csv),
        passports_by_note=load_passports(passports_dir),
    )
    finalize_rows(rows)

    write_csv_out(out_csv, rows)
    write_txt_out(out_txt, rows)
    write_meta_json(
        out_meta_json,
        inputs={
            "phase_clusters_dir": str(phase_clusters_dir),
            "range_regime_csv": str(range_regime_csv),
            "comparative_csv": str(comparative_csv),
            "phase_instability_csv": str(phase_instability_csv),
            "passports_dir": str(passports_dir),
        },
        outputs={
            "composition_spiral_field_csv": str(out_csv),
            "composition_spiral_field_txt": str(out_txt),
        },
    )

    print("composition spiral field build complete")
    print(json.dumps(
        {
            "row_count": len(rows),
            "out_csv": str(out_csv),
            "out_txt": str(out_txt),
            "out_meta_json": str(out_meta_json),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()