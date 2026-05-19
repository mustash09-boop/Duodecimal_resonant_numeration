from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from music12.blocks.Block002_audio_recogn.regime_harmonic_confirmation_core import (
    HarmonicRelationCandidate,
    rank_regime_confirmations,
)


# ============================================================
# HELPERS
# ============================================================

def safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def safe_int(v, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def safe_str(v) -> str:
    if v is None:
        return ""
    return str(v).strip()


def count_harmonics_field(value: str) -> int:
    value = safe_str(value)
    if not value:
        return 0
    return len([x for x in value.split() if x.strip()])


def build_chain_support_ratio(support_hits: int, scale: float = 8.0) -> float:
    x = float(support_hits) / float(scale)
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def build_arc_dispersion(spiral_consistency_score: float) -> float:
    sc = safe_float(spiral_consistency_score, 0.0)
    sc = max(0.0, min(1.0, sc))
    return 1.0 - sc


def build_arc_stability_score(
    spiral_consistency_score: float,
    window_chain_match_score: float,
) -> float:
    s = 0.65 * safe_float(spiral_consistency_score, 0.0) + 0.35 * safe_float(window_chain_match_score, 0.0)
    if s < 0.0:
        return 0.0
    if s > 1.0:
        return 1.0
    return s


def load_candidates(path: Path) -> list[HarmonicRelationCandidate]:
    rows: list[HarmonicRelationCandidate] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for r in reader:
            note_token = safe_str(r.get("note_token", ""))
            if not note_token:
                continue

            support_hits = safe_int(r.get("support_hits", ""), 0)
            spiral_consistency_score = safe_float(r.get("spiral_coherence_score", r.get("spiral_consistency_score", "")), 0.0)
            window_chain_match_score = safe_float(r.get("window_chain_match_score", ""), 0.0)

            rows.append(
                HarmonicRelationCandidate(
                    note_token=note_token,

                    spiral_coherence_score=spiral_consistency_score,
                    adaptive_score=safe_float(r.get("adaptive_score", ""), 0.0),

                    arc_stability_score=build_arc_stability_score(
                        spiral_consistency_score=spiral_consistency_score,
                        window_chain_match_score=window_chain_match_score,
                    ),
                    arc_dispersion=build_arc_dispersion(spiral_consistency_score),

                    time_span_start_60=safe_int(r.get("time_span_start_60", r.get("window_start_frame", "")), 0),
                    time_span_end_60=safe_int(r.get("time_span_end_60", r.get("window_end_frame", "")), 0),

                    regime_id=safe_int(r.get("regime_id", ""), 0),
                    regime_name=safe_str(r.get("regime_name", "")),
                    regime_confidence=safe_float(r.get("regime_confidence", ""), 0.0),

                    target_zone_ratio=safe_float(r.get("target_zone_ratio", ""), 0.0),
                    core_ratio=safe_float(r.get("core_ratio", ""), 0.0),
                    mean_target_convergence_score=safe_float(r.get("mean_target_convergence_score", ""), 0.0),

                    chain_support_ratio=build_chain_support_ratio(support_hits),
                    support_hits=support_hits,
                    window_chain_match_score=window_chain_match_score,

                    competing_root_ratio=safe_float(r.get("competing_root_ratio", ""), 0.0),

                    rank=safe_int(r.get("rank", r.get("active_note_rank", "")), 0),
                    source_cluster_id=safe_int(r.get("source_cluster_id", r.get("segment_index", "")), 0),
                )
            )

    return rows


def _pairs_to_json(pairs) -> str:
    payload = []
    for p in pairs:
        payload.append(
            {
                "left_note_token": p.left_note_token,
                "right_note_token": p.right_note_token,
                "interval_semitones": p.interval_semitones,
                "arc_distance": p.arc_distance,
                "time_overlap_ratio": p.time_overlap_ratio,
                "pair_type": p.pair_type,
                "pair_score": p.pair_score,
            }
        )
    return json.dumps(payload, ensure_ascii=False)


def write_confirmed_csv(path: Path, confirmations) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "root_note_token",
        "regime_id",
        "regime_name",
        "confirmation_mode",
        "root_self_score",
        "pair_support_score",
        "structural_support_score",
        "confirmation_score",
        "supporting_pair_count",
        "supporting_pairs_json",
        "explanation",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for c in confirmations:
            writer.writerow(
                {
                    "root_note_token": c.root_note_token,
                    "regime_id": c.regime_id,
                    "regime_name": c.regime_name,
                    "confirmation_mode": c.confirmation_mode,
                    "root_self_score": round(c.root_self_score, 6),
                    "pair_support_score": round(c.pair_support_score, 6),
                    "structural_support_score": round(c.structural_support_score, 6),
                    "confirmation_score": round(c.confirmation_score, 6),
                    "supporting_pair_count": len(c.supporting_pairs),
                    "supporting_pairs_json": _pairs_to_json(c.supporting_pairs),
                    "explanation": c.explanation,
                }
            )


def write_meta_json(
    path: Path,
    *,
    input_csv: Path,
    output_csv: Path,
    candidate_count: int,
    output_count: int,
    max_notes: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "inputs": {
            "input_csv": str(input_csv),
        },
        "outputs": {
            "confirmed_csv": str(output_csv),
            "meta_json": str(path),
        },
        "counts": {
            "input_candidate_count": candidate_count,
            "output_confirmation_count": output_count,
        },
        "params": {
            "max_notes": max_notes,
        },
        "semantic_note": (
            "Regime chain confirmation in spiral/chain logic. "
            "No phase/radial geometry is used. "
            "Confirmation is based on adaptive root score, pair support, structural support, and time overlap."
        ),
    }

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Confirm root candidates using regime-aware chain logic. "
            "This version does not use phase/radial metrics."
        )
    )
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--max_notes", type=int, default=8)

    args = ap.parse_args()

    input_csv = Path(args.input_csv).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_meta_json = Path(args.out_meta_json).resolve()

    candidates = load_candidates(input_csv)
    confirmations = rank_regime_confirmations(
        candidates,
        max_notes=args.max_notes,
    )

    write_confirmed_csv(out_csv, confirmations)
    write_meta_json(
        out_meta_json,
        input_csv=input_csv,
        output_csv=out_csv,
        candidate_count=len(candidates),
        output_count=len(confirmations),
        max_notes=args.max_notes,
    )

    print("regime chain confirmation complete")
    print(json.dumps(
        {
            "input_candidate_count": len(candidates),
            "output_confirmation_count": len(confirmations),
            "max_notes": args.max_notes,
            "out_csv": str(out_csv),
            "out_meta_json": str(out_meta_json),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()