from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from music12.blocks.Block002_audio_recogn.adaptive_root_selection_core import (
    AdaptiveRootCandidate,
    rank_adaptive_root_candidates,
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
    """
    Convert consistency into inverse-like dispersion proxy.
    High consistency -> low dispersion.
    """
    sc = safe_float(spiral_consistency_score, 0.0)
    sc = max(0.0, min(1.0, sc))
    return 1.0 - sc


def build_arc_stability_score(
    spiral_consistency_score: float,
    window_chain_match_score: float,
) -> float:
    """
    Lightweight stability proxy from already available chain-window metrics.
    """
    s = 0.65 * safe_float(spiral_consistency_score, 0.0) + 0.35 * safe_float(window_chain_match_score, 0.0)
    if s < 0.0:
        return 0.0
    if s > 1.0:
        return 1.0
    return s


def load_candidates(path: Path) -> list[AdaptiveRootCandidate]:
    rows: list[AdaptiveRootCandidate] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for r in reader:
            note_token = safe_str(r.get("best_theoretical_root_token", ""))
            if not note_token:
                continue

            support_hits = safe_int(r.get("support_hits", ""), 0)
            spiral_consistency_score = safe_float(r.get("spiral_consistency_score", ""), 0.0)
            window_chain_match_score = safe_float(r.get("window_chain_match_score", ""), 0.0)

            rows.append(
                AdaptiveRootCandidate(
                    note_token=note_token,

                    raw_note_confidence=safe_float(r.get("rc_chain_score", ""), 0.0),
                    spiral_coherence_score=spiral_consistency_score,
                    arc_stability_score=build_arc_stability_score(
                        spiral_consistency_score=spiral_consistency_score,
                        window_chain_match_score=window_chain_match_score,
                    ),
                    arc_dispersion=build_arc_dispersion(spiral_consistency_score),

                    chain_support_ratio=build_chain_support_ratio(support_hits),
                    support_hits=support_hits,
                    window_chain_match_score=window_chain_match_score,

                    regime_id=safe_int(r.get("regime_id", ""), 0),
                    regime_name=safe_str(r.get("regime_name", "")),
                    regime_confidence=safe_float(r.get("regime_confidence", ""), 0.0),

                    target_zone_ratio=safe_float(r.get("target_zone_ratio", ""), 0.0),
                    core_ratio=safe_float(r.get("core_ratio", ""), 0.0),
                    mean_target_convergence_score=safe_float(r.get("mean_target_convergence_score", ""), 0.0),

                    competing_root_ratio=safe_float(r.get("competing_root_ratio", ""), 0.0),

                    template_match_score=safe_float(r.get("template_match_score", ""), 0.0),
                    composition_field_score=safe_float(r.get("composition_field_score", ""), 0.0),

                    rank=safe_int(r.get("active_note_rank", ""), 0),
                    source_cluster_id=safe_int(r.get("segment_index", ""), 0),
                )
            )

    return rows


def write_ranked_csv(path: Path, ranked) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "note_token",
        "adaptive_score",

        "regime_id",
        "regime_name",

        "weight_core",
        "weight_convergence",
        "weight_spiral",
        "weight_context",

        "core_term",
        "convergence_term",
        "spiral_term",
        "context_term",

        "raw_note_confidence",
        "spiral_coherence_score",
        "arc_stability_score",
        "arc_dispersion",
        "chain_support_ratio",
        "support_hits",
        "target_zone_ratio",
        "core_ratio",
        "mean_target_convergence_score",
        "window_chain_match_score",
        "competing_root_ratio",

        "rank",
        "source_cluster_id",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for d in ranked:
            writer.writerow(
                {
                    "note_token": d.note_token,
                    "adaptive_score": round(d.adaptive_score, 6),

                    "regime_id": d.regime_id,
                    "regime_name": d.regime_name,

                    "weight_core": round(d.weight_core, 6),
                    "weight_convergence": round(d.weight_convergence, 6),
                    "weight_spiral": round(d.weight_spiral, 6),
                    "weight_context": round(d.weight_context, 6),

                    "core_term": round(d.core_term, 6),
                    "convergence_term": round(d.convergence_term, 6),
                    "spiral_term": round(d.spiral_term, 6),
                    "context_term": round(d.context_term, 6),

                    "raw_note_confidence": round(d.raw_note_confidence, 6),
                    "spiral_coherence_score": round(d.spiral_coherence_score, 6),
                    "arc_stability_score": round(d.arc_stability_score, 6),
                    "arc_dispersion": round(d.arc_dispersion, 6),
                    "chain_support_ratio": round(d.chain_support_ratio, 6),
                    "support_hits": d.support_hits,
                    "target_zone_ratio": round(d.target_zone_ratio, 6),
                    "core_ratio": round(d.core_ratio, 6),
                    "mean_target_convergence_score": round(d.mean_target_convergence_score, 6),
                    "window_chain_match_score": round(d.window_chain_match_score, 6),
                    "competing_root_ratio": round(d.competing_root_ratio, 6),

                    "rank": d.rank,
                    "source_cluster_id": d.source_cluster_id,
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
            "ranked_csv": str(output_csv),
            "meta_json": str(path),
        },
        "counts": {
            "input_candidate_count": candidate_count,
            "output_ranked_count": output_count,
        },
        "params": {
            "max_notes": max_notes,
        },
        "semantic_note": (
            "Adaptive root ranking in spiral/chain logic. "
            "No phase/radial geometry is used. "
            "Ranking is based on chain support, spiral consistency, convergence, and context."
        ),
    }

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Rank root candidates using spiral/chain logic. "
            "This version does not use mean phase or mean radial metrics."
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
    ranked = rank_adaptive_root_candidates(
        candidates,
        max_notes=args.max_notes,
    )

    write_ranked_csv(out_csv, ranked)
    write_meta_json(
        out_meta_json,
        input_csv=input_csv,
        output_csv=out_csv,
        candidate_count=len(candidates),
        output_count=len(ranked),
        max_notes=args.max_notes,
    )

    print("adaptive spiral root ranking complete")
    print(json.dumps(
        {
            "input_candidate_count": len(candidates),
            "output_ranked_count": len(ranked),
            "max_notes": args.max_notes,
            "out_csv": str(out_csv),
            "out_meta_json": str(out_meta_json),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()