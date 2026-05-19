# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set


# ============================================================
# Safe helpers
# ============================================================

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ============================================================
# Token helpers
# ============================================================

def _split_token_micro(token: str) -> tuple[str, str]:
    token = str(token or "").strip()
    if "'" not in token:
        return token, ""
    coarse, micro = token.split("'", 1)
    return coarse, micro or "-"


def _token_coarse(token: str) -> str:
    coarse, _micro = _split_token_micro(token)
    return coarse


def _tokens(raw: str) -> Set[str]:
    return {x.strip() for x in str(raw or "").split() if x.strip()}


def _tokens_json_or_space(raw_json: str, raw_space: str = "") -> Set[str]:
    raw_json = str(raw_json or "").strip()
    if raw_json:
        try:
            data = json.loads(raw_json)
            if isinstance(data, list):
                return {str(x).strip() for x in data if str(x).strip()}
        except Exception:
            pass
    return _tokens(raw_space)


def _row_micro_tokens(row: Dict[str, Any]) -> Set[str]:
    micro = _tokens_json_or_space(
        str(row.get("token_union_micro_json", "")),
        str(row.get("token_union_micro", "")),
    )
    if micro:
        return micro
    return _tokens(row.get("token_union", ""))


def _row_coarse_tokens(row: Dict[str, Any]) -> Set[str]:
    coarse = _tokens_json_or_space(
        str(row.get("token_union_coarse_json", "")),
        str(row.get("token_union_coarse", "")),
    )
    if coarse:
        return coarse
    return {_token_coarse(t) for t in _row_micro_tokens(row)}


def _row_roots_micro(row: Dict[str, Any]) -> Set[str]:
    raw = str(row.get("observed_roots_micro", "")).strip()
    if raw:
        return _tokens(raw)
    return _tokens(row.get("observed_roots", ""))


def _row_roots_coarse(row: Dict[str, Any]) -> Set[str]:
    raw = str(row.get("observed_roots_coarse", "")).strip()
    if raw:
        return _tokens(raw)
    return {_token_coarse(t) for t in _row_roots_micro(row)}


def _root_hint_micro(row: Dict[str, Any]) -> str:
    return (
        str(row.get("root_hint_micro_not_identity", "")).strip()
        or str(row.get("root_hint_not_identity", "")).strip()
    )


def _root_hint_coarse(row: Dict[str, Any]) -> str:
    return (
        str(row.get("root_hint_coarse_not_identity", "")).strip()
        or (_token_coarse(_root_hint_micro(row)) if _root_hint_micro(row) else "")
    )


# ============================================================
# Entity / similarity
# ============================================================

def _entity_id(row: Dict[str, Any]) -> str:
    return str(
        row.get(
            "ecology_entity_id",
            row.get(
                "trajectory_entity_id",
                row.get("stable_entity_id", row.get("entity_id", "")),
            ),
        )
    ).strip()


def _entity_signature(row: Dict[str, Any]) -> Dict[str, Any]:
    micro_tokens = _row_micro_tokens(row)
    coarse_tokens = _row_coarse_tokens(row)
    roots_micro = _row_roots_micro(row)
    roots_coarse = _row_roots_coarse(row)

    return {
        "id": _entity_id(row),
        "birth_frame": _safe_int(row.get("birth_frame"), 0),
        "end_frame": _safe_int(row.get("end_frame"), 0),
        "duration_frames": _safe_int(row.get("duration_frames"), 0),
        "frame_count": _safe_int(row.get("frame_count"), 0),

        "micro_tokens": micro_tokens,
        "coarse_tokens": coarse_tokens,

        # Backward-compatible alias.
        "tokens": micro_tokens,

        "micro_count": len(micro_tokens),
        "coarse_count": len(coarse_tokens),
        "micro_preview": " ".join(sorted(micro_tokens)[:24]),
        "coarse_preview": " ".join(sorted(coarse_tokens)[:24]),

        "roots_micro": roots_micro,
        "roots_coarse": roots_coarse,

        # Backward-compatible alias.
        "roots": roots_micro,

        "signatures": _tokens(row.get("topology_signatures", "")),

        "mean_score": _safe_float(row.get("mean_family_score"), 0.0),
        "max_score": _safe_float(row.get("max_family_score"), 0.0),

        "coherence": _safe_float(row.get("mean_topology_coherence"), 0.0),
        "micro_coherence": _safe_float(row.get("mean_micro_topology_coherence"), 0.0),
        "coarse_coherence": _safe_float(row.get("mean_coarse_topology_coherence"), 0.0),

        "pairwise_topology_coherence": max(
            _safe_float(row.get("ecology_pairwise_topology_coherence"), 0.0),
            _safe_float(row.get("trajectory_pairwise_topology_coherence"), 0.0),
            _safe_float(row.get("group_pairwise_topology_coherence"), 0.0),
        ),

        "root_hint_micro_not_identity": _root_hint_micro(row),
        "root_hint_coarse_not_identity": _root_hint_coarse(row),
    }


def _time_overlap(a: Dict[str, Any], b: Dict[str, Any]) -> int:
    start = max(a["birth_frame"], b["birth_frame"])
    end = min(a["end_frame"], b["end_frame"])
    return max(0, end - start + 1)


def _overlap_ratio(overlap_frames: int, a: Dict[str, Any], b: Dict[str, Any]) -> float:
    denom = max(min(a.get("duration_frames", 0), b.get("duration_frames", 0)), 1)
    return min(float(overlap_frames) / float(denom), 1.0)


def _containment_similarity(a: Set[str], b: Set[str]) -> float:
    """
    Direction-friendly overlap:
        common / min(len(a), len(b))

    This is useful for influence because a source can contain only a subset
    of a larger resonance field and still be causally relevant.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / max(min(len(a), len(b)), 1)


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def _structured_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, float]:
    micro_token = _containment_similarity(a["micro_tokens"], b["micro_tokens"])
    coarse_token = _containment_similarity(a["coarse_tokens"], b["coarse_tokens"])

    micro_root = _containment_similarity(a["roots_micro"], b["roots_micro"])
    coarse_root = _containment_similarity(a["roots_coarse"], b["roots_coarse"])

    # topology_signatures are already compressed identifiers, so Jaccard is safer.
    micro_topology = _jaccard(a["signatures"], b["signatures"])
    coarse_topology = micro_topology

    token_similarity = 0.70 * micro_token + 0.30 * coarse_token
    root_similarity = 0.70 * micro_root + 0.30 * coarse_root
    topology_similarity = 0.70 * micro_topology + 0.30 * coarse_topology

    direct_topology_support = (
        token_similarity * 0.34
        + root_similarity * 0.16
        + topology_similarity * 0.50
    )

    return {
        "micro_token_similarity": micro_token,
        "coarse_token_similarity": coarse_token,
        "token_similarity": token_similarity,

        "micro_root_similarity": micro_root,
        "coarse_root_similarity": coarse_root,
        "root_similarity": root_similarity,

        "micro_topology_similarity": micro_topology,
        "coarse_topology_similarity": coarse_topology,
        "topology_similarity": topology_similarity,

        "direct_topology_support": direct_topology_support,
    }


def _confidence_basis(
    *,
    sim: Dict[str, float],
    overlap_frames: int,
    overlap_ratio: float,
    temporal_precedence: float,
    strength_advantage: float,
) -> str:
    parts = []

    if sim["micro_token_similarity"] > 0.0:
        parts.append("micro_token")
    if sim["coarse_token_similarity"] > 0.0:
        parts.append("coarse_token")
    if sim["micro_root_similarity"] > 0.0:
        parts.append("micro_root")
    if sim["topology_similarity"] > 0.0:
        parts.append("topology_signature")
    if overlap_frames > 0:
        parts.append("time_overlap")
    if overlap_ratio >= 0.35:
        parts.append("sustained_overlap")
    if temporal_precedence > 0.0:
        parts.append("temporal_precedence")
    if strength_advantage > 0.0:
        parts.append("source_strength_advantage")

    return "+".join(parts) if parts else "weak_or_missing_basis"


def _directional_influence(
    a: Dict[str, Any],
    b: Dict[str, Any],
    *,
    min_direct_topology: float,
    min_micro_token_similarity: float,
    min_coarse_token_similarity: float,
) -> Dict[str, Any]:
    overlap_frames = _time_overlap(a, b)
    if overlap_frames <= 0:
        return {
            "score_ab": 0.0,
            "score_ba": 0.0,
            "rejected_reason": "no_time_overlap",
        }

    sim = _structured_similarity(a, b)

    # Direct topology continuity is mandatory.
    # Coarse can support, but cannot fully replace micro identity.
    if sim["direct_topology_support"] < min_direct_topology:
        return {
            "score_ab": 0.0,
            "score_ba": 0.0,
            "rejected_reason": "low_direct_topology_support",
            **sim,
            "overlap_frames": overlap_frames,
        }

    if (
        sim["micro_token_similarity"] < min_micro_token_similarity
        and sim["coarse_token_similarity"] < min_coarse_token_similarity
    ):
        return {
            "score_ab": 0.0,
            "score_ba": 0.0,
            "rejected_reason": "low_micro_and_coarse_similarity",
            **sim,
            "overlap_frames": overlap_frames,
        }

    overlap_ratio = _overlap_ratio(overlap_frames, a, b)

    migration_support = (
        sim["micro_root_similarity"] * 0.35
        + sim["coarse_root_similarity"] * 0.25
        + sim["topology_similarity"] * 0.25
        + overlap_ratio * 0.15
    )

    if (
        sim["micro_token_similarity"] <= 0.0
        and sim["coarse_token_similarity"] <= 0.0
        and sim["topology_similarity"] >= 0.85
        and migration_support < 0.32
    ):
        return {
            "score_ab": 0.0,
            "score_ba": 0.0,
            "rejected_reason": "topology_tags_without_migration_support",
            **sim,
            "migration_support": migration_support,
            "overlap_frames": overlap_frames,
        }


    # Temporal precedence
    a_precedes = 1.0 if a["birth_frame"] <= b["birth_frame"] else 0.0
    b_precedes = 1.0 if b["birth_frame"] <= a["birth_frame"] else 0.0

    # Stronger/stabler entities more likely primary, but this must not dominate topology.
    a_strength = (
        a["mean_score"] * 0.45
        + a["coherence"] * 0.25
        + a["micro_coherence"] * 0.15
        + a["pairwise_topology_coherence"] * 0.15
    )

    b_strength = (
        b["mean_score"] * 0.45
        + b["coherence"] * 0.25
        + b["micro_coherence"] * 0.15
        + b["pairwise_topology_coherence"] * 0.15
    )

    a_strength_advantage = max(a_strength - b_strength, 0.0)
    b_strength_advantage = max(b_strength - a_strength, 0.0)

    overlap_strength = sim["direct_topology_support"]

    # Influence score: topology-first, then time, then strength.
    score_ab = 0.0
    score_ab += overlap_strength * 0.56
    score_ab += sim["micro_token_similarity"] * 0.10
    score_ab += a_precedes * 0.10
    score_ab += min(a_strength_advantage, 2.0) * 0.10
    score_ab += overlap_ratio * 0.14

    score_ba = 0.0
    score_ba += overlap_strength * 0.56
    score_ba += sim["micro_token_similarity"] * 0.10
    score_ba += b_precedes * 0.10
    score_ba += min(b_strength_advantage, 2.0) * 0.10
    score_ba += overlap_ratio * 0.14

    basis_ab = _confidence_basis(
        sim=sim,
        overlap_frames=overlap_frames,
        overlap_ratio=overlap_ratio,
        temporal_precedence=a_precedes,
        strength_advantage=a_strength_advantage,
    )

    basis_ba = _confidence_basis(
        sim=sim,
        overlap_frames=overlap_frames,
        overlap_ratio=overlap_ratio,
        temporal_precedence=b_precedes,
        strength_advantage=b_strength_advantage,
    )

    return {
        "score_ab": score_ab,
        "score_ba": score_ba,

        **sim,

        "overlap_frames": overlap_frames,
        "overlap_ratio": overlap_ratio,

        "a_strength": a_strength,
        "b_strength": b_strength,

        "a_strength_advantage": a_strength_advantage,
        "b_strength_advantage": b_strength_advantage,

        "a_precedes": a_precedes,
        "b_precedes": b_precedes,

        "confidence_basis_ab": basis_ab,
        "confidence_basis_ba": basis_ba,

        "rejected_reason": "",
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build directional resonance influence graph between acoustic entities "
            "with micro/coarse topology preserved."
        )
    )

    ap.add_argument("--ecology_entities_csv", required=True)

    ap.add_argument("--out_influence_graph_csv", required=True)
    ap.add_argument("--out_primary_secondary_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--min_influence_score", type=float, default=0.24)

    # Anti-collapse gates.
    ap.add_argument("--min_direct_topology", type=float, default=0.05)
    ap.add_argument("--min_micro_token_similarity", type=float, default=0.00)
    ap.add_argument("--min_coarse_token_similarity", type=float, default=0.06)

    args = ap.parse_args()

    rows = _load_csv(Path(args.ecology_entities_csv))
    sigs = [_entity_signature(r) for r in rows]

    influence_rows = []
    primary_secondary_rows = []

    incoming: Dict[str, float] = defaultdict(float)
    outgoing: Dict[str, float] = defaultdict(float)

    rejected_counts: Dict[str, int] = defaultdict(int)

    for i, a in enumerate(sigs):
        for j, b in enumerate(sigs):
            if i == j:
                continue

            inf = _directional_influence(
                a,
                b,
                min_direct_topology=args.min_direct_topology,
                min_micro_token_similarity=args.min_micro_token_similarity,
                min_coarse_token_similarity=args.min_coarse_token_similarity,
            )

            if inf.get("rejected_reason"):
                rejected_counts[str(inf.get("rejected_reason"))] += 1

            score_ab = _safe_float(inf.get("score_ab"), 0.0)
            if score_ab < args.min_influence_score:
                continue

            row = {
                "source_entity": a["id"],
                "target_entity": b["id"],
                "influence_score": f"{score_ab:.9f}",

                # Structured support fields.
                "direct_topology_support": f"{_safe_float(inf.get('direct_topology_support'), 0.0):.9f}",

                "micro_token_similarity": f"{_safe_float(inf.get('micro_token_similarity'), 0.0):.9f}",
                "coarse_token_similarity": f"{_safe_float(inf.get('coarse_token_similarity'), 0.0):.9f}",

                # Backward-compatible aggregate.
                "token_similarity": f"{_safe_float(inf.get('token_similarity'), 0.0):.9f}",

                "micro_root_similarity": f"{_safe_float(inf.get('micro_root_similarity'), 0.0):.9f}",
                "coarse_root_similarity": f"{_safe_float(inf.get('coarse_root_similarity'), 0.0):.9f}",

                # Backward-compatible aggregate.
                "root_similarity": f"{_safe_float(inf.get('root_similarity'), 0.0):.9f}",

                "micro_topology_similarity": f"{_safe_float(inf.get('micro_topology_similarity'), 0.0):.9f}",
                "coarse_topology_similarity": f"{_safe_float(inf.get('coarse_topology_similarity'), 0.0):.9f}",

                # Backward-compatible aggregate.
                "topology_similarity": f"{_safe_float(inf.get('topology_similarity'), 0.0):.9f}",

                "overlap_frames": _safe_int(inf.get("overlap_frames"), 0),
                "overlap_ratio": f"{_safe_float(inf.get('overlap_ratio'), 0.0):.9f}",

                "source_birth_frame": a["birth_frame"],
                "target_birth_frame": b["birth_frame"],

                "source_root_hint_micro": a["root_hint_micro_not_identity"],
                "target_root_hint_micro": b["root_hint_micro_not_identity"],
                "source_root_hint_coarse": a["root_hint_coarse_not_identity"],
                "target_root_hint_coarse": b["root_hint_coarse_not_identity"],

                "source_micro_count": a["micro_count"],
                "target_micro_count": b["micro_count"],
                "source_coarse_count": a["coarse_count"],
                "target_coarse_count": b["coarse_count"],

                "source_micro_preview": a["micro_preview"],
                "target_micro_preview": b["micro_preview"],
                "source_coarse_preview": a["coarse_preview"],
                "target_coarse_preview": b["coarse_preview"],

                # Backward-compatible aliases.
                "source_root_hint": a["root_hint_micro_not_identity"],
                "target_root_hint": b["root_hint_micro_not_identity"],

                "source_strength": f"{_safe_float(inf.get('a_strength'), 0.0):.9f}",
                "target_strength": f"{_safe_float(inf.get('b_strength'), 0.0):.9f}",
                "source_strength_advantage": f"{_safe_float(inf.get('a_strength_advantage'), 0.0):.9f}",

                "confidence_basis": str(inf.get("confidence_basis_ab", "")),
            }

            influence_rows.append(row)

            outgoing[a["id"]] += score_ab
            incoming[b["id"]] += score_ab

    influence_rows.sort(
        key=lambda r: -_safe_float(r.get("influence_score"), 0.0)
    )

    for s in sigs:
        eid = s["id"]

        out_score = outgoing.get(eid, 0.0)
        in_score = incoming.get(eid, 0.0)

        dominance = out_score - in_score

        # Role assignment is still graph-based, but now it is explicitly downstream
        # of topology-gated influence edges.
        if dominance >= 0.40:
            role = "PRIMARY_RESONANCE"
        elif dominance <= -0.40:
            role = "SECONDARY_RESPONSE"
        else:
            role = "INTERACTIVE_RESONANCE"

        primary_secondary_rows.append({
            "entity_id": eid,
            "outgoing_influence": f"{out_score:.9f}",
            "incoming_influence": f"{in_score:.9f}",
            "dominance_score": f"{dominance:.9f}",
            "role": role,
            "birth_frame": s["birth_frame"],
            "end_frame": s["end_frame"],
            "duration_frames": s["duration_frames"],
            "root_hint_micro_not_identity": s["root_hint_micro_not_identity"],
            "root_hint_coarse_not_identity": s["root_hint_coarse_not_identity"],

            # Backward-compatible alias.
            "root_hint_not_identity": s["root_hint_micro_not_identity"],
        })

    role_distribution: Dict[str, int] = defaultdict(int)
    for r in primary_secondary_rows:
        role_distribution[r["role"]] += 1

    out_graph = Path(args.out_influence_graph_csv)
    out_roles = Path(args.out_primary_secondary_csv)
    out_meta = Path(args.out_meta_json)
    out_txt = Path(args.out_summary_txt)

    out_graph.parent.mkdir(parents=True, exist_ok=True)

    graph_fields = [
        "source_entity",
        "target_entity",
        "influence_score",

        "direct_topology_support",

        "micro_token_similarity",
        "coarse_token_similarity",
        "token_similarity",

        "micro_root_similarity",
        "coarse_root_similarity",
        "root_similarity",

        "micro_topology_similarity",
        "coarse_topology_similarity",
        "topology_similarity",

        "overlap_frames",
        "overlap_ratio",

        "source_birth_frame",
        "target_birth_frame",

        "source_root_hint_micro",
        "target_root_hint_micro",
        "source_root_hint_coarse",
        "target_root_hint_coarse",

        "source_micro_count",
        "target_micro_count",
        "source_coarse_count",
        "target_coarse_count",

        "source_micro_preview",
        "target_micro_preview",
        "source_coarse_preview",
        "target_coarse_preview",

        # Backward-compatible aliases.
        "source_root_hint",
        "target_root_hint",

        "source_strength",
        "target_strength",
        "source_strength_advantage",

        "confidence_basis",
    ]

    with out_graph.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=graph_fields)
        w.writeheader()
        w.writerows(influence_rows)

    role_fields = [
        "entity_id",
        "outgoing_influence",
        "incoming_influence",
        "dominance_score",
        "role",
        "birth_frame",
        "end_frame",
        "duration_frames",
        "root_hint_micro_not_identity",
        "root_hint_coarse_not_identity",
        "root_hint_not_identity",
    ]

    with out_roles.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=role_fields)
        w.writeheader()
        w.writerows(primary_secondary_rows)

    meta = {
        "stage": "resonance_influence_graph",
        "semantic_version": "structured_micro_coarse_influence_v2",
        "inputs": {
            "ecology_entities_csv": args.ecology_entities_csv,
        },
        "outputs": {
            "influence_graph_csv": args.out_influence_graph_csv,
            "primary_secondary_csv": args.out_primary_secondary_csv,
            "meta_json": args.out_meta_json,
            "summary_txt": args.out_summary_txt,
        },
        "parameters": {
            "min_influence_score": args.min_influence_score,
            "min_direct_topology": args.min_direct_topology,
            "min_micro_token_similarity": args.min_micro_token_similarity,
            "min_coarse_token_similarity": args.min_coarse_token_similarity,
            "similarity_model": {
                "token_similarity": "micro 0.70 + coarse 0.30",
                "root_similarity": "micro 0.70 + coarse 0.30",
                "direct_topology_support": "token 0.52 + root 0.18 + topology 0.30",
                "influence_score": (
                    "direct_topology_support 0.56 + micro_token 0.10 + "
                    "temporal_precedence 0.10 + source_strength_advantage 0.10 + overlap_ratio 0.14"
                ),
            },
        },
        "result": {
            "input_entities": len(rows),
            "influence_edges": len(influence_rows),
            "role_distribution": dict(role_distribution),
            "rejected_counts": dict(rejected_counts),
        },
        "ontology_note": (
            "Influence edges preserve micro/coarse support fields. "
            "Influence cannot arise from overlap or graph centrality alone; "
            "direct topology support is mandatory."
        ),
    }

    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    txt = [
        "RESONANCE INFLUENCE GRAPH",
        "=" * 72,
        f"ecology_entities_csv : {args.ecology_entities_csv}",
        "",
        f"input_entities        : {len(rows)}",
        f"influence_edges       : {len(influence_rows)}",
        "",
        "Role distribution:",
    ]

    for k in sorted(role_distribution):
        txt.append(f"  {k}: {role_distribution[k]}")

    txt.append("")
    txt.append("Rejected candidate edges:")
    for k in sorted(rejected_counts):
        txt.append(f"  {k}: {rejected_counts[k]}")

    txt.extend([
        "",
        "Principle:",
        "  Acoustic overlap is not symmetric.",
        "  Some resonance entities act as primary excitation fields,",
        "  while others behave as secondary responses or coupled resonance structures.",
        "  But influence requires direct resonance topology support;",
        "  graph centrality alone is not causal evidence.",
        "",
    ])

    out_txt.write_text("\n".join(txt), encoding="utf-8")

    print("resonance influence graph complete")
    print(json.dumps(meta["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
