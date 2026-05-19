# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


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


def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(v, 0.0) for v in weights.values())

    if total <= 0.0:
        return {k: 0.0 for k in weights}

    return {
        k: max(v, 0.0) / total
        for k, v in weights.items()
    }


# ============================================================
# Ownership support logic
# ============================================================

def _continuity_support(edge: Dict[str, Any]) -> float:
    """
    Topology-first continuity support.

    Ownership must emerge from preserved resonance continuity,
    not from graph centrality or role popularity.
    """
    micro_count_bonus = min(
        (
            _safe_int(edge.get("source_micro_count"), 0)
            + _safe_int(edge.get("target_micro_count"), 0)
        ) / 120.0,
        0.10,
    )

    return (
        _safe_float(edge.get("direct_topology_support"), 0.0) * 0.44
        + _safe_float(edge.get("continuity_support"), 0.0) * 0.22
        + _safe_float(edge.get("micro_token_similarity"), 0.0) * 0.14
        + _safe_float(edge.get("micro_root_similarity"), 0.0) * 0.10
        + _safe_float(edge.get("topology_similarity"), 0.0) * 0.10
        + micro_count_bonus
    )


def _causal_support(edge: Dict[str, Any]) -> float:
    return (
        _safe_float(edge.get("causal_confidence"), 0.0) * 0.56
        + _safe_float(edge.get("flow_score"), 0.0) * 0.26
        + _safe_float(edge.get("overlap_ratio"), 0.0) * 0.18
    )


def _topology_basis(edge: Dict[str, Any]) -> str:
    parts = []

    if _safe_float(edge.get("micro_token_similarity"), 0.0) > 0.0:
        parts.append("micro_token")

    if _safe_float(edge.get("coarse_token_similarity"), 0.0) > 0.0:
        parts.append("coarse_token")

    if _safe_float(edge.get("micro_root_similarity"), 0.0) > 0.0:
        parts.append("micro_root")

    if _safe_float(edge.get("topology_similarity"), 0.0) > 0.0:
        parts.append("topology_signature")

    if _safe_float(edge.get("continuity_support"), 0.0) > 0.0:
        parts.append("continuity_support")

    if _safe_float(edge.get("causal_confidence"), 0.0) > 0.0:
        parts.append("causal_confidence")

    return "+".join(parts) if parts else "weak_or_missing_basis"


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Resolve causal resonance ownership probabilities "
            "while preserving micro/coarse continuity support."
        )
    )

    ap.add_argument("--flow_edges_csv", required=True)
    ap.add_argument("--flow_nodes_csv", required=True)
    ap.add_argument("--field_windows_csv", required=True)

    ap.add_argument("--out_ownership_csv", required=True)
    ap.add_argument("--out_entity_roles_csv", required=True)
    ap.add_argument("--out_readable_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    # Anti-collapse thresholds.
    ap.add_argument("--min_direct_topology", type=float, default=0.08)
    ap.add_argument("--min_continuity_support", type=float, default=0.10)
    ap.add_argument("--min_causal_confidence", type=float, default=0.12)

    args = ap.parse_args()

    edges = _load_csv(Path(args.flow_edges_csv))
    nodes = _load_csv(Path(args.flow_nodes_csv))
    fields = _load_csv(Path(args.field_windows_csv))

    node_map = {
        str(r.get("entity_id", "")).strip(): r
        for r in nodes
    }

    field_states = []
    for f in fields:
        field_states.append({
            "start": _safe_int(f.get("window_start_frame"), 0),
            "end": _safe_int(f.get("window_end_frame"), 0),
            "state": str(f.get("field_state", "")).strip(),
            "source_pressure": _safe_float(f.get("source_pressure"), 0.0),
            "carrier_pressure": _safe_float(f.get("carrier_pressure"), 0.0),
            "sink_pressure": _safe_float(f.get("sink_pressure"), 0.0),
        })

    ownership_rows = []
    readable_rows = []

    entity_summary: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {
            "owned": 0.0,
            "fed": 0.0,
            "masked": 0.0,
            "carried": 0.0,
            "events": 0.0,
        }
    )

    rejected_edges: Dict[str, int] = defaultdict(int)

    for e in edges:
        src = str(e.get("source_entity", "")).strip()
        dst = str(e.get("target_entity", "")).strip()

        if not src or not dst:
            rejected_edges["missing_entity"] += 1
            continue

        flow_kind = str(e.get("flow_kind", "")).strip()

        flow_score = _safe_float(e.get("flow_score"), 0.0)
        causal_confidence = _safe_float(e.get("causal_confidence"), 0.0)

        direct_topology_support = _safe_float(
            e.get("direct_topology_support"), 0.0
        )

        continuity_support = _continuity_support(e)
        causal_support = _causal_support(e)

        if direct_topology_support < args.min_direct_topology:
            rejected_edges["low_direct_topology"] += 1
            continue

        if continuity_support < args.min_continuity_support:
            rejected_edges["low_continuity_support"] += 1
            continue

        if causal_confidence < args.min_causal_confidence:
            rejected_edges["low_causal_confidence"] += 1
            continue

        src_node = node_map.get(src, {})
        dst_node = node_map.get(dst, {})

        src_role = str(src_node.get("causal_flow_role", "")).strip()
        dst_role = str(dst_node.get("causal_flow_role", "")).strip()

        src_rank = _safe_float(src_node.get("causal_rank_score"), 0.0)
        dst_rank = _safe_float(dst_node.get("causal_rank_score"), 0.0)

        source_pressure = 0.0
        carrier_pressure = 0.0
        sink_pressure = 0.0

        edge_birth = _safe_int(e.get("source_birth_frame"), 0)

        for fw in field_states:
            if fw["start"] <= edge_birth < fw["end"]:
                source_pressure = fw["source_pressure"]
                carrier_pressure = fw["carrier_pressure"]
                sink_pressure = fw["sink_pressure"]
                break

        ownership = {
            "ownership": 0.0,
            "feeding": 0.0,
            "masking": 0.0,
            "carrying": 0.0,
        }

        # ====================================================
        # OWNERSHIP
        # ====================================================

        ownership["ownership"] += continuity_support * 0.42
        ownership["ownership"] += causal_support * 0.16
        ownership["ownership"] += max(src_rank, 0.0) * 0.08

        if src_role == "FLOW_SOURCE":
            ownership["ownership"] += 0.10

        if flow_kind == "CAUSAL_SEEDING":
            ownership["ownership"] += 0.18

        # ====================================================
        # FEEDING
        # ====================================================

        ownership["feeding"] += causal_support * 0.26

        if flow_kind == "DELAYED_FEEDING":
            ownership["feeding"] += 0.22

        ownership["feeding"] += flow_score * 0.10

        # ====================================================
        # CARRYING
        # ====================================================

        if src_role == "FLOW_CARRIER":
            ownership["carrying"] += 0.24

        ownership["carrying"] += carrier_pressure * 0.00018
        if src_role == "FLOW_CARRIER" or dst_role == "FLOW_CARRIER":
            ownership["carrying"] += continuity_support * 0.12

        # ====================================================
        # MASKING
        # ====================================================

        # Masking must not become a universal sink.
        if flow_kind == "MASKING_OR_ABSORPTION":
            ownership["masking"] += 0.22

        if continuity_support < 0.14:
            ownership["masking"] += sink_pressure * 0.0004

        # ====================================================
        # NORMALIZATION
        # ====================================================

        probs = _normalize(ownership)

        entity_summary[src]["owned"] += probs["ownership"]
        entity_summary[src]["fed"] += probs["feeding"]
        entity_summary[src]["carried"] += probs["carrying"]
        entity_summary[src]["events"] += 1.0

        entity_summary[dst]["masked"] += probs["masking"]
        entity_summary[dst]["events"] += 1.0

        ownership_rows.append({
            "source_entity": src,
            "target_entity": dst,

            "flow_kind": flow_kind,

            "ownership_probability": f"{probs['ownership']:.9f}",
            "feeding_probability": f"{probs['feeding']:.9f}",
            "carrying_probability": f"{probs['carrying']:.9f}",
            "masking_probability": f"{probs['masking']:.9f}",

            "source_role": src_role,
            "target_role": dst_role,

            "flow_score": f"{flow_score:.9f}",
            "causal_confidence": f"{causal_confidence:.9f}",

            "direct_topology_support": f"{direct_topology_support:.9f}",
            "continuity_support": f"{continuity_support:.9f}",
            "causal_support": f"{causal_support:.9f}",

            "micro_token_similarity": f"{_safe_float(e.get('micro_token_similarity'), 0.0):.9f}",
            "coarse_token_similarity": f"{_safe_float(e.get('coarse_token_similarity'), 0.0):.9f}",

            "micro_root_similarity": f"{_safe_float(e.get('micro_root_similarity'), 0.0):.9f}",
            "coarse_root_similarity": f"{_safe_float(e.get('coarse_root_similarity'), 0.0):.9f}",

            "topology_similarity": f"{_safe_float(e.get('topology_similarity'), 0.0):.9f}",

            "ownership_confidence_basis": _topology_basis(e),

            "source_root_hint_micro": str(
                e.get("source_root_hint_micro", e.get("source_root_hint", ""))
            ).strip(),

            "target_root_hint_micro": str(
                e.get("target_root_hint_micro", e.get("target_root_hint", ""))
            ).strip(),

            "source_root_hint_coarse": str(
                e.get("source_root_hint_coarse", "")
            ).strip(),

            "target_root_hint_coarse": str(
                e.get("target_root_hint_coarse", "")
            ).strip(),
            
            "source_micro_count": _safe_int(e.get("source_micro_count"), 0),
            "target_micro_count": _safe_int(e.get("target_micro_count"), 0),

            "source_micro_preview": str(
                e.get("source_micro_preview", "")
            ).strip(),

            "target_micro_preview": str(
                e.get("target_micro_preview", "")
            ).strip(),

            "source_coarse_preview": str(
                e.get("source_coarse_preview", "")
            ).strip(),

            "target_coarse_preview": str(
                e.get("target_coarse_preview", "")
            ).strip(),
        })

        readable_rows.append({
            "interaction": f"E{src} → E{dst}",
            "flow_kind": flow_kind,
            "ownership": f"{probs['ownership']:.3f}",
            "feeding": f"{probs['feeding']:.3f}",
            "carrying": f"{probs['carrying']:.3f}",
            "masking": f"{probs['masking']:.3f}",
            "continuity_support": f"{continuity_support:.3f}",
            "causal_confidence": f"{causal_confidence:.3f}",
        })

    role_rows = []

    role_distribution = defaultdict(int)

    for eid, stats in entity_summary.items():
        ev = max(stats["events"], 1.0)

        own = stats["owned"] / ev
        fed = stats["fed"] / ev
        carried = stats["carried"] / ev
        masked = stats["masked"] / ev

        dominance_gap = own - max(fed, carried, masked)

        if (
            own >= 0.46
            and dominance_gap >= 0.12
        ):
            role = "PRIMARY_OWNER"

        elif (
            own >= 0.34
            and fed >= 0.18
            and abs(own - fed) <= 0.10
        ):
            role = "SHARED_RESONANCE_CLUSTER"

        elif (
            carried >= 0.26
            and carried >= own * 0.82
            and carried >= masked
        ):
            role = "TRANSITIONAL_CARRIER"

        elif (
            masked >= 0.24
            and masked >= own * 0.72
        ):
            role = "LOCAL_MASKING_REGION"

        elif (
            fed >= 0.22
            and fed >= carried * 0.72
        ):
            role = "FEEDING_STRUCTURE"

        elif (
            own >= 0.28
            and dominance_gap >= 0.04
        ):
            role = "MICRO_DOMINANT_OWNER"

        else:
            role = "MIXED_OWNERSHIP_FIELD"

        role_distribution[role] += 1

        role_rows.append({
            "entity_id": eid,

            "ownership_strength": f"{own:.9f}",
            "feeding_strength": f"{fed:.9f}",
            "carrying_strength": f"{carried:.9f}",
            "masking_strength": f"{masked:.9f}",

            "ownership_role": role,
        })

    ownership_rows.sort(
        key=lambda r: (
            -_safe_float(r.get("ownership_probability"), 0.0),
            -_safe_float(r.get("continuity_support"), 0.0),
        )
    )

    out_ownership = Path(args.out_ownership_csv)
    out_roles = Path(args.out_entity_roles_csv)
    out_readable = Path(args.out_readable_csv)
    out_meta = Path(args.out_meta_json)
    out_txt = Path(args.out_summary_txt)

    out_ownership.parent.mkdir(parents=True, exist_ok=True)

    ownership_fields = [
        "source_entity",
        "target_entity",

        "flow_kind",

        "ownership_probability",
        "feeding_probability",
        "carrying_probability",
        "masking_probability",

        "source_role",
        "target_role",

        "flow_score",
        "causal_confidence",

        "direct_topology_support",
        "continuity_support",
        "causal_support",

        "micro_token_similarity",
        "coarse_token_similarity",

        "micro_root_similarity",
        "coarse_root_similarity",

        "topology_similarity",

        "ownership_confidence_basis",

        "source_root_hint_micro",
        "target_root_hint_micro",
        "source_root_hint_coarse",
        "target_root_hint_coarse",
        "source_micro_count",
        "target_micro_count",
        "source_micro_preview",
        "target_micro_preview",
        "source_coarse_preview",
        "target_coarse_preview",
    ]

    with out_ownership.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ownership_fields)
        w.writeheader()
        w.writerows(ownership_rows)

    role_fields = [
        "entity_id",
        "ownership_strength",
        "feeding_strength",
        "carrying_strength",
        "masking_strength",
        "ownership_role",
    ]

    with out_roles.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=role_fields)
        w.writeheader()
        w.writerows(role_rows)

    with out_readable.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "interaction",
                "flow_kind",
                "ownership",
                "feeding",
                "carrying",
                "masking",
                "continuity_support",
                "causal_confidence",
            ],
        )
        w.writeheader()
        w.writerows(readable_rows)

    meta = {
        "stage": "resonance_ownership_resolution",
        "semantic_version": "structured_micro_coarse_ownership_v2",
        "inputs": {
            "flow_edges_csv": args.flow_edges_csv,
            "flow_nodes_csv": args.flow_nodes_csv,
            "field_windows_csv": args.field_windows_csv,
        },
        "outputs": {
            "ownership_csv": args.out_ownership_csv,
            "entity_roles_csv": args.out_entity_roles_csv,
            "readable_csv": args.out_readable_csv,
            "meta_json": args.out_meta_json,
            "summary_txt": args.out_summary_txt,
        },
        "parameters": {
            "min_direct_topology": args.min_direct_topology,
            "min_continuity_support": args.min_continuity_support,
            "min_causal_confidence": args.min_causal_confidence,
        },
        "result": {
            "ownership_rows": len(ownership_rows),
            "entity_roles": len(role_rows),
            "role_distribution": dict(role_distribution),
            "rejected_edges": dict(rejected_edges),
        },
        "ontology_note": (
            "Ownership is probabilistic resonance continuity. "
            "Ownership cannot emerge from role popularity or graph centrality alone; "
            "direct topology and causal continuity support are mandatory."
        ),
    }

    out_meta.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    txt = [
        "RESONANCE OWNERSHIP RESOLUTION",
        "=" * 72,
        f"flow_edges_csv    : {args.flow_edges_csv}",
        f"flow_nodes_csv    : {args.flow_nodes_csv}",
        f"field_windows_csv : {args.field_windows_csv}",
        "",
        f"ownership_rows    : {len(ownership_rows)}",
        f"entity_roles      : {len(role_rows)}",
        "",
        "Role distribution:",
    ]

    for k in sorted(role_distribution):
        txt.append(f"  {k}: {role_distribution[k]}")

    txt.append("")
    txt.append("Rejected edges:")
    for k in sorted(rejected_edges):
        txt.append(f"  {k}: {rejected_edges[k]}")

    txt.extend([
        "",
        "Principle:",
        "  Harmonic topology ownership is probabilistic.",
        "  Ownership emerges from preserved resonance continuity,",
        "  causal support and topology stability.",
        "  An entity may own, feed, carry or mask resonance structures simultaneously.",
        "",
    ])

    out_txt.write_text("\n".join(txt), encoding="utf-8")

    print("resonance ownership resolution complete")
    print(json.dumps(meta["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
