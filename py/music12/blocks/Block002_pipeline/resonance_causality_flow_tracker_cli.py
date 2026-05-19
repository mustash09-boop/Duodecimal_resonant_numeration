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


# ============================================================
# Entity / edge loading
# ============================================================

def _entity_id(row: Dict[str, Any]) -> str:
    return str(
        row.get(
            "ecology_entity_id",
            row.get("trajectory_entity_id", row.get("stable_entity_id", row.get("entity_id", ""))),
        )
    ).strip()


def _load_entities(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = _load_csv(path)
    out: Dict[str, Dict[str, Any]] = {}

    for r in rows:
        eid = _entity_id(r)
        if not eid:
            continue

        root_micro = (
            str(r.get("root_hint_micro_not_identity", "")).strip()
            or str(r.get("root_hint_not_identity", "")).strip()
        )
        root_coarse = str(r.get("root_hint_coarse_not_identity", "")).strip()

        out[eid] = {
            "entity_id": eid,
            "birth_frame": _safe_int(r.get("birth_frame"), 0),
            "end_frame": _safe_int(r.get("end_frame"), 0),
            "duration_frames": _safe_int(r.get("duration_frames"), 0),
            "mean_score": _safe_float(r.get("mean_family_score"), 0.0),
            "max_score": _safe_float(r.get("max_family_score"), 0.0),
            "coherence": _safe_float(r.get("mean_topology_coherence"), 0.0),
            "micro_coherence": _safe_float(r.get("mean_micro_topology_coherence"), 0.0),
            "coarse_coherence": _safe_float(r.get("mean_coarse_topology_coherence"), 0.0),
            "root_hint_micro_not_identity": root_micro,
            "root_hint_coarse_not_identity": root_coarse,
            # Backward-compatible alias.
            "root_hint_not_identity": root_micro,
        }

    return out


def _load_edges(path: Path, min_score: float) -> List[Dict[str, Any]]:
    rows = _load_csv(path)
    out = []

    for r in rows:
        score = _safe_float(r.get("influence_score"), 0.0)
        if score < min_score:
            continue

        src = str(r.get("source_entity", "")).strip()
        dst = str(r.get("target_entity", "")).strip()

        if not src or not dst or src == dst:
            continue

        direct_topology_support = _safe_float(
            r.get("direct_topology_support"),
            _safe_float(r.get("topology_similarity"), 0.0),
        )

        micro_token_similarity = _safe_float(r.get("micro_token_similarity"), 0.0)
        coarse_token_similarity = _safe_float(r.get("coarse_token_similarity"), 0.0)
        token_similarity = _safe_float(r.get("token_similarity"), 0.0)

        micro_root_similarity = _safe_float(r.get("micro_root_similarity"), 0.0)
        coarse_root_similarity = _safe_float(r.get("coarse_root_similarity"), 0.0)
        root_similarity = _safe_float(r.get("root_similarity"), 0.0)

        micro_topology_similarity = _safe_float(r.get("micro_topology_similarity"), 0.0)
        coarse_topology_similarity = _safe_float(r.get("coarse_topology_similarity"), 0.0)
        topology_similarity = _safe_float(r.get("topology_similarity"), 0.0)

        out.append({
            "source_entity": src,
            "target_entity": dst,
            "influence_score": score,

            "direct_topology_support": direct_topology_support,

            "micro_token_similarity": micro_token_similarity,
            "coarse_token_similarity": coarse_token_similarity,
            "token_similarity": token_similarity,

            "micro_root_similarity": micro_root_similarity,
            "coarse_root_similarity": coarse_root_similarity,
            "root_similarity": root_similarity,

            "micro_topology_similarity": micro_topology_similarity,
            "coarse_topology_similarity": coarse_topology_similarity,
            "topology_similarity": topology_similarity,

            "overlap_frames": _safe_int(r.get("overlap_frames"), 0),
            "overlap_ratio": _safe_float(r.get("overlap_ratio"), 0.0),

            "source_birth_frame": _safe_int(r.get("source_birth_frame"), 0),
            "target_birth_frame": _safe_int(r.get("target_birth_frame"), 0),

            "source_root_hint_micro": str(r.get("source_root_hint_micro", r.get("source_root_hint", ""))).strip(),
            "target_root_hint_micro": str(r.get("target_root_hint_micro", r.get("target_root_hint", ""))).strip(),
            "source_root_hint_coarse": str(r.get("source_root_hint_coarse", "")).strip(),
            "target_root_hint_coarse": str(r.get("target_root_hint_coarse", "")).strip(),

            "source_micro_count": _safe_int(r.get("source_micro_count"), 0),
            "target_micro_count": _safe_int(r.get("target_micro_count"), 0),

            "source_coarse_count": _safe_int(r.get("source_coarse_count"), 0),
            "target_coarse_count": _safe_int(r.get("target_coarse_count"), 0),

            "source_micro_preview": str(r.get("source_micro_preview", "")).strip(),
            "target_micro_preview": str(r.get("target_micro_preview", "")).strip(),

            "source_coarse_preview": str(r.get("source_coarse_preview", "")).strip(),
            "target_coarse_preview": str(r.get("target_coarse_preview", "")).strip(),

            "source_strength": _safe_float(r.get("source_strength"), 0.0),
            "target_strength": _safe_float(r.get("target_strength"), 0.0),
            "source_strength_advantage": _safe_float(r.get("source_strength_advantage"), 0.0),

            "confidence_basis": str(r.get("confidence_basis", "")).strip(),
        })

    out.sort(
        key=lambda r: (
            r["source_birth_frame"],
            r["target_birth_frame"],
            -r["influence_score"],
        )
    )

    return out


# ============================================================
# Flow logic
# ============================================================

def _edge_continuity_support(edge: Dict[str, Any]) -> float:
    """
    Topology-first continuity support.

    Influence_score is already topology-gated by influence_graph, but we keep
    direct support explicit here so causality does not become pure graph flow.
    """
    return (
        edge["direct_topology_support"] * 0.46
        + edge["micro_token_similarity"] * 0.18
        + edge["coarse_token_similarity"] * 0.08
        + edge["micro_root_similarity"] * 0.10
        + edge["topology_similarity"] * 0.18
    )


def _edge_causal_confidence(edge: Dict[str, Any], src: Dict[str, Any], dst: Dict[str, Any]) -> float:
    birth_gap = dst["birth_frame"] - src["birth_frame"]
    temporal_support = 1.0 if 0 <= birth_gap <= 12 else 0.45 if birth_gap > 12 else 0.0
    overlap_support = min(edge["overlap_frames"] / 32.0, 1.0)
    continuity = _edge_continuity_support(edge)
    coherence = max(
        min(src.get("coherence", 0.0), dst.get("coherence", 0.0)),
        min(src.get("micro_coherence", 0.0), dst.get("micro_coherence", 0.0)),
    )

    return (
        continuity * 0.45
        + edge["influence_score"] * 0.22
        + temporal_support * 0.12
        + overlap_support * 0.11
        + coherence * 0.10
    )


def _chain_strength(ent: Dict[str, Any]) -> float:
    return (
        ent["mean_score"] * 0.42
        + ent["max_score"] * 0.18
        + ent["coherence"] * 0.18
        + ent["micro_coherence"] * 0.14
        + ent["coarse_coherence"] * 0.08
    )


def _classify_flow(
    src: Dict[str, Any],
    dst: Dict[str, Any],
    edge: Dict[str, Any],
    source_total_out: float,
    target_total_in: float,
    *,
    min_direct_topology: float,
    min_micro_continuity: float,
) -> tuple[str, str, float]:
    birth_gap = dst["birth_frame"] - src["birth_frame"]
    score = edge["influence_score"]
    continuity = _edge_continuity_support(edge)
    confidence = _edge_causal_confidence(edge, src, dst)
    chain_gap = max(_chain_strength(src) - _chain_strength(dst), 0.0)
    box_like_continuity = (
        edge["coarse_token_similarity"] * 0.34
        + edge["topology_similarity"] * 0.38
        + edge["overlap_ratio"] * 0.18
        + continuity * 0.10
    )
    tail_like_continuity = (
        edge["topology_similarity"] * 0.34
        + edge["coarse_root_similarity"] * 0.20
        + edge["overlap_ratio"] * 0.26
        + max(0.10 - edge["micro_token_similarity"], 0.0) * 2.0
        + continuity * 0.10
    )

    if edge["direct_topology_support"] < min_direct_topology:
        return "WEAK_FLOW", "low_direct_topology_support", confidence

    if edge["micro_token_similarity"] < min_micro_continuity and edge["coarse_token_similarity"] < 0.10:
        return "WEAK_FLOW", "weak_micro_and_coarse_continuity", confidence

    if (
        birth_gap >= 0
        and birth_gap <= 10
        and score >= 0.28
        and continuity >= 0.14
        and edge["micro_token_similarity"] >= max(min_micro_continuity, 0.05)
        and chain_gap >= 0.02
    ):
        return (
            "EXCITATION_TO_CHAIN",
            "early_birth_gap+micro_continuity+source_chain_strength",
            confidence,
        )

    if (
        birth_gap >= 0
        and edge["overlap_frames"] >= 8
        and score >= 0.22
        and continuity >= 0.09
        and edge["micro_token_similarity"] < 0.10
        and box_like_continuity >= 0.16
        and chain_gap >= 0.00
    ):
        return (
            "NOTE_TO_BOX_TRANSFER",
            "overlap+coarse_topology_continuity+reduced_micro_lock",
            confidence,
        )

    if (
        birth_gap >= 4
        and edge["overlap_frames"] >= 6
        and score >= 0.20
        and continuity >= 0.08
        and edge["micro_token_similarity"] < 0.08
        and tail_like_continuity >= 0.18
        and target_total_in >= source_total_out * 0.60
    ):
        return (
            "BOX_TO_SECONDARY_RESONANCE",
            "late_birth+tail_continuity+incoming_secondary_accumulation",
            confidence,
        )

    if (
        birth_gap >= 0
        and birth_gap <= 16
        and score >= 0.30
        and continuity >= 0.12
    ):
        return (
            "CAUSAL_SEEDING",
            "early_birth_gap+moderate_influence+migration_continuity",
            confidence,
        )

    if (
        birth_gap > 8
        and edge["overlap_frames"] >= 6
        and score >= 0.24
        and continuity >= 0.10
    ):
        return (
            "DELAYED_FEEDING",
            "delayed_birth+overlap+migration_continuity",
            confidence,
        )

    # Masking/absorption must not be centrality-only; require low source continuity
    # or clear incoming dominance.
    if target_total_in > source_total_out * 1.60 and continuity < 0.12:
        return "MASKING_OR_ABSORPTION", "incoming_dominance+weak_source_continuity", confidence

    if (
        edge["overlap_frames"] >= 12
        and score >= 0.22
        and continuity >= 0.09
    ):
        return (
            "SUSTAINED_COUPLING",
            "sustained_overlap+migration_continuity",
            confidence,
        )

    return "WEAK_FLOW", "insufficient_causal_evidence", confidence


def _flow_role_from_stats(
    *,
    out_score: float,
    in_score: float,
    excitation_count: int,
    seeding_count: int,
    box_transfer_count: int,
    tail_count: int,
    delayed_count: int,
    sustained_count: int,
    absorbed_count: int,
    mean_out_continuity: float,
    mean_in_continuity: float,
    mean_causal_confidence: float,
) -> tuple[str, float]:
    """
    Flow roles must emerge from:
    - topology continuity,
    - causal confidence,
    - migration persistence,
    - directional asymmetry,
    not graph degree alone.
    """

    directional_bias = out_score - in_score

    source_energy = (
        out_score * 0.34
        + excitation_count * 0.20
        + seeding_count * 0.24
        + delayed_count * 0.08
        + sustained_count * 0.08
        + mean_out_continuity * 0.16
        + mean_causal_confidence * 0.08
    )

    box_energy = (
        min(out_score, in_score) * 0.22
        + box_transfer_count * 0.30
        + sustained_count * 0.14
        + mean_out_continuity * 0.16
        + mean_in_continuity * 0.10
        + mean_causal_confidence * 0.08
    )

    tail_energy = (
        in_score * 0.30
        + tail_count * 0.28
        + absorbed_count * 0.14
        + mean_in_continuity * 0.14
        + (1.0 - mean_out_continuity) * 0.08
        + (1.0 - mean_causal_confidence) * 0.06
    )

    sink_energy = (
        in_score * 0.34
        + absorbed_count * 0.26
        + (1.0 - mean_out_continuity) * 0.12
        + (1.0 - mean_causal_confidence) * 0.08
    )

    carrier_energy = (
        min(out_score, in_score) * 0.32
        + sustained_count * 0.22
        + mean_in_continuity * 0.18
        + mean_out_continuity * 0.18
        + mean_causal_confidence * 0.10
    )

    if (
        source_energy >= max(carrier_energy * 1.05, 0.08)
        and directional_bias > 0.18
        and mean_out_continuity >= 0.10
    ):
        role = "FLOW_SOURCE"
        rank = source_energy

    elif (
        box_energy >= max(source_energy * 0.90, carrier_energy * 0.96, 0.10)
        and box_transfer_count > 0
        and mean_in_continuity >= 0.08
    ):
        role = "BOX_CARRIER"
        rank = box_energy

    elif (
        tail_energy >= max(carrier_energy * 0.98, sink_energy * 0.92, 0.08)
        and tail_count > 0
        and directional_bias < 0.08
    ):
        role = "SECONDARY_TAIL"
        rank = tail_energy

    elif (
        sink_energy >= max(carrier_energy * 1.04, 0.06)
        and directional_bias < -0.12
    ):
        role = "FLOW_SINK"
        rank = sink_energy

    elif (
        carrier_energy >= 0.04
        or (out_score > 0 and in_score > 0)
    ):
        role = "FLOW_CARRIER"
        rank = carrier_energy

    else:
        role = "FLOW_ISOLATE"
        rank = 0.0

    return role, rank


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Track dynamic causality flow through resonance influence graph "
            "with micro/coarse continuity preserved."
        )
    )

    ap.add_argument("--ecology_entities_csv", required=True)
    ap.add_argument("--influence_graph_csv", required=True)

    ap.add_argument("--out_flow_edges_csv", required=True)
    ap.add_argument("--out_flow_nodes_csv", required=True)
    ap.add_argument("--out_readable_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--min_influence_score", type=float, default=0.24)
    ap.add_argument("--max_flow_depth", type=int, default=4)

    # New anti-collapse gates.
    ap.add_argument("--min_direct_topology", type=float, default=0.08)
    ap.add_argument("--min_micro_continuity", type=float, default=0.03)

    args = ap.parse_args()

    entities = _load_entities(Path(args.ecology_entities_csv))
    edges = _load_edges(Path(args.influence_graph_csv), args.min_influence_score)

    outgoing: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    incoming: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    total_out: Dict[str, float] = defaultdict(float)
    total_in: Dict[str, float] = defaultdict(float)

    continuity_out_sum: Dict[str, float] = defaultdict(float)
    continuity_out_count: Dict[str, int] = defaultdict(int)

    for e in edges:
        outgoing[e["source_entity"]].append(e)
        incoming[e["target_entity"]].append(e)
        total_out[e["source_entity"]] += e["influence_score"]
        total_in[e["target_entity"]] += e["influence_score"]
        continuity_out_sum[e["source_entity"]] += _edge_continuity_support(e)
        continuity_out_count[e["source_entity"]] += 1

    flow_edges = []

    for e in edges:
        src = entities.get(e["source_entity"])
        dst = entities.get(e["target_entity"])

        if not src or not dst:
            continue

        flow_kind, flow_basis, causal_confidence = _classify_flow(
            src,
            dst,
            e,
            total_out[e["source_entity"]],
            total_in[e["target_entity"]],
            min_direct_topology=args.min_direct_topology,
            min_micro_continuity=args.min_micro_continuity,
        )

        birth_gap = dst["birth_frame"] - src["birth_frame"]
        continuity_support = _edge_continuity_support(e)

        flow_edges.append({
            "source_entity": e["source_entity"],
            "target_entity": e["target_entity"],
            "flow_kind": flow_kind,
            "flow_score": f"{e['influence_score']:.9f}",
            "causal_confidence": f"{causal_confidence:.9f}",
            "flow_confidence_basis": flow_basis,
            "influence_confidence_basis": e.get("confidence_basis", ""),

            "birth_gap_frames": birth_gap,
            "overlap_frames": e["overlap_frames"],
            "overlap_ratio": f"{e['overlap_ratio']:.9f}",

            "source_root_hint_micro": e.get("source_root_hint_micro") or src["root_hint_micro_not_identity"],
            "target_root_hint_micro": e.get("target_root_hint_micro") or dst["root_hint_micro_not_identity"],
            "source_root_hint_coarse": e.get("source_root_hint_coarse") or src["root_hint_coarse_not_identity"],
            "target_root_hint_coarse": e.get("target_root_hint_coarse") or dst["root_hint_coarse_not_identity"],
            
            "source_micro_count": e.get("source_micro_count", 0),
            "target_micro_count": e.get("target_micro_count", 0),

            "source_coarse_count": e.get("source_coarse_count", 0),
            "target_coarse_count": e.get("target_coarse_count", 0),

            "source_micro_preview": e.get("source_micro_preview", ""),
            "target_micro_preview": e.get("target_micro_preview", ""),

            "source_coarse_preview": e.get("source_coarse_preview", ""),
            "target_coarse_preview": e.get("target_coarse_preview", ""),

            # Backward-compatible aliases.
            "source_root_hint": e.get("source_root_hint_micro") or src["root_hint_not_identity"],
            "target_root_hint": e.get("target_root_hint_micro") or dst["root_hint_not_identity"],

            "source_birth_frame": src["birth_frame"],
            "target_birth_frame": dst["birth_frame"],

            "direct_topology_support": f"{e['direct_topology_support']:.9f}",
            "continuity_support": f"{continuity_support:.9f}",

            "micro_token_similarity": f"{e['micro_token_similarity']:.9f}",
            "coarse_token_similarity": f"{e['coarse_token_similarity']:.9f}",
            "token_similarity": f"{e['token_similarity']:.9f}",

            "micro_root_similarity": f"{e['micro_root_similarity']:.9f}",
            "coarse_root_similarity": f"{e['coarse_root_similarity']:.9f}",
            "root_similarity": f"{e['root_similarity']:.9f}",

            "micro_topology_similarity": f"{e['micro_topology_similarity']:.9f}",
            "coarse_topology_similarity": f"{e['coarse_topology_similarity']:.9f}",
            "topology_similarity": f"{e['topology_similarity']:.9f}",
        })

    flow_edges.sort(
        key=lambda r: (
            _safe_int(r.get("source_birth_frame"), 0),
            _safe_int(r.get("target_birth_frame"), 0),
            -_safe_float(r.get("causal_confidence"), 0.0),
        )
    )

    node_rows = []

    for eid, ent in entities.items():
        out_score = total_out.get(eid, 0.0)
        in_score = total_in.get(eid, 0.0)

        seeding_count = sum(
            1 for e in flow_edges
            if e["source_entity"] == eid and e["flow_kind"] == "CAUSAL_SEEDING"
        )

        excitation_count = sum(
            1 for e in flow_edges
            if e["source_entity"] == eid and e["flow_kind"] == "EXCITATION_TO_CHAIN"
        )

        box_transfer_count = sum(
            1 for e in flow_edges
            if e["source_entity"] == eid and e["flow_kind"] == "NOTE_TO_BOX_TRANSFER"
        )

        tail_count = sum(
            1 for e in flow_edges
            if e["target_entity"] == eid and e["flow_kind"] == "BOX_TO_SECONDARY_RESONANCE"
        )

        delayed_count = sum(
            1 for e in flow_edges
            if e["source_entity"] == eid and e["flow_kind"] == "DELAYED_FEEDING"
        )

        sustained_count = sum(
            1 for e in flow_edges
            if e["source_entity"] == eid and e["flow_kind"] == "SUSTAINED_COUPLING"
        )

        absorbed_count = sum(
            1 for e in flow_edges
            if e["target_entity"] == eid and e["flow_kind"] == "MASKING_OR_ABSORPTION"
        )

        mean_out_continuity = (
            continuity_out_sum.get(eid, 0.0)
            / max(continuity_out_count.get(eid, 0), 1)
        )

        incoming_continuity_values = [
            _edge_continuity_support(e)
            for e in incoming.get(eid, [])
        ]

        mean_in_continuity = (
            sum(incoming_continuity_values)
            / max(len(incoming_continuity_values), 1)
        )

        outgoing_conf_values = [
            _edge_causal_confidence(
                e,
                entities[e["source_entity"]],
                entities[e["target_entity"]],
            )
            for e in outgoing.get(eid, [])
        ]

        mean_causal_confidence = (
            sum(outgoing_conf_values)
            / max(len(outgoing_conf_values), 1)
        )

        causal_role, causal_rank_score = _flow_role_from_stats(
            out_score=out_score,
            in_score=in_score,
            excitation_count=excitation_count,
            seeding_count=seeding_count,
            box_transfer_count=box_transfer_count,
            tail_count=tail_count,
            delayed_count=delayed_count,
            sustained_count=sustained_count,
            absorbed_count=absorbed_count,
            mean_out_continuity=mean_out_continuity,
            mean_in_continuity=mean_in_continuity,
            mean_causal_confidence=mean_causal_confidence,
        )

        node_rows.append({
            "entity_id": eid,
            "causal_flow_role": causal_role,
            "causal_rank_score": f"{causal_rank_score:.9f}",

            "total_outgoing_flow": f"{out_score:.9f}",
            "total_incoming_flow": f"{in_score:.9f}",

            "mean_outgoing_continuity_support": f"{mean_out_continuity:.9f}",
            "mean_incoming_continuity_support": f"{mean_in_continuity:.9f}",
            "mean_causal_confidence": f"{mean_causal_confidence:.9f}",

            "excitation_to_chain_count": excitation_count,
            "seeding_count": seeding_count,
            "note_to_box_transfer_count": box_transfer_count,
            "box_to_secondary_tail_count": tail_count,
            "delayed_feeding_count": delayed_count,
            "sustained_coupling_count": sustained_count,
            "absorbed_count": absorbed_count,

            "birth_frame": ent["birth_frame"],
            "end_frame": ent["end_frame"],
            "duration_frames": ent["duration_frames"],

            "root_hint_micro_not_identity": ent["root_hint_micro_not_identity"],
            "root_hint_coarse_not_identity": ent["root_hint_coarse_not_identity"],

            # Backward-compatible alias.
            "root_hint_not_identity": ent["root_hint_not_identity"],
        })

    node_rows.sort(
        key=lambda r: (
            _safe_int(r.get("birth_frame"), 0),
            -_safe_float(r.get("causal_rank_score"), 0.0),
        )
    )

    readable_rows = []

    for e in flow_edges[:5000]:
        readable_rows.append({
            "source": (
                f"E{e['source_entity']}"
                f":{e['source_root_hint_micro']}"
                f":M{e.get('source_micro_count', 0)}"
                f":{e.get('source_micro_preview', '')[:120]}"
            ),

            "target": (
                f"E{e['target_entity']}"
                f":{e['target_root_hint_micro']}"
                f":M{e.get('target_micro_count', 0)}"
                f":{e.get('target_micro_preview', '')[:120]}"
            ),
            "flow_kind": e["flow_kind"],
            "flow_score": e["flow_score"],
            "causal_confidence": e["causal_confidence"],
            "continuity_support": e["continuity_support"],
            "birth_gap_frames": e["birth_gap_frames"],
            "overlap_frames": e["overlap_frames"],
        })

    flow_kind_counts: Dict[str, int] = defaultdict(int)
    for e in flow_edges:
        flow_kind_counts[e["flow_kind"]] += 1

    node_role_counts: Dict[str, int] = defaultdict(int)
    for n in node_rows:
        node_role_counts[n["causal_flow_role"]] += 1

    out_flow = Path(args.out_flow_edges_csv)
    out_nodes = Path(args.out_flow_nodes_csv)
    out_readable = Path(args.out_readable_csv)
    out_meta = Path(args.out_meta_json)
    out_txt = Path(args.out_summary_txt)

    out_flow.parent.mkdir(parents=True, exist_ok=True)

    flow_fields = [
        "source_entity",
        "target_entity",
        "flow_kind",
        "flow_score",
        "causal_confidence",
        "flow_confidence_basis",
        "influence_confidence_basis",

        "birth_gap_frames",
        "overlap_frames",
        "overlap_ratio",

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

        "source_root_hint",
        "target_root_hint",

        "source_birth_frame",
        "target_birth_frame",

        "direct_topology_support",
        "continuity_support",

        "micro_token_similarity",
        "coarse_token_similarity",
        "token_similarity",

        "micro_root_similarity",
        "coarse_root_similarity",
        "root_similarity",

        "micro_topology_similarity",
        "coarse_topology_similarity",
        "topology_similarity",
    ]

    with out_flow.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flow_fields)
        w.writeheader()
        w.writerows(flow_edges)

    node_fields = [
        "entity_id",
        "causal_flow_role",
        "causal_rank_score",
        "total_outgoing_flow",
        "total_incoming_flow",
        "mean_outgoing_continuity_support",
        "mean_incoming_continuity_support",
        "mean_causal_confidence",
        "excitation_to_chain_count",
        "seeding_count",
        "note_to_box_transfer_count",
        "box_to_secondary_tail_count",
        "delayed_feeding_count",
        "sustained_coupling_count",
        "absorbed_count",
        "birth_frame",
        "end_frame",
        "duration_frames",
        "root_hint_micro_not_identity",
        "root_hint_coarse_not_identity",
        "root_hint_not_identity",
    ]

    with out_nodes.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=node_fields)
        w.writeheader()
        w.writerows(node_rows)

    with out_readable.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "source",
                "target",
                "flow_kind",
                "flow_score",
                "causal_confidence",
                "continuity_support",
                "birth_gap_frames",
                "overlap_frames",
            ],
        )
        w.writeheader()
        w.writerows(readable_rows)

    meta = {
        "stage": "resonance_causality_flow_tracker",
        "semantic_version": "structured_micro_coarse_causality_v2",
        "inputs": {
            "ecology_entities_csv": args.ecology_entities_csv,
            "influence_graph_csv": args.influence_graph_csv,
        },
        "outputs": {
            "flow_edges_csv": args.out_flow_edges_csv,
            "flow_nodes_csv": args.out_flow_nodes_csv,
            "readable_csv": args.out_readable_csv,
            "meta_json": args.out_meta_json,
            "summary_txt": args.out_summary_txt,
        },
        "parameters": {
            "min_influence_score": args.min_influence_score,
            "max_flow_depth": args.max_flow_depth,
            "min_direct_topology": args.min_direct_topology,
            "min_micro_continuity": args.min_micro_continuity,
        },
        "result": {
            "entities": len(entities),
            "input_influence_edges": len(edges),
            "flow_edges": len(flow_edges),
            "flow_kind_counts": dict(flow_kind_counts),
            "node_role_counts": dict(node_role_counts),
        },
        "ontology_note": (
            "Causality flow preserves micro/coarse support fields from influence graph. "
            "The tracker now separates early excitation-to-chain birth, "
            "note-to-box transfer, and box-to-secondary tail continuation. "
            "Flow roles are not allowed to emerge from graph centrality alone; "
            "direct topology and continuity support are carried forward explicitly."
        ),
    }

    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    txt = [
        "RESONANCE CAUSALITY FLOW TRACKER",
        "=" * 72,
        f"ecology_entities_csv : {args.ecology_entities_csv}",
        f"influence_graph_csv  : {args.influence_graph_csv}",
        "",
        f"entities             : {len(entities)}",
        f"input_edges          : {len(edges)}",
        f"flow_edges           : {len(flow_edges)}",
        "",
        "Flow kind counts:",
    ]

    for k in sorted(flow_kind_counts):
        txt.append(f"  {k}: {flow_kind_counts[k]}")

    txt.append("")
    txt.append("Node role counts:")
    for k in sorted(node_role_counts):
        txt.append(f"  {k}: {node_role_counts[k]}")

    txt.extend([
        "",
        "Principle:",
        "  Influence is not only an edge.",
        "  Causality flows through acoustic scene over time:",
        "  excitation birth, chain seeding, note-to-box transfer, box-to-secondary tail,",
        "  delayed feeding, masking, absorption and sustained coupling.",
        "  But causal flow must preserve direct resonance topology support;",
        "  graph centrality alone is not causal evidence.",
        "",
    ])

    out_txt.write_text("\n".join(txt), encoding="utf-8")

    print("resonance causality flow tracker complete")
    print(json.dumps(meta["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
