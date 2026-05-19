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


def _split_tokens(raw: str) -> Set[str]:
    return {x.strip() for x in str(raw or "").split() if x.strip()}


def _mean(xs: List[int]) -> float:
    return sum(xs) / max(len(xs), 1)


def _mean_float(xs: List[float]) -> float:
    return sum(xs) / max(len(xs), 1)


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build dynamic resonance fields from causality flow edges "
            "with micro/coarse topology richness preserved."
        )
    )

    ap.add_argument("--flow_edges_csv", required=True)
    ap.add_argument("--flow_nodes_csv", required=True)

    ap.add_argument("--out_field_windows_csv", required=True)
    ap.add_argument("--out_field_roles_csv", required=True)
    ap.add_argument("--out_readable_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--window_frames", type=int, default=120)
    ap.add_argument("--step_frames", type=int, default=30)

    args = ap.parse_args()

    edges = _load_csv(Path(args.flow_edges_csv))
    nodes = _load_csv(Path(args.flow_nodes_csv))

    node_role = {
        str(r.get("entity_id", "")).strip(): str(r.get("causal_flow_role", "")).strip()
        for r in nodes
    }

    min_frame = 0
    max_frame = 0

    if edges:
        min_frame = min(_safe_int(e.get("source_birth_frame"), 0) for e in edges)
        max_frame = max(_safe_int(e.get("target_birth_frame"), 0) for e in edges)

    field_rows: List[Dict[str, Any]] = []
    readable_rows: List[Dict[str, Any]] = []

    frame = min_frame

    while frame <= max_frame:
        start = frame
        end = frame + args.window_frames

        win_edges = [
            e
            for e in edges
            if (
                start <= _safe_int(e.get("source_birth_frame"), 0) < end
                or start <= _safe_int(e.get("target_birth_frame"), 0) < end
            )
        ]

        involved = set()
        flow_counts = defaultdict(int)
        role_counts = defaultdict(int)

        total_flow = 0.0
        chain_pressure = 0.0
        source_pressure = 0.0
        box_pressure = 0.0
        carrier_pressure = 0.0
        tail_pressure = 0.0
        sink_pressure = 0.0

        micro_tokens: Set[str] = set()
        coarse_tokens: Set[str] = set()

        micro_entity_counts: List[int] = []
        coarse_entity_counts: List[int] = []

        continuity_values: List[float] = []
        causal_confidence_values: List[float] = []
        direct_topology_values: List[float] = []

        for e in win_edges:
            src = str(e.get("source_entity", "")).strip()
            dst = str(e.get("target_entity", "")).strip()
            score = _safe_float(e.get("flow_score"), 0.0)
            kind = str(e.get("flow_kind", "")).strip()

            src_micro = str(e.get("source_micro_preview", "")).strip()
            dst_micro = str(e.get("target_micro_preview", "")).strip()
            src_coarse = str(e.get("source_coarse_preview", "")).strip()
            dst_coarse = str(e.get("target_coarse_preview", "")).strip()

            if src_micro:
                micro_tokens.update(_split_tokens(src_micro))
            if dst_micro:
                micro_tokens.update(_split_tokens(dst_micro))
            if src_coarse:
                coarse_tokens.update(_split_tokens(src_coarse))
            if dst_coarse:
                coarse_tokens.update(_split_tokens(dst_coarse))

            micro_entity_counts.append(_safe_int(e.get("source_micro_count"), 0))
            micro_entity_counts.append(_safe_int(e.get("target_micro_count"), 0))
            coarse_entity_counts.append(_safe_int(e.get("source_coarse_count"), 0))
            coarse_entity_counts.append(_safe_int(e.get("target_coarse_count"), 0))

            continuity_values.append(_safe_float(e.get("continuity_support"), 0.0))
            causal_confidence_values.append(_safe_float(e.get("causal_confidence"), 0.0))
            direct_topology_values.append(_safe_float(e.get("direct_topology_support"), 0.0))

            involved.add(src)
            involved.add(dst)
            flow_counts[kind] += 1
            total_flow += score

            sr = node_role.get(src, "")
            dr = node_role.get(dst, "")

            role_counts[sr] += 1
            role_counts[dr] += 1

            if sr == "FLOW_SOURCE":
                chain_pressure += score
                source_pressure += score
            if sr == "BOX_CARRIER" or dr == "BOX_CARRIER":
                box_pressure += score
            if sr == "FLOW_CARRIER" or dr == "FLOW_CARRIER":
                carrier_pressure += score
            if dr == "SECONDARY_TAIL" or kind == "BOX_TO_SECONDARY_RESONANCE":
                tail_pressure += score
            if dr == "FLOW_SINK":
                sink_pressure += score

        field_density = len(win_edges) / max(args.window_frames, 1)
        entity_density = len(involved) / max(args.window_frames, 1)

        micro_token_richness = len(micro_tokens)
        coarse_token_richness = len(coarse_tokens)

        mean_micro_entity_count = _mean(micro_entity_counts)
        mean_coarse_entity_count = _mean(coarse_entity_counts)
        mean_continuity_support = _mean_float(continuity_values)
        mean_causal_confidence = _mean_float(causal_confidence_values)
        mean_direct_topology_support = _mean_float(direct_topology_values)

        if chain_pressure > max(box_pressure * 1.15, tail_pressure * 1.40, sink_pressure * 1.20):
            field_state = "CHAIN_DOMINANT_FIELD"
        elif box_pressure >= max(chain_pressure * 0.95, tail_pressure * 1.10, sink_pressure * 1.05) and total_flow > 0:
            field_state = "BOX_TRANSFER_FIELD"
        elif tail_pressure >= max(box_pressure * 0.90, chain_pressure * 1.10, 0.05):
            field_state = "SECONDARY_RESONANCE_FIELD"
        elif carrier_pressure >= source_pressure and carrier_pressure >= sink_pressure and total_flow > 0:
            field_state = "CARRIER_DOMINANT_FIELD"
        elif sink_pressure > source_pressure * 1.20:
            field_state = "ABSORPTION_DOMINANT_FIELD"
        elif total_flow > 0:
            field_state = "BALANCED_RESONANCE_FIELD"
        else:
            field_state = "QUIET_FIELD"

        if micro_token_richness >= 80:
            micro_field_texture = "MICRO_RICH_FIELD"
        elif micro_token_richness >= 32:
            micro_field_texture = "MICRO_ACTIVE_FIELD"
        elif micro_token_richness > 0:
            micro_field_texture = "MICRO_SPARSE_FIELD"
        else:
            micro_field_texture = "MICRO_EMPTY_FIELD"

        if mean_continuity_support >= 0.35:
            continuity_texture = "HIGH_CONTINUITY_FIELD"
        elif mean_continuity_support >= 0.18:
            continuity_texture = "MEDIUM_CONTINUITY_FIELD"
        elif mean_continuity_support > 0.0:
            continuity_texture = "LOW_CONTINUITY_FIELD"
        else:
            continuity_texture = "NO_CONTINUITY_FIELD"

        row = {
            "window_start_frame": start,
            "window_end_frame": end,
            "field_state": field_state,
            "micro_field_texture": micro_field_texture,
            "continuity_texture": continuity_texture,
            "edge_count": len(win_edges),
            "entity_count": len(involved),
            "field_density": f"{field_density:.9f}",
            "entity_density": f"{entity_density:.9f}",
            "total_flow": f"{total_flow:.9f}",
            "chain_pressure": f"{chain_pressure:.9f}",
            "source_pressure": f"{source_pressure:.9f}",
            "box_pressure": f"{box_pressure:.9f}",
            "carrier_pressure": f"{carrier_pressure:.9f}",
            "tail_pressure": f"{tail_pressure:.9f}",
            "sink_pressure": f"{sink_pressure:.9f}",
            "excitation_to_chain_count": flow_counts.get("EXCITATION_TO_CHAIN", 0),
            "causal_seeding_count": flow_counts.get("CAUSAL_SEEDING", 0),
            "note_to_box_transfer_count": flow_counts.get("NOTE_TO_BOX_TRANSFER", 0),
            "box_to_secondary_resonance_count": flow_counts.get("BOX_TO_SECONDARY_RESONANCE", 0),
            "delayed_feeding_count": flow_counts.get("DELAYED_FEEDING", 0),
            "masking_absorption_count": flow_counts.get("MASKING_OR_ABSORPTION", 0),
            "sustained_coupling_count": flow_counts.get("SUSTAINED_COUPLING", 0),
            "weak_flow_count": flow_counts.get("WEAK_FLOW", 0),
            "micro_token_richness": micro_token_richness,
            "coarse_token_richness": coarse_token_richness,
            "mean_micro_entity_count": f"{mean_micro_entity_count:.9f}",
            "mean_coarse_entity_count": f"{mean_coarse_entity_count:.9f}",
            "mean_continuity_support": f"{mean_continuity_support:.9f}",
            "mean_causal_confidence": f"{mean_causal_confidence:.9f}",
            "mean_direct_topology_support": f"{mean_direct_topology_support:.9f}",
            "micro_token_preview": " ".join(sorted(micro_tokens)[:80]),
            "coarse_token_preview": " ".join(sorted(coarse_tokens)[:80]),
        }

        field_rows.append(row)

        readable_rows.append({
            "window": f"{start}-{end}",
            "field_state": field_state,
            "micro_field_texture": micro_field_texture,
            "continuity_texture": continuity_texture,
            "summary": (
                f"entities={len(involved)} edges={len(win_edges)} "
                f"chain={chain_pressure:.2f} box={box_pressure:.2f} "
                f"tail={tail_pressure:.2f} sink={sink_pressure:.2f} micro={micro_token_richness} "
                f"mean_cont={mean_continuity_support:.3f}"
            ),
        })

        frame += args.step_frames

    role_rows: List[Dict[str, Any]] = []

    field_state_counts = defaultdict(int)
    micro_texture_counts = defaultdict(int)
    continuity_texture_counts = defaultdict(int)

    for r in field_rows:
        field_state_counts[r["field_state"]] += 1
        micro_texture_counts[r["micro_field_texture"]] += 1
        continuity_texture_counts[r["continuity_texture"]] += 1

    for state, count in sorted(field_state_counts.items()):
        role_rows.append({"kind": "field_state", "role": state, "window_count": count})
    for state, count in sorted(micro_texture_counts.items()):
        role_rows.append({"kind": "micro_field_texture", "role": state, "window_count": count})
    for state, count in sorted(continuity_texture_counts.items()):
        role_rows.append({"kind": "continuity_texture", "role": state, "window_count": count})

    out_windows = Path(args.out_field_windows_csv)
    out_roles = Path(args.out_field_roles_csv)
    out_readable = Path(args.out_readable_csv)
    out_meta = Path(args.out_meta_json)
    out_txt = Path(args.out_summary_txt)

    out_windows.parent.mkdir(parents=True, exist_ok=True)

    window_fields = [
        "window_start_frame",
        "window_end_frame",
        "field_state",
        "micro_field_texture",
        "continuity_texture",
        "edge_count",
        "entity_count",
        "field_density",
        "entity_density",
        "total_flow",
        "chain_pressure",
        "source_pressure",
        "box_pressure",
        "carrier_pressure",
        "tail_pressure",
        "sink_pressure",
        "excitation_to_chain_count",
        "causal_seeding_count",
        "note_to_box_transfer_count",
        "box_to_secondary_resonance_count",
        "delayed_feeding_count",
        "masking_absorption_count",
        "sustained_coupling_count",
        "weak_flow_count",
        "micro_token_richness",
        "coarse_token_richness",
        "mean_micro_entity_count",
        "mean_coarse_entity_count",
        "mean_continuity_support",
        "mean_causal_confidence",
        "mean_direct_topology_support",
        "micro_token_preview",
        "coarse_token_preview",
    ]

    with out_windows.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=window_fields)
        w.writeheader()
        w.writerows(field_rows)

    with out_roles.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "role", "window_count"])
        w.writeheader()
        w.writerows(role_rows)

    with out_readable.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "window",
                "field_state",
                "micro_field_texture",
                "continuity_texture",
                "summary",
            ],
        )
        w.writeheader()
        w.writerows(readable_rows)

    meta = {
        "stage": "resonance_field_dynamics",
        "semantic_version": "micro_topology_field_dynamics_v2",
        "inputs": {
            "flow_edges_csv": args.flow_edges_csv,
            "flow_nodes_csv": args.flow_nodes_csv,
        },
        "outputs": {
            "field_windows_csv": args.out_field_windows_csv,
            "field_roles_csv": args.out_field_roles_csv,
            "readable_csv": args.out_readable_csv,
            "meta_json": args.out_meta_json,
            "summary_txt": args.out_summary_txt,
        },
        "parameters": {
            "window_frames": args.window_frames,
            "step_frames": args.step_frames,
        },
        "result": {
            "input_edges": len(edges),
            "input_nodes": len(nodes),
            "field_windows": len(field_rows),
            "field_state_counts": dict(field_state_counts),
            "micro_texture_counts": dict(micro_texture_counts),
            "continuity_texture_counts": dict(continuity_texture_counts),
        },
        "ontology_note": (
            "Causality is not only edge-to-edge. Dynamic resonance fields must preserve "
            "chain/box/secondary pressures together with source/carrier/sink tendencies, "
            "micro/coarse topology richness, "
            "continuity support and causal confidence."
        ),
    }

    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    txt = [
        "RESONANCE FIELD DYNAMICS",
        "=" * 72,
        f"flow_edges_csv : {args.flow_edges_csv}",
        f"flow_nodes_csv : {args.flow_nodes_csv}",
        "",
        f"input_edges    : {len(edges)}",
        f"input_nodes    : {len(nodes)}",
        f"field_windows  : {len(field_rows)}",
        "",
        "Field state counts:",
    ]

    for k in sorted(field_state_counts):
        txt.append(f"  {k}: {field_state_counts[k]}")

    txt.append("")
    txt.append("Micro texture counts:")
    for k in sorted(micro_texture_counts):
        txt.append(f"  {k}: {micro_texture_counts[k]}")

    txt.append("")
    txt.append("Continuity texture counts:")
    for k in sorted(continuity_texture_counts):
        txt.append(f"  {k}: {continuity_texture_counts[k]}")

    txt.extend([
        "",
        "Principle:",
        "  Causality is not only edge-to-edge.",
        "  Musical resonance forms dynamic fields with chain pressure,",
        "  note-to-box transfer, secondary tail pressure and changing acoustic scene state.",
        "  This version also preserves micro/coarse topology richness,",
        "  continuity support and causal confidence per field window.",
        "",
    ])

    out_txt.write_text("\n".join(txt), encoding="utf-8")

    print("resonance field dynamics complete")
    print(json.dumps(meta["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
