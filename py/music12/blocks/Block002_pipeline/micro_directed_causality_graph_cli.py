# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple


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


def _degree(token: str) -> str:
    try:
        return token.split(".", 1)[1].split("'", 1)[0]
    except Exception:
        return ""


def _anchor(token: str) -> str:
    if "'" not in token:
        return token
    return token.split("'", 1)[0] + "'-"


def _group_by_frame(rows: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        frame = _safe_int(r.get("frame_index"), 0)
        out[frame].append(r)
    return out


def _family_root_token(row: Dict[str, Any]) -> str:
    return (
        str(row.get("family_root_note_micro", "")).strip()
        or str(row.get("family_root_note", "")).strip()
        or str(row.get("family_root_note_coarse", "")).strip()
    )


def _support_quality(
    evidence_count: int,
    root_micro_count: int,
    root_micro_diversity: int,
) -> float:
    quality = 0.0
    quality += min(max(evidence_count, 0), 4) * 0.16
    quality += min(max(root_micro_count, 0), 12) * 0.025
    quality += min(max(root_micro_diversity, 0), 12) * 0.020
    return min(1.0, quality)


def _top_nodes_for_frame(rows: List[Dict[str, Any]], max_nodes: int) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    for r in rows:
        token = _family_root_token(r)
        family_score = _safe_float(r.get("family_score"), 0.0)
        evidence_count = _safe_int(r.get("evidence_count"), 0)
        root_micro_count = _safe_int(r.get("root_micro_count"), 0)
        root_micro_diversity = _safe_int(r.get("root_micro_diversity"), 0)

        if not token:
            continue

        support_quality = _support_quality(
            evidence_count,
            root_micro_count,
            root_micro_diversity,
        )
        selection_score = family_score + support_quality * 0.18

        items.append({
            "token": token,
            "family_score": family_score,
            "selection_score": selection_score,
            "evidence_count": evidence_count,
            "root_micro_count": root_micro_count,
            "root_micro_diversity": root_micro_diversity,
            "support_quality": support_quality,
        })

    items.sort(
        key=lambda x: (
            -_safe_float(x["selection_score"]),
            -_safe_float(x["family_score"]),
            -_safe_int(x["root_micro_count"]),
            -_safe_int(x["root_micro_diversity"]),
        )
    )
    return items[:max_nodes]


def _reciprocity(weight: float, reverse_weight: float) -> float:
    total = weight + reverse_weight
    if total <= 0.0:
        return 0.0
    return 1.0 - abs(weight - reverse_weight) / total


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build directed temporal causality graph from micro harmonic families."
    )

    ap.add_argument("--micro_family_csv", required=True)

    ap.add_argument("--out_directed_edges_csv", required=True)
    ap.add_argument("--out_nodes_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--max_nodes_per_frame", type=int, default=12)
    ap.add_argument("--lag_min_frames", type=int, default=1)
    ap.add_argument("--lag_max_frames", type=int, default=6)
    ap.add_argument("--min_causal_frames", type=int, default=3)
    ap.add_argument("--min_causal_weight", type=float, default=0.008)
    ap.add_argument("--same_degree_weight_scale", type=float, default=0.35)

    args = ap.parse_args()

    rows = _load_csv(Path(args.micro_family_csv))
    by_frame = _group_by_frame(rows)

    frame_indices = sorted(by_frame.keys())
    frame_nodes: Dict[int, List[Dict[str, Any]]] = {}

    for frame in frame_indices:
        frame_nodes[frame] = _top_nodes_for_frame(
            by_frame[frame],
            args.max_nodes_per_frame,
        )

    edge_count: Dict[Tuple[str, str, str], int] = defaultdict(int)
    edge_weight: Dict[Tuple[str, str, str], float] = defaultdict(float)
    edge_support_weight: Dict[Tuple[str, str, str], float] = defaultdict(float)
    edge_first_source_frame: Dict[Tuple[str, str, str], int] = {}
    edge_first_response_frame: Dict[Tuple[str, str, str], int] = {}
    edge_lag_sum: Dict[Tuple[str, str, str], int] = defaultdict(int)

    node_source_count: Dict[str, int] = defaultdict(int)
    node_response_count: Dict[str, int] = defaultdict(int)
    frame_set = set(frame_indices)

    for frame in frame_indices:
        sources = frame_nodes.get(frame, [])

        if not sources:
            continue

        for lag in range(args.lag_min_frames, args.lag_max_frames + 1):
            target_frame = frame + lag

            if target_frame not in frame_set:
                continue

            responses = frame_nodes.get(target_frame, [])

            if not responses:
                continue

            for src_item, dst_item in product(sources, responses):
                src = str(src_item["token"])
                dst = str(dst_item["token"])

                if src == dst:
                    continue

                same_degree = _degree(src) == _degree(dst)
                transition_kind = (
                    "same_degree_sustain"
                    if same_degree
                    else "cross_degree_propagation"
                )

                base_weight = min(
                    _safe_float(src_item["family_score"]),
                    _safe_float(dst_item["family_score"]),
                ) / max(lag, 1)

                support_quality = (
                    _safe_float(src_item["support_quality"])
                    + _safe_float(dst_item["support_quality"])
                ) / 2.0

                weight = base_weight * (1.0 + support_quality * 0.45)
                if same_degree:
                    weight *= args.same_degree_weight_scale

                key = (src, dst, transition_kind)
                edge_count[key] += 1
                edge_weight[key] += weight
                edge_support_weight[key] += support_quality
                edge_lag_sum[key] += lag
                if key not in edge_first_source_frame or frame < edge_first_source_frame[key]:
                    edge_first_source_frame[key] = frame
                if key not in edge_first_response_frame or target_frame < edge_first_response_frame[key]:
                    edge_first_response_frame[key] = target_frame

                node_source_count[src] += 1
                node_response_count[dst] += 1

    total_frames = max(len(frame_indices), 1)
    raw_edge_rows: List[Dict[str, Any]] = []

    for (src, dst, transition_kind), count in edge_count.items():
        norm_weight = edge_weight[(src, dst, transition_kind)] / total_frames
        support_mean = edge_support_weight[(src, dst, transition_kind)] / max(count, 1)
        first_source_frame = edge_first_source_frame.get((src, dst, transition_kind), 0)
        first_response_frame = edge_first_response_frame.get((src, dst, transition_kind), first_source_frame)
        mean_lag = edge_lag_sum[(src, dst, transition_kind)] / max(count, 1)
        raw_edge_rows.append({
            "source_node": src,
            "response_node": dst,
            "transition_kind": transition_kind,
            "causal_frames": count,
            "raw_causal_weight": norm_weight,
            "support_mean": support_mean,
            "first_source_frame": first_source_frame,
            "first_response_frame": first_response_frame,
            "mean_lag_frames": mean_lag,
            "source_degree": _degree(src),
            "response_degree": _degree(dst),
            "source_anchor": _anchor(src),
            "response_anchor": _anchor(dst),
        })

    raw_weight_by_key: Dict[Tuple[str, str, str], float] = {
        (
            str(row["source_node"]),
            str(row["response_node"]),
            str(row["transition_kind"]),
        ): _safe_float(row["raw_causal_weight"])
        for row in raw_edge_rows
    }

    edge_rows: List[Dict[str, Any]] = []
    for row in raw_edge_rows:
        src = str(row["source_node"])
        dst = str(row["response_node"])
        transition_kind = str(row["transition_kind"])
        count = _safe_int(row["causal_frames"])
        raw_weight = _safe_float(row["raw_causal_weight"])

        reverse_weight = raw_weight_by_key.get((dst, src, transition_kind), 0.0)
        reciprocity = _reciprocity(raw_weight, reverse_weight)
        asymmetry = 1.0 - reciprocity

        loop_damping = 1.0
        if transition_kind == "cross_degree_propagation":
            loop_damping = max(0.35, 1.0 - reciprocity * 0.45)
        elif transition_kind == "same_degree_sustain":
            loop_damping = max(0.20, 1.0 - reciprocity * 0.20)

        directional_restore = 1.0 + asymmetry * 0.20
        adjusted_weight = raw_weight * loop_damping * directional_restore

        if count < args.min_causal_frames:
            continue
        if adjusted_weight < args.min_causal_weight:
            continue

        edge_rows.append({
            "source_node": src,
            "response_node": dst,
            "transition_kind": transition_kind,
            "causal_frames": count,
            "causal_weight": f"{adjusted_weight:.9f}",
            "raw_causal_weight": f"{raw_weight:.9f}",
            "reverse_weight": f"{reverse_weight:.9f}",
            "reciprocity": f"{reciprocity:.9f}",
            "loop_damping": f"{loop_damping:.9f}",
            "support_mean": f"{_safe_float(row['support_mean']):.9f}",
            "first_source_frame": _safe_int(row["first_source_frame"]),
            "first_response_frame": _safe_int(row["first_response_frame"]),
            "birth_delta_frames": _safe_int(row["first_response_frame"]) - _safe_int(row["first_source_frame"]),
            "mean_lag_frames": f"{_safe_float(row['mean_lag_frames']):.6f}",
            "source_degree": str(row["source_degree"]),
            "response_degree": str(row["response_degree"]),
            "source_anchor": str(row["source_anchor"]),
            "response_anchor": str(row["response_anchor"]),
        })

    edge_rows.sort(
        key=lambda r: (
            -_safe_float(r["causal_weight"]),
            -_safe_int(r["causal_frames"]),
            r["transition_kind"],
        )
    )

    filtered_node_source_weight: Dict[str, float] = defaultdict(float)
    filtered_node_response_weight: Dict[str, float] = defaultdict(float)
    all_nodes = set()

    for row in edge_rows:
        src = str(row["source_node"])
        dst = str(row["response_node"])
        weight = _safe_float(row["causal_weight"])
        all_nodes.add(src)
        all_nodes.add(dst)
        filtered_node_source_weight[src] += weight
        filtered_node_response_weight[dst] += weight

    node_rows = []
    for node in sorted(all_nodes):
        src_count = node_source_count.get(node, 0)
        resp_count = node_response_count.get(node, 0)
        src_weight = filtered_node_source_weight.get(node, 0.0)
        resp_weight = filtered_node_response_weight.get(node, 0.0)
        role_balance = src_weight - resp_weight

        if role_balance > 0.010:
            role = "exciter_like"
        elif role_balance < -0.010:
            role = "response_like"
        else:
            role = "balanced"

        node_rows.append({
            "node": node,
            "source_count": src_count,
            "response_count": resp_count,
            "source_weight": f"{src_weight:.9f}",
            "response_weight": f"{resp_weight:.9f}",
            "role_balance": f"{role_balance:.9f}",
            "causal_role": role,
            "degree": _degree(node),
            "anchor": _anchor(node),
        })

    node_rows.sort(
        key=lambda r: (
            r["causal_role"],
            -abs(_safe_float(r["role_balance"], 0.0)),
        )
    )

    out_edges = Path(args.out_directed_edges_csv)
    out_nodes = Path(args.out_nodes_csv)
    out_meta = Path(args.out_meta_json)
    out_txt = Path(args.out_summary_txt)

    out_edges.parent.mkdir(parents=True, exist_ok=True)

    with out_edges.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "source_node",
                "response_node",
                "transition_kind",
                "causal_frames",
                "causal_weight",
                "raw_causal_weight",
                "reverse_weight",
                "reciprocity",
                "loop_damping",
                "support_mean",
                "first_source_frame",
                "first_response_frame",
                "birth_delta_frames",
                "mean_lag_frames",
                "source_degree",
                "response_degree",
                "source_anchor",
                "response_anchor",
            ],
        )
        w.writeheader()
        w.writerows(edge_rows)

    with out_nodes.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "node",
                "source_count",
                "response_count",
                "source_weight",
                "response_weight",
                "role_balance",
                "causal_role",
                "degree",
                "anchor",
            ],
        )
        w.writeheader()
        w.writerows(node_rows)

    meta = {
        "stage": "micro_directed_causality_graph",
        "inputs": {
            "micro_family_csv": args.micro_family_csv,
        },
        "outputs": {
            "directed_edges_csv": args.out_directed_edges_csv,
            "nodes_csv": args.out_nodes_csv,
            "meta_json": args.out_meta_json,
            "summary_txt": args.out_summary_txt,
        },
        "parameters": {
            "max_nodes_per_frame": args.max_nodes_per_frame,
            "lag_min_frames": args.lag_min_frames,
            "lag_max_frames": args.lag_max_frames,
            "min_causal_frames": args.min_causal_frames,
            "min_causal_weight": args.min_causal_weight,
            "same_degree_weight_scale": args.same_degree_weight_scale,
        },
        "result": {
            "input_rows": len(rows),
            "frames": len(frame_indices),
            "directed_edges": len(edge_rows),
            "nodes": len(node_rows),
        },
    }

    out_meta.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    cross_degree_edges = sum(
        1 for row in edge_rows
        if row["transition_kind"] == "cross_degree_propagation"
    )
    same_degree_edges = sum(
        1 for row in edge_rows
        if row["transition_kind"] == "same_degree_sustain"
    )
    mean_reciprocity = 0.0
    if edge_rows:
        mean_reciprocity = sum(_safe_float(row["reciprocity"]) for row in edge_rows) / len(edge_rows)

    txt = [
        "MICRO DIRECTED CAUSALITY GRAPH",
        "=" * 72,
        f"micro_family_csv : {args.micro_family_csv}",
        "",
        f"frames           : {len(frame_indices)}",
        f"directed_edges   : {len(edge_rows)}",
        f"cross_degree     : {cross_degree_edges}",
        f"same_degree      : {same_degree_edges}",
        f"mean_reciprocity : {mean_reciprocity:.6f}",
        f"nodes            : {len(node_rows)}",
        "",
        "Principle:",
        "  Resonance is treated as directed temporal causality:",
        "  earlier exciter-like regions -> later response-like regions.",
        "  Same-degree continuity is not discarded anymore; it is kept as",
        "  a weaker sustain/self-carry signal beside cross-degree propagation.",
        "  Highly reciprocal loop-pairs are damped so directed drift can",
        "  survive above closed resonance circulation.",
        "",
    ]

    out_txt.write_text("\n".join(txt), encoding="utf-8")

    print("micro directed causality graph complete")
    print(json.dumps(meta["result"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
