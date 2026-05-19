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


def _top_nodes_for_frame(rows: List[Dict[str, Any]], max_nodes: int) -> List[Tuple[str, float]]:
    items = []

    for r in rows:
        token = str(r.get("family_root_note", "")).strip()
        score = _safe_float(r.get("family_score"), 0.0)

        if not token:
            continue

        items.append((token, score))

    items.sort(key=lambda x: x[1], reverse=True)
    return items[:max_nodes]


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
    ap.add_argument("--min_causal_frames", type=int, default=5)
    ap.add_argument("--min_causal_weight", type=float, default=0.015)

    args = ap.parse_args()

    rows = _load_csv(Path(args.micro_family_csv))
    by_frame = _group_by_frame(rows)

    frame_indices = sorted(by_frame.keys())

    frame_nodes: Dict[int, List[Tuple[str, float]]] = {}

    for frame in frame_indices:
        frame_nodes[frame] = _top_nodes_for_frame(
            by_frame[frame],
            args.max_nodes_per_frame,
        )

    edge_count: Dict[Tuple[str, str], int] = defaultdict(int)
    edge_weight: Dict[Tuple[str, str], float] = defaultdict(float)

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

            for (src, src_score), (dst, dst_score) in product(sources, responses):
                if src == dst:
                    continue

                # Same degree likely belongs to same harmonic ladder / sustain,
                # not causal propagation.
                if _degree(src) == _degree(dst):
                    continue

                weight = min(src_score, dst_score) / max(lag, 1)

                key = (src, dst)

                edge_count[key] += 1
                edge_weight[key] += weight

                node_source_count[src] += 1
                node_response_count[dst] += 1

    total_frames = max(len(frame_indices), 1)

    edge_rows = []

    for (src, dst), count in edge_count.items():
        norm_weight = edge_weight[(src, dst)] / total_frames

        if count < args.min_causal_frames:
            continue

        if norm_weight < args.min_causal_weight:
            continue

        edge_rows.append({
            "source_node": src,
            "response_node": dst,
            "causal_frames": count,
            "causal_weight": f"{norm_weight:.9f}",
            "source_degree": _degree(src),
            "response_degree": _degree(dst),
            "source_anchor": _anchor(src),
            "response_anchor": _anchor(dst),
        })

    edge_rows.sort(
        key=lambda r: (
            -_safe_float(r["causal_weight"]),
            -_safe_int(r["causal_frames"]),
        )
    )

    all_nodes = set()
    for src, dst in edge_count.keys():
        all_nodes.add(src)
        all_nodes.add(dst)

    node_rows = []

    for node in sorted(all_nodes):
        src_count = node_source_count.get(node, 0)
        resp_count = node_response_count.get(node, 0)

        role_balance = src_count - resp_count

        if role_balance > 0:
            role = "exciter_like"
        elif role_balance < 0:
            role = "response_like"
        else:
            role = "balanced"

        node_rows.append({
            "node": node,
            "source_count": src_count,
            "response_count": resp_count,
            "role_balance": role_balance,
            "causal_role": role,
            "degree": _degree(node),
            "anchor": _anchor(node),
        })

    node_rows.sort(
        key=lambda r: (
            r["causal_role"],
            -abs(_safe_int(r["role_balance"], 0)),
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
                "causal_frames",
                "causal_weight",
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

    txt = [
        "MICRO DIRECTED CAUSALITY GRAPH",
        "=" * 72,
        f"micro_family_csv : {args.micro_family_csv}",
        "",
        f"frames           : {len(frame_indices)}",
        f"directed_edges   : {len(edge_rows)}",
        f"nodes            : {len(node_rows)}",
        "",
        "Principle:",
        "  Resonance is treated as directed temporal causality:",
        "  earlier exciter-like regions -> later response-like regions.",
        "  Same-degree sustain ladders are excluded from causal propagation edges.",
        "",
    ]

    out_txt.write_text("\n".join(txt), encoding="utf-8")

    print("micro directed causality graph complete")
    print(json.dumps(meta["result"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()