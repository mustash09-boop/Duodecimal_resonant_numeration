# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from itertools import combinations
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


def _members(raw: str) -> List[str]:
    return [x.strip() for x in str(raw or "").split() if x.strip()]


def _is_same_harmonic_ladder(a: str, b: str) -> bool:
    return _degree(a) == _degree(b)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build shared resonance ownership graph from micro harmonic families."
    )

    ap.add_argument("--micro_family_csv", required=True)

    ap.add_argument("--out_edges_csv", required=True)
    ap.add_argument("--out_nodes_csv", required=True)
    ap.add_argument("--out_frame_summary_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--min_co_frames", type=int, default=6)
    ap.add_argument("--min_edge_weight", type=float, default=0.02)
    ap.add_argument("--max_nodes_per_frame", type=int, default=12)

    args = ap.parse_args()

    in_csv = Path(args.micro_family_csv)

    out_edges = Path(args.out_edges_csv)
    out_nodes = Path(args.out_nodes_csv)
    out_frame = Path(args.out_frame_summary_csv)
    out_meta = Path(args.out_meta_json)
    out_txt = Path(args.out_summary_txt)

    rows = _load_csv(in_csv)

    by_frame: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for r in rows:
        frame = _safe_int(r.get("frame_index"), 0)
        by_frame[frame].append(r)

    node_counter: Counter[str] = Counter()
    node_energy: Dict[str, float] = defaultdict(float)
    edge_counter: Counter[Tuple[str, str]] = Counter()
    edge_weight: Dict[Tuple[str, str], float] = defaultdict(float)

    frame_rows = []

    for frame in sorted(by_frame):
        fams = sorted(
            by_frame[frame],
            key=lambda r: _safe_float(r.get("family_score"), 0.0),
            reverse=True,
        )[: args.max_nodes_per_frame]

        nodes = []

        for r in fams:
            root = str(r.get("family_root_note", "")).strip()
            score = _safe_float(r.get("family_score"), 0.0)

            if not root:
                continue

            nodes.append((root, score))
            node_counter[root] += 1
            node_energy[root] += score

        for (a, ea), (b, eb) in combinations(nodes, 2):
            if a == b:
                continue

            # Same harmonic ladder is likely note-family duplication, not box shared field.
            if _is_same_harmonic_ladder(a, b):
                continue

            key = tuple(sorted((a, b)))
            edge_counter[key] += 1
            edge_weight[key] += min(ea, eb)

        frame_rows.append({
            "frame_index": frame,
            "node_count": len(nodes),
            "top_nodes": " | ".join(f"{n}:{s:.3f}" for n, s in nodes[:12]),
        })

    edge_rows = []

    total_frames = max(len(by_frame), 1)

    for (a, b), co_frames in edge_counter.items():
        weight = edge_weight[(a, b)] / total_frames

        if co_frames < args.min_co_frames:
            continue

        if weight < args.min_edge_weight:
            continue

        edge_rows.append({
            "node_a": a,
            "node_b": b,
            "co_frames": co_frames,
            "edge_weight": f"{weight:.9f}",
            "degree_a": _degree(a),
            "degree_b": _degree(b),
            "anchor_a": _anchor(a),
            "anchor_b": _anchor(b),
        })

    edge_rows.sort(
        key=lambda r: (
            -_safe_float(r["edge_weight"]),
            -_safe_int(r["co_frames"]),
        )
    )

    node_rows = []

    for node, count in node_counter.items():
        node_rows.append({
            "node": node,
            "frames_present": count,
            "mean_energy": f"{node_energy[node] / max(count, 1):.9f}",
            "degree": _degree(node),
            "anchor": _anchor(node),
        })

    node_rows.sort(
        key=lambda r: (
            -_safe_int(r["frames_present"]),
            -_safe_float(r["mean_energy"]),
        )
    )

    out_edges.parent.mkdir(parents=True, exist_ok=True)

    with out_edges.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "node_a",
                "node_b",
                "co_frames",
                "edge_weight",
                "degree_a",
                "degree_b",
                "anchor_a",
                "anchor_b",
            ],
        )
        w.writeheader()
        w.writerows(edge_rows)

    with out_nodes.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "node",
                "frames_present",
                "mean_energy",
                "degree",
                "anchor",
            ],
        )
        w.writeheader()
        w.writerows(node_rows)

    with out_frame.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "frame_index",
                "node_count",
                "top_nodes",
            ],
        )
        w.writeheader()
        w.writerows(frame_rows)

    meta = {
        "stage": "micro_resonance_graph",
        "inputs": {
            "micro_family_csv": str(in_csv),
        },
        "outputs": {
            "edges_csv": str(out_edges),
            "nodes_csv": str(out_nodes),
            "frame_summary_csv": str(out_frame),
            "meta_json": str(out_meta),
            "summary_txt": str(out_txt),
        },
        "parameters": {
            "min_co_frames": args.min_co_frames,
            "min_edge_weight": args.min_edge_weight,
            "max_nodes_per_frame": args.max_nodes_per_frame,
        },
        "result": {
            "input_rows": len(rows),
            "frames": len(by_frame),
            "nodes": len(node_rows),
            "edges": len(edge_rows),
        },
    }

    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    txt = [
        "MICRO RESONANCE GRAPH",
        "=" * 72,
        f"micro_family_csv : {in_csv}",
        "",
        f"frames           : {len(by_frame)}",
        f"nodes            : {len(node_rows)}",
        f"edges            : {len(edge_rows)}",
        "",
        "Principle:",
        "  Box resonance is searched as shared co-excitation topology,",
        "  not as exact token similarity.",
        "  Same-degree harmonic ladder duplicates are excluded from graph edges.",
        "",
    ]

    out_txt.write_text("\n".join(txt), encoding="utf-8")

    print("micro resonance graph complete")
    print(json.dumps(meta["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()