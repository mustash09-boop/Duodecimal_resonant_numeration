# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set


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


def _build_graph(rows: List[Dict[str, Any]], min_weight: float):
    graph: Dict[str, Set[str]] = defaultdict(set)
    edge_weights: Dict[tuple[str, str], float] = {}

    for r in rows:
        a = str(r.get("node_a", "")).strip()
        b = str(r.get("node_b", "")).strip()

        if not a or not b:
            continue

        w = _safe_float(r.get("edge_weight"), 0.0)

        if w < min_weight:
            continue

        graph[a].add(b)
        graph[b].add(a)

        edge_weights[tuple(sorted((a, b)))] = w

    return graph, edge_weights


def _connected_components(graph: Dict[str, Set[str]]) -> List[Set[str]]:
    visited = set()
    communities = []

    for node in graph:
        if node in visited:
            continue

        stack = [node]
        comp = set()

        while stack:
            n = stack.pop()

            if n in visited:
                continue

            visited.add(n)
            comp.add(n)

            for nxt in graph[n]:
                if nxt not in visited:
                    stack.append(nxt)

        communities.append(comp)

    return communities


def _community_score(
    community: Set[str],
    edge_weights: Dict[tuple[str, str], float],
) -> float:
    score = 0.0

    nodes = sorted(community)

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            key = tuple(sorted((nodes[i], nodes[j])))
            score += edge_weights.get(key, 0.0)

    return score


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build resonance communities from shared resonance graph."
    )

    ap.add_argument("--edges_csv", required=True)

    ap.add_argument("--out_community_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--min_edge_weight", type=float, default=0.02)
    ap.add_argument("--min_community_size", type=int, default=3)

    args = ap.parse_args()

    in_csv = Path(args.edges_csv)

    out_csv = Path(args.out_community_csv)
    out_meta = Path(args.out_meta_json)
    out_txt = Path(args.out_summary_txt)

    rows = _load_csv(in_csv)

    graph, edge_weights = _build_graph(
        rows,
        args.min_edge_weight,
    )

    communities = _connected_components(graph)

    out_rows = []

    kept = 0

    for idx, comm in enumerate(communities, start=1):
        if len(comm) < args.min_community_size:
            continue

        kept += 1

        nodes = sorted(comm)

        score = _community_score(
            comm,
            edge_weights,
        )

        degrees = sorted(set(_degree(n) for n in nodes))
        anchors = sorted(set(_anchor(n) for n in nodes))

        out_rows.append({
            "community_id": kept,
            "community_size": len(nodes),
            "community_score": f"{score:.9f}",
            "degree_count": len(degrees),
            "anchor_count": len(anchors),
            "degrees": " ".join(degrees),
            "anchors": " ".join(anchors),
            "nodes": " ".join(nodes),
        })

    out_rows.sort(
        key=lambda r: (
            -_safe_float(r["community_score"]),
            -_safe_int(r["community_size"]),
        )
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "community_id",
                "community_size",
                "community_score",
                "degree_count",
                "anchor_count",
                "degrees",
                "anchors",
                "nodes",
            ],
        )
        w.writeheader()
        w.writerows(out_rows)

    meta = {
        "stage": "micro_resonance_community",
        "inputs": {
            "edges_csv": str(in_csv),
        },
        "outputs": {
            "community_csv": str(out_csv),
            "meta_json": str(out_meta),
            "summary_txt": str(out_txt),
        },
        "parameters": {
            "min_edge_weight": args.min_edge_weight,
            "min_community_size": args.min_community_size,
        },
        "result": {
            "input_edges": len(rows),
            "graph_nodes": len(graph),
            "communities_total": len(communities),
            "communities_kept": len(out_rows),
        },
    }

    out_meta.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    txt = [
        "MICRO RESONANCE COMMUNITIES",
        "=" * 72,
        f"edges_csv           : {in_csv}",
        "",
        f"input_edges         : {len(rows)}",
        f"graph_nodes         : {len(graph)}",
        f"communities_total   : {len(communities)}",
        f"communities_kept    : {len(out_rows)}",
        "",
        "Principle:",
        "  Instrument identity is searched as stable",
        "  resonance interaction communities.",
        "",
    ]

    out_txt.write_text(
        "\n".join(txt),
        encoding="utf-8",
    )

    print("micro resonance community complete")
    print(json.dumps(meta["result"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()