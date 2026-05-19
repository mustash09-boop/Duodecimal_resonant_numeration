# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
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


def _octave(token: str) -> str:
    try:
        return token.split(".", 1)[0]
    except Exception:
        return ""


def _range_zone(token: str) -> str:
    o = _octave(token)
    if o in ("5", "6"):
        return "bass"
    if o in ("7", "8", "9"):
        return "mid"
    return "treble"


def _build_weighted_graph(rows: List[Dict[str, Any]], min_edge_weight: float):
    graph: Dict[str, Dict[str, float]] = defaultdict(dict)

    for r in rows:
        a = str(r.get("node_a", "")).strip()
        b = str(r.get("node_b", "")).strip()
        w = _safe_float(r.get("edge_weight"), 0.0)

        if not a or not b or w < min_edge_weight:
            continue

        graph[a][b] = w
        graph[b][a] = w

    return graph


def _threshold_components(graph: Dict[str, Dict[str, float]], threshold: float) -> List[Set[str]]:
    visited = set()
    comps = []

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

            for nxt, w in graph[n].items():
                if w >= threshold and nxt not in visited:
                    stack.append(nxt)

        comps.append(comp)

    return comps


def _component_score(comp: Set[str], graph: Dict[str, Dict[str, float]]) -> float:
    nodes = sorted(comp)
    score = 0.0

    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            score += graph.get(a, {}).get(b, 0.0)

    return score


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Decompose a global resonance community into weighted subcommunities."
    )

    ap.add_argument("--edges_csv", required=True)
    ap.add_argument("--out_subcommunity_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--min_edge_weight", type=float, default=0.02)
    ap.add_argument("--strong_edge_weight", type=float, default=0.05)
    ap.add_argument("--min_subcommunity_size", type=int, default=3)

    args = ap.parse_args()

    rows = _load_csv(Path(args.edges_csv))
    graph = _build_weighted_graph(rows, args.min_edge_weight)

    comps = _threshold_components(graph, args.strong_edge_weight)

    out_rows = []

    for comp in comps:
        if len(comp) < args.min_subcommunity_size:
            continue

        nodes = sorted(comp)
        zones = sorted(set(_range_zone(n) for n in nodes))
        degrees = sorted(set(_degree(n) for n in nodes))

        score = _component_score(comp, graph)

        out_rows.append({
            "subcommunity_id": len(out_rows) + 1,
            "subcommunity_size": len(nodes),
            "subcommunity_score": f"{score:.9f}",
            "zone_count": len(zones),
            "degree_count": len(degrees),
            "zones": " ".join(zones),
            "degrees": " ".join(degrees),
            "nodes": " ".join(nodes),
        })

    out_rows.sort(
        key=lambda r: (
            -_safe_float(r["subcommunity_score"]),
            -int(r["subcommunity_size"]),
        )
    )

    for i, r in enumerate(out_rows, start=1):
        r["subcommunity_id"] = i

    out_csv = Path(args.out_subcommunity_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "subcommunity_id",
        "subcommunity_size",
        "subcommunity_score",
        "zone_count",
        "degree_count",
        "zones",
        "degrees",
        "nodes",
    ]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    meta = {
        "stage": "micro_resonance_subcommunity",
        "inputs": {"edges_csv": args.edges_csv},
        "outputs": {
            "subcommunity_csv": args.out_subcommunity_csv,
            "meta_json": args.out_meta_json,
            "summary_txt": args.out_summary_txt,
        },
        "parameters": {
            "min_edge_weight": args.min_edge_weight,
            "strong_edge_weight": args.strong_edge_weight,
            "min_subcommunity_size": args.min_subcommunity_size,
        },
        "result": {
            "input_edges": len(rows),
            "graph_nodes": len(graph),
            "subcommunities": len(out_rows),
        },
    }

    Path(args.out_meta_json).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    txt = [
        "MICRO RESONANCE SUBCOMMUNITIES",
        "=" * 72,
        f"edges_csv        : {args.edges_csv}",
        "",
        f"input_edges      : {len(rows)}",
        f"graph_nodes      : {len(graph)}",
        f"subcommunities   : {len(out_rows)}",
        "",
        "Principle:",
        "  Decompose one global resonance body",
        "  into stronger weighted resonance subsystems.",
        "",
    ]

    Path(args.out_summary_txt).write_text("\n".join(txt), encoding="utf-8")

    print("micro resonance subcommunity complete")
    print(json.dumps(meta["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()