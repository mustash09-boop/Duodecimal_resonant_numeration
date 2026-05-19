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
        return float(str(x).replace(",", "."))
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(str(x).replace(",", ".")))
    except Exception:
        return default


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _tokens(raw: Any) -> Set[str]:
    return {x.strip() for x in str(raw or "").replace("|", " ").replace(",", " ").split() if x.strip()}


def _build_graph(links: List[Dict[str, Any]], min_score: float) -> Dict[str, Set[str]]:
    graph: Dict[str, Set[str]] = defaultdict(set)

    for r in links:
        score = _safe_float(r.get("interaction_score"), 0.0)
        if score < min_score:
            continue

        a = str(r.get("source_identity_id", "")).strip()
        b = str(r.get("target_identity_id", "")).strip()

        if not a or not b or a == b:
            continue

        graph[a].add(b)
        graph[b].add(a)

    return graph


def _components(graph: Dict[str, Set[str]], all_nodes: Set[str]) -> List[Set[str]]:
    visited = set()
    comps = []

    for node in sorted(all_nodes):
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

            for nxt in graph.get(n, set()):
                if nxt not in visited:
                    stack.append(nxt)

        comps.append(comp)

    return comps


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Cluster attractor ecology nodes into candidate instrument-body resonance communities."
    )

    ap.add_argument("--polyphonic_ecology_nodes_csv", required=True)
    ap.add_argument("--polyphonic_ecology_links_csv", required=True)
    ap.add_argument("--attractor_events_csv", required=True)

    ap.add_argument("--out_instrument_ecology_clusters_csv", required=True)
    ap.add_argument("--out_identity_to_cluster_csv", required=True)
    ap.add_argument("--out_cluster_frame_csv", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--min_cluster_link_score", type=float, default=0.22)
    ap.add_argument("--fps", type=float, default=60.0)

    args = ap.parse_args()

    nodes = _load_csv(Path(args.polyphonic_ecology_nodes_csv))
    links = _load_csv(Path(args.polyphonic_ecology_links_csv))
    attractors = _load_csv(Path(args.attractor_events_csv))

    node_map = {str(r.get("identity_id", "")).strip(): r for r in nodes}
    attractor_map = {str(r.get("identity_id", "")).strip(): r for r in attractors}

    all_nodes = set(node_map.keys())
    graph = _build_graph(links, args.min_cluster_link_score)
    comps = _components(graph, all_nodes)

    cluster_rows = []
    mapping_rows = []
    frame_rows = []

    for cid, comp in enumerate(comps, start=1):
        notes = []
        behaviors = []
        statuses = []
        ecology_energy = 0.0

        births = []
        ends = []

        body_tokens = set()
        secondary_tokens = set()
        core_tokens = set()

        for iid in comp:
            n = node_map.get(iid, {})
            a = attractor_map.get(iid, {})

            notes.append(str(n.get("attractor_note", "")).strip())
            behaviors.append(str(n.get("field_behavior", "")).strip())
            statuses.append(str(n.get("ecology_status", "")).strip())

            ecology_energy += _safe_float(n.get("ecology_energy"), 0.0)

            birth = _safe_int(n.get("birth_frame"), 0)
            end = _safe_int(n.get("end_frame"), birth)
            births.append(birth)
            ends.append(end)

            body_tokens |= _tokens(a.get("instrument_body_tokens", ""))
            secondary_tokens |= _tokens(a.get("secondary_field_tokens", ""))
            core_tokens |= _tokens(a.get("excitation_core_tokens", ""))

            mapping_rows.append({
                "cluster_id": cid,
                "identity_id": iid,
                "attractor_note": n.get("attractor_note", ""),
                "field_behavior": n.get("field_behavior", ""),
                "ecology_energy": n.get("ecology_energy", ""),
                "ecology_status": n.get("ecology_status", ""),
            })

        behavior_counts = defaultdict(int)
        for b in behaviors:
            if b:
                behavior_counts[b] += 1

        status_counts = defaultdict(int)
        for s in statuses:
            if s:
                status_counts[s] += 1

        note_counts = defaultdict(int)
        for note in notes:
            if note:
                note_counts[note] += 1

        birth = min(births) if births else 0
        end = max(ends) if ends else birth

        cluster_density = ecology_energy / max(len(comp), 1)

        if len(comp) >= 8 and cluster_density >= 3.5:
            cluster_kind = "STRONG_INSTRUMENT_BODY_CLUSTER"
        elif len(comp) >= 4:
            cluster_kind = "POSSIBLE_INSTRUMENT_BODY_CLUSTER"
        else:
            cluster_kind = "LOCAL_RESONANCE_FRAGMENT"

        cluster_rows.append({
            "cluster_id": cid,
            "cluster_kind": cluster_kind,
            "identity_count": len(comp),
            "birth_frame": birth,
            "end_frame": end,
            "duration_frames": end - birth + 1,
            "ecology_energy_sum": f"{ecology_energy:.9f}",
            "ecology_energy_mean": f"{cluster_density:.9f}",
            "dominant_notes": " | ".join(f"{k}:{v}" for k, v in sorted(note_counts.items(), key=lambda x: (-x[1], x[0]))[:16]),
            "field_behavior_distribution": " | ".join(f"{k}:{v}" for k, v in sorted(behavior_counts.items(), key=lambda x: (-x[1], x[0]))),
            "ecology_status_distribution": " | ".join(f"{k}:{v}" for k, v in sorted(status_counts.items(), key=lambda x: (-x[1], x[0]))),
            "shared_core_token_count": len(core_tokens),
            "shared_body_token_count": len(body_tokens),
            "shared_secondary_token_count": len(secondary_tokens),
            "shared_core_tokens": " ".join(sorted(core_tokens)[:160]),
            "shared_body_tokens": " ".join(sorted(body_tokens)[:220]),
            "shared_secondary_tokens": " ".join(sorted(secondary_tokens)[:220]),
        })

        for frame in range(birth, end + 1):
            frame_rows.append({
                "frame_index": frame,
                "time_sec": f"{frame / max(args.fps, 1e-9):.9f}",
                "cluster_id": cid,
                "cluster_kind": cluster_kind,
                "identity_count": len(comp),
                "ecology_energy_mean": f"{cluster_density:.9f}",
            })

    _write_csv(
        Path(args.out_instrument_ecology_clusters_csv),
        cluster_rows,
        [
            "cluster_id", "cluster_kind", "identity_count",
            "birth_frame", "end_frame", "duration_frames",
            "ecology_energy_sum", "ecology_energy_mean",
            "dominant_notes", "field_behavior_distribution", "ecology_status_distribution",
            "shared_core_token_count", "shared_body_token_count", "shared_secondary_token_count",
            "shared_core_tokens", "shared_body_tokens", "shared_secondary_tokens",
        ],
    )

    _write_csv(
        Path(args.out_identity_to_cluster_csv),
        mapping_rows,
        ["cluster_id", "identity_id", "attractor_note", "field_behavior", "ecology_energy", "ecology_status"],
    )

    _write_csv(
        Path(args.out_cluster_frame_csv),
        frame_rows,
        ["frame_index", "time_sec", "cluster_id", "cluster_kind", "identity_count", "ecology_energy_mean"],
    )

    summary = {
        "input_nodes": len(nodes),
        "input_links": len(links),
        "clusters": len(cluster_rows),
        "identity_mappings": len(mapping_rows),
        "frame_rows": len(frame_rows),
        "cluster_kind_counts": dict(defaultdict(int, {
            k: sum(1 for r in cluster_rows if r["cluster_kind"] == k)
            for k in sorted(set(r["cluster_kind"] for r in cluster_rows))
        })),
    }

    Path(args.out_summary_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_summary_txt).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()