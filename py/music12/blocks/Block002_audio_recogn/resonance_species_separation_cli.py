# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
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


def _sim(a: float, b: float, scale: float = 1.0) -> float:
    return max(0.0, 1.0 - min(abs(a - b) / max(scale, 1e-9), 1.0))


def _token_similarity(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def _cluster_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    score = 0.0
    score += _sim(_safe_float(a.get("mean_turbulence")), _safe_float(b.get("mean_turbulence")), 0.45) * 0.18
    score += _sim(_safe_float(a.get("mean_living_breath")), _safe_float(b.get("mean_living_breath")), 0.45) * 0.16
    score += _sim(_safe_float(a.get("mean_mechanical_sustain")), _safe_float(b.get("mean_mechanical_sustain")), 0.35) * 0.12
    score += _sim(_safe_float(a.get("mean_rebirth_irregularity")), _safe_float(b.get("mean_rebirth_irregularity")), 0.45) * 0.14
    score += _sim(_safe_float(a.get("asymmetry_score")), _safe_float(b.get("asymmetry_score")), 0.50) * 0.14
    score += _sim(_safe_float(a.get("uniformity_score")), _safe_float(b.get("uniformity_score")), 0.50) * 0.10
    score += (1.0 if a.get("regional_type") == b.get("regional_type") else 0.0) * 0.10
    score += (1.0 if a.get("cluster_kind") == b.get("cluster_kind") else 0.0) * 0.06
    return max(0.0, min(score, 1.0))


def _components(graph: Dict[str, Set[str]], nodes: Set[str]) -> List[Set[str]]:
    visited = set()
    comps = []

    for n in sorted(nodes, key=lambda x: _safe_int(x, 0)):
        if n in visited:
            continue

        stack = [n]
        comp = set()

        while stack:
            cur = stack.pop()
            if cur in visited:
                continue

            visited.add(cur)
            comp.add(cur)

            for nxt in graph.get(cur, set()):
                if nxt not in visited:
                    stack.append(nxt)

        comps.append(comp)

    return comps


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Separate resonance ecology clusters into behavioral species before naming instruments."
    )

    ap.add_argument("--cluster_asymmetry_csv", required=True)
    ap.add_argument("--instrument_ecology_clusters_csv", required=True)
    ap.add_argument("--identity_to_cluster_csv", required=True)
    ap.add_argument("--attractor_events_csv", required=True)

    ap.add_argument("--out_species_csv", required=True)
    ap.add_argument("--out_cluster_to_species_csv", required=True)
    ap.add_argument("--out_identity_to_species_csv", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--min_species_similarity", type=float, default=0.56)

    args = ap.parse_args()

    asym_rows = _load_csv(Path(args.cluster_asymmetry_csv))
    cluster_rows = _load_csv(Path(args.instrument_ecology_clusters_csv))
    identity_map_rows = _load_csv(Path(args.identity_to_cluster_csv))
    attractor_rows = _load_csv(Path(args.attractor_events_csv))

    asym_by_cluster = {str(r.get("cluster_id", "")).strip(): r for r in asym_rows}
    ecology_by_cluster = {str(r.get("cluster_id", "")).strip(): r for r in cluster_rows}
    attractor_by_id = {str(r.get("identity_id", "")).strip(): r for r in attractor_rows}

    ids_by_cluster: Dict[str, List[str]] = defaultdict(list)
    for r in identity_map_rows:
        cid = str(r.get("cluster_id", "")).strip()
        iid = str(r.get("identity_id", "")).strip()
        if cid and iid:
            ids_by_cluster[cid].append(iid)

    nodes = set(asym_by_cluster.keys())
    graph: Dict[str, Set[str]] = defaultdict(set)

    node_list = sorted(nodes, key=lambda x: _safe_int(x, 0))

    for i, a_id in enumerate(node_list):
        for b_id in node_list[i + 1:]:
            s = _cluster_similarity(asym_by_cluster[a_id], asym_by_cluster[b_id])
            if s >= args.min_species_similarity:
                graph[a_id].add(b_id)
                graph[b_id].add(a_id)

    comps = _components(graph, nodes)

    species_rows = []
    cluster_to_species = []
    identity_to_species = []

    for sid, comp in enumerate(comps, start=1):
        regional_counts = Counter()
        kind_counts = Counter()
        note_counts = Counter()
        breath_class_counts = Counter()

        turbulences = []
        living_scores = []
        mechanical_scores = []
        asym_scores = []
        uniform_scores = []
        body_tokens = set()
        secondary_tokens = set()

        identities = []

        for cid in comp:
            asym = asym_by_cluster.get(cid, {})
            eco = ecology_by_cluster.get(cid, {})

            regional_counts[str(asym.get("regional_type", "")).strip()] += 1
            kind_counts[str(asym.get("cluster_kind", "")).strip()] += 1

            turbulences.append(_safe_float(asym.get("mean_turbulence")))
            living_scores.append(_safe_float(asym.get("mean_living_breath")))
            mechanical_scores.append(_safe_float(asym.get("mean_mechanical_sustain")))
            asym_scores.append(_safe_float(asym.get("asymmetry_score")))
            uniform_scores.append(_safe_float(asym.get("uniformity_score")))

            body_tokens |= _tokens(eco.get("shared_body_tokens", ""))
            secondary_tokens |= _tokens(eco.get("shared_secondary_tokens", ""))

            for iid in ids_by_cluster.get(cid, []):
                identities.append(iid)
                att = attractor_by_id.get(iid, {})
                note = str(att.get("attractor_note", "")).strip()
                if note:
                    note_counts[note] += 1

            cluster_to_species.append({
                "species_id": sid,
                "cluster_id": cid,
                "cluster_kind": asym.get("cluster_kind", ""),
                "regional_type": asym.get("regional_type", ""),
                "asymmetry_score": asym.get("asymmetry_score", ""),
                "uniformity_score": asym.get("uniformity_score", ""),
            })

        for iid in identities:
            att = attractor_by_id.get(iid, {})
            identity_to_species.append({
                "species_id": sid,
                "identity_id": iid,
                "attractor_note": att.get("attractor_note", ""),
                "attractor_status": att.get("attractor_status", ""),
                "field_behavior": att.get("field_behavior", ""),
            })

        mean_turb = sum(turbulences) / max(len(turbulences), 1)
        mean_living = sum(living_scores) / max(len(living_scores), 1)
        mean_mechanical = sum(mechanical_scores) / max(len(mechanical_scores), 1)
        mean_asym = sum(asym_scores) / max(len(asym_scores), 1)
        mean_uniform = sum(uniform_scores) / max(len(uniform_scores), 1)

        if mean_asym > mean_uniform and mean_asym >= 0.48:
            species_type = "LIVING_BODY_SPECIES"
        elif mean_uniform >= 0.58:
            species_type = "SYNTHETIC_UNIFORM_SPECIES"
        elif mean_living >= 0.70 and mean_turb >= 0.62:
            species_type = "TURBULENT_BODY_SPECIES"
        else:
            species_type = "MIXED_BODY_SPECIES"

        species_rows.append({
            "species_id": sid,
            "species_type": species_type,
            "cluster_count": len(comp),
            "identity_count": len(identities),
            "mean_turbulence": f"{mean_turb:.9f}",
            "mean_living_breath": f"{mean_living:.9f}",
            "mean_mechanical_sustain": f"{mean_mechanical:.9f}",
            "mean_asymmetry": f"{mean_asym:.9f}",
            "mean_uniformity": f"{mean_uniform:.9f}",
            "regional_distribution": " | ".join(f"{k}:{v}" for k, v in regional_counts.most_common()),
            "cluster_kind_distribution": " | ".join(f"{k}:{v}" for k, v in kind_counts.most_common()),
            "dominant_notes": " | ".join(f"{k}:{v}" for k, v in note_counts.most_common(24)),
            "body_token_count": len(body_tokens),
            "secondary_token_count": len(secondary_tokens),
            "body_tokens": " ".join(sorted(body_tokens)[:260]),
            "secondary_tokens": " ".join(sorted(secondary_tokens)[:260]),
        })

    _write_csv(
        Path(args.out_species_csv),
        species_rows,
        [
            "species_id",
            "species_type",
            "cluster_count",
            "identity_count",
            "mean_turbulence",
            "mean_living_breath",
            "mean_mechanical_sustain",
            "mean_asymmetry",
            "mean_uniformity",
            "regional_distribution",
            "cluster_kind_distribution",
            "dominant_notes",
            "body_token_count",
            "secondary_token_count",
            "body_tokens",
            "secondary_tokens",
        ],
    )

    _write_csv(
        Path(args.out_cluster_to_species_csv),
        cluster_to_species,
        ["species_id", "cluster_id", "cluster_kind", "regional_type", "asymmetry_score", "uniformity_score"],
    )

    _write_csv(
        Path(args.out_identity_to_species_csv),
        identity_to_species,
        ["species_id", "identity_id", "attractor_note", "attractor_status", "field_behavior"],
    )

    summary = {
        "species_count": len(species_rows),
        "cluster_count": len(cluster_to_species),
        "identity_count": len(identity_to_species),
        "species_type_distribution": dict(Counter(r["species_type"] for r in species_rows)),
    }

    Path(args.out_summary_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_summary_txt).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()