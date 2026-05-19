# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if not s:
            return default
        return float(s.replace(",", "."))
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if not s:
            return default
        return int(float(s.replace(",", ".")))
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


def _split_tokens(raw: Any) -> Set[str]:
    return {x.strip() for x in str(raw or "").replace(",", " ").replace("|", " ").split() if x.strip()}


def _join(tokens: Iterable[str], limit: int = 96) -> str:
    return " ".join(sorted(set(tokens))[:limit])


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _tokens(row: Dict[str, Any], *cols: str) -> Set[str]:
    out = set()
    for c in cols:
        out.update(_split_tokens(row.get(c, "")))
    return out


def _lineage_core_tokens(row: Dict[str, Any]) -> Set[str]:
    return _tokens(row, "parent_tokens", "offspring_tokens")


def _lineage_box_tokens(row: Dict[str, Any]) -> Set[str]:
    return _tokens(row, "residual_box_tokens", "box_residual_signature")


def _lineage_strength(row: Dict[str, Any]) -> float:
    return _safe_float(row.get("lineage_strength"), 0.0)


def _h57(row: Dict[str, Any]) -> float:
    return _safe_float(row.get("harmonic_5_7_parenthood_score"), 0.0)


def _territory_base_score(row: Dict[str, Any]) -> float:
    strength = _lineage_strength(row)
    h57 = _h57(row)
    parenthood = _safe_float(row.get("harmonic_parenthood_score"), 0.0)
    offspring_count = min(_safe_float(row.get("offspring_count"), 0.0) / 8.0, 1.0)
    delayed_count = min(_safe_float(row.get("delayed_offspring_count"), 0.0) / 4.0, 1.0)
    residual_box = min(_safe_float(row.get("residual_box_count"), 0.0) / 96.0, 1.0)
    return _clamp(strength * 0.30 + h57 * 0.26 + parenthood * 0.18 + offspring_count * 0.10 + delayed_count * 0.08 + residual_box * 0.08)


def _incoming_outgoing_maps(links: List[Dict[str, Any]]) -> tuple[dict, dict]:
    incoming = defaultdict(list)
    outgoing = defaultdict(list)
    for e in links:
        s = str(e.get("source_lineage_id", "")).strip()
        t = str(e.get("target_lineage_id", "")).strip()
        if s:
            outgoing[s].append(e)
        if t:
            incoming[t].append(e)
    return incoming, outgoing


def _branch_energy_for(lid: str, incoming: dict, outgoing: dict) -> tuple[float, float, float]:
    out_e = sum(_safe_float(e.get("branch_score"), 0.0) for e in outgoing.get(lid, []))
    in_e = sum(_safe_float(e.get("branch_score"), 0.0) for e in incoming.get(lid, []))
    return in_e, out_e, in_e + out_e


def _shared_pressure(lid: str, incoming: dict, outgoing: dict, relation: str) -> float:
    rows = incoming.get(lid, []) + outgoing.get(lid, [])
    if not rows:
        return 0.0
    shared = [_safe_float(e.get("branch_score"), 0.0) for e in rows if str(e.get("branch_relation", "")).strip() == relation]
    return sum(shared) / max(len(rows), 1)


def _territory_role(*, base_score: float, branch_total: float, branch_in: float, branch_out: float, shared_offspring: float, shared_box: float) -> tuple[str, float]:
    branch_balance = 1.0 - min(abs(branch_out - branch_in) / max(branch_total, 1e-9), 1.0)
    outgoing_bias = max(branch_out - branch_in, 0.0) / max(branch_total, 1e-9)
    incoming_bias = max(branch_in - branch_out, 0.0) / max(branch_total, 1e-9)

    ownership_score = base_score * 0.48 + outgoing_bias * 0.16 + (1.0 - shared_box) * 0.12 + shared_offspring * 0.12 + branch_balance * 0.12
    secondary_score = base_score * 0.30 + incoming_bias * 0.18 + branch_balance * 0.20 + shared_offspring * 0.18 + shared_box * 0.14
    sympathetic_score = shared_offspring * 0.28 + shared_box * 0.26 + incoming_bias * 0.18 + (1.0 - base_score) * 0.18 + branch_balance * 0.10
    transient_score = min(branch_total / 8.0, 1.0) * 0.10 + incoming_bias * 0.22 + (1.0 - base_score) * 0.28 + shared_box * 0.18 + shared_offspring * 0.22

    scores = {
        "DOMINANT_LINEAGE_TERRITORY": ownership_score,
        "SECONDARY_LINEAGE_TERRITORY": secondary_score,
        "SYMPATHETIC_RESONANCE_TERRITORY": sympathetic_score,
        "TRANSIENT_SHARED_TERRITORY": transient_score,
    }
    role = max(scores, key=scores.get)
    return role, scores[role]


def _territory_boundary_strength(base_score: float, shared_offspring: float, shared_box: float, branch_total: float) -> float:
    return _clamp(base_score * 0.54 + shared_offspring * 0.18 + (1.0 - shared_box) * 0.18 + min(branch_total / 8.0, 1.0) * 0.10)


def _boundary_label(x: float) -> str:
    if x >= 0.64:
        return "CLEAR_TERRITORY_BOUNDARY"
    if x >= 0.42:
        return "SOFT_TERRITORY_BOUNDARY"
    if x >= 0.24:
        return "WEAK_TERRITORY_BOUNDARY"
    return "OPEN_SHARED_FIELD"


def _competing_roots_for(lid: str, links: List[Dict[str, Any]], root_by_id: Dict[str, str]) -> Set[str]:
    roots = set()
    for e in links:
        s = str(e.get("source_lineage_id", "")).strip()
        t = str(e.get("target_lineage_id", "")).strip()
        if s == lid and t in root_by_id:
            roots.add(root_by_id[t])
        elif t == lid and s in root_by_id:
            roots.add(root_by_id[s])
    return roots


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Resolve lineage territories inside shared polyphonic resonance field. "
            "This does not break universal resonance dependency; it separates dominant, "
            "secondary, sympathetic and transient ownership inside the shared field."
        )
    )
    ap.add_argument("--lineages_csv", required=True)
    ap.add_argument("--branch_links_csv", required=True)
    ap.add_argument("--branch_nodes_csv", required=True)
    ap.add_argument("--out_territory_nodes_csv", required=True)
    ap.add_argument("--out_territory_links_csv", required=True)
    ap.add_argument("--out_territory_frame_csv", required=True)
    ap.add_argument("--out_readable_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)
    ap.add_argument("--fps", type=float, default=60.0)
    args = ap.parse_args()

    lineages = _load_csv(Path(args.lineages_csv))
    links = _load_csv(Path(args.branch_links_csv))
    branch_nodes = _load_csv(Path(args.branch_nodes_csv))
    incoming, outgoing = _incoming_outgoing_maps(links)
    lineage_by_id = {str(r.get("lineage_id", "")).strip(): r for r in lineages}
    branch_node_by_id = {str(r.get("lineage_id", "")).strip(): r for r in branch_nodes}
    root_by_id = {lid: str(r.get("root_candidate", "")).strip() for lid, r in lineage_by_id.items()}

    territory_nodes = []
    territory_frame = []
    readable = []
    role_counts = defaultdict(int)
    boundary_counts = defaultdict(int)

    for lid, row in lineage_by_id.items():
        branch_node = branch_node_by_id.get(lid, {})
        base_score = _territory_base_score(row)
        branch_in, branch_out, branch_total = _branch_energy_for(lid, incoming, outgoing)
        shared_offspring = _shared_pressure(lid, incoming, outgoing, "SHARED_HARMONIC_OFFSPRING")
        shared_box = _shared_pressure(lid, incoming, outgoing, "SHARED_BOX_RESIDUAL_FIELD")
        territory_role, territory_score = _territory_role(
            base_score=base_score,
            branch_total=branch_total,
            branch_in=branch_in,
            branch_out=branch_out,
            shared_offspring=shared_offspring,
            shared_box=shared_box,
        )
        boundary_strength = _territory_boundary_strength(base_score, shared_offspring, shared_box, branch_total)
        boundary = _boundary_label(boundary_strength)
        competing_roots = _competing_roots_for(lid, links, root_by_id)
        birth = _safe_int(row.get("birth_frame"), 0)
        end = _safe_int(row.get("end_frame"), birth)

        territory_nodes.append({
            "lineage_id": lid,
            "root_candidate": row.get("root_candidate", ""),
            "root_candidate_micro": row.get("root_candidate_micro", ""),
            "territory_role": territory_role,
            "territory_score": f"{territory_score:.9f}",
            "territory_boundary": boundary,
            "territory_boundary_strength": f"{boundary_strength:.9f}",
            "lineage_strength": row.get("lineage_strength", ""),
            "harmonic_5_7_parenthood_score": row.get("harmonic_5_7_parenthood_score", ""),
            "harmonic_parenthood_score": row.get("harmonic_parenthood_score", ""),
            "territory_base_score": f"{base_score:.9f}",
            "branch_in_energy": f"{branch_in:.9f}",
            "branch_out_energy": f"{branch_out:.9f}",
            "branch_total_energy": f"{branch_total:.9f}",
            "branch_count": branch_node.get("branch_count", "0"),
            "branch_node_status": branch_node.get("branch_node_status", ""),
            "shared_offspring_pressure": f"{shared_offspring:.9f}",
            "shared_box_pressure": f"{shared_box:.9f}",
            "competing_root_count": len(competing_roots),
            "competing_roots": " ".join(sorted(competing_roots)[:64]),
            "residual_box_count": row.get("residual_box_count", ""),
            "residual_box_tokens": row.get("residual_box_tokens", ""),
            "present_harmonics": row.get("present_harmonics", ""),
            "missing_harmonics": row.get("missing_harmonics", ""),
            "offspring_tokens": row.get("offspring_tokens", ""),
            "register_class": row.get("register_class", ""),
            "birth_frame": birth,
            "end_frame": end,
            "duration_frames": row.get("duration_frames", ""),
        })
        role_counts[territory_role] += 1
        boundary_counts[boundary] += 1
        readable.append({
            "lineage_id": lid,
            "root": row.get("root_candidate", ""),
            "role": territory_role,
            "boundary": boundary,
            "score": f"{territory_score:.3f}",
            "base": f"{base_score:.3f}",
            "shared_offspring": f"{shared_offspring:.3f}",
            "shared_box": f"{shared_box:.3f}",
            "competing": len(competing_roots),
        })
        for frame in range(birth, end + 1):
            territory_frame.append({
                "frame_index": frame,
                "time_sec": f"{frame / max(args.fps, 1e-9):.9f}",
                "lineage_id": lid,
                "root_candidate": row.get("root_candidate", ""),
                "territory_role": territory_role,
                "territory_score": f"{territory_score:.9f}",
                "territory_boundary": boundary,
                "territory_boundary_strength": f"{boundary_strength:.9f}",
            })

    role_by_id = {r["lineage_id"]: r["territory_role"] for r in territory_nodes}
    territory_links = []
    link_kind_counts = defaultdict(int)

    for e in links:
        s = str(e.get("source_lineage_id", "")).strip()
        t = str(e.get("target_lineage_id", "")).strip()
        s_role = role_by_id.get(s, "")
        t_role = role_by_id.get(t, "")
        branch_score = _safe_float(e.get("branch_score"), 0.0)
        if s_role == "DOMINANT_LINEAGE_TERRITORY" and t_role != "DOMINANT_LINEAGE_TERRITORY":
            territory_relation = "DOMINANT_TO_DEPENDENT_TERRITORY"
        elif t_role == "DOMINANT_LINEAGE_TERRITORY" and s_role != "DOMINANT_LINEAGE_TERRITORY":
            territory_relation = "DEPENDENT_TO_DOMINANT_TERRITORY"
        elif s_role == t_role and branch_score >= 0.42:
            territory_relation = "CO_TERRITORIAL_RESONANCE"
        elif "SYMPATHETIC" in s_role or "SYMPATHETIC" in t_role:
            territory_relation = "SYMPATHETIC_TERRITORY_BRIDGE"
        else:
            territory_relation = "SHARED_TERRITORY_BRIDGE"
        link_kind_counts[territory_relation] += 1
        territory_links.append({
            "source_lineage_id": s,
            "target_lineage_id": t,
            "source_root": e.get("source_root", ""),
            "target_root": e.get("target_root", ""),
            "source_territory_role": s_role,
            "target_territory_role": t_role,
            "territory_relation": territory_relation,
            "branch_kind": e.get("branch_kind", ""),
            "branch_relation": e.get("branch_relation", ""),
            "branch_score": e.get("branch_score", ""),
            "h57_bridge": e.get("h57_bridge", ""),
            "box_shared": e.get("box_shared", ""),
            "harmonic_shared": e.get("harmonic_shared", ""),
            "token_shared": e.get("token_shared", ""),
        })

    territory_nodes.sort(key=lambda r: (_safe_int(r.get("birth_frame"), 0), -_safe_float(r.get("territory_score"), 0.0)))

    node_fields = [
        "lineage_id", "root_candidate", "root_candidate_micro", "territory_role", "territory_score",
        "territory_boundary", "territory_boundary_strength", "lineage_strength",
        "harmonic_5_7_parenthood_score", "harmonic_parenthood_score", "territory_base_score",
        "branch_in_energy", "branch_out_energy", "branch_total_energy", "branch_count", "branch_node_status",
        "shared_offspring_pressure", "shared_box_pressure", "competing_root_count", "competing_roots",
        "residual_box_count", "residual_box_tokens", "present_harmonics", "missing_harmonics", "offspring_tokens",
        "register_class", "birth_frame", "end_frame", "duration_frames",
    ]
    _write_csv(Path(args.out_territory_nodes_csv), territory_nodes, node_fields)

    link_fields = [
        "source_lineage_id", "target_lineage_id", "source_root", "target_root",
        "source_territory_role", "target_territory_role", "territory_relation",
        "branch_kind", "branch_relation", "branch_score", "h57_bridge", "box_shared", "harmonic_shared", "token_shared",
    ]
    _write_csv(Path(args.out_territory_links_csv), territory_links, link_fields)

    frame_fields = [
        "frame_index", "time_sec", "lineage_id", "root_candidate", "territory_role", "territory_score",
        "territory_boundary", "territory_boundary_strength",
    ]
    _write_csv(Path(args.out_territory_frame_csv), territory_frame, frame_fields)

    readable_fields = ["lineage_id", "root", "role", "boundary", "score", "base", "shared_offspring", "shared_box", "competing"]
    _write_csv(Path(args.out_readable_csv), readable, readable_fields)

    meta = {
        "stage": "lineage_territory_resolver",
        "semantic_version": "lineage_territory_resolver_v1",
        "inputs": {
            "lineages_csv": args.lineages_csv,
            "branch_links_csv": args.branch_links_csv,
            "branch_nodes_csv": args.branch_nodes_csv,
        },
        "outputs": {
            "territory_nodes_csv": args.out_territory_nodes_csv,
            "territory_links_csv": args.out_territory_links_csv,
            "territory_frame_csv": args.out_territory_frame_csv,
            "readable_csv": args.out_readable_csv,
            "meta_json": args.out_meta_json,
            "summary_txt": args.out_summary_txt,
        },
        "parameters": {"fps": args.fps},
        "result": {
            "lineages": len(lineages),
            "branch_links": len(links),
            "territory_nodes": len(territory_nodes),
            "territory_links": len(territory_links),
            "territory_frame_rows": len(territory_frame),
            "territory_role_counts": dict(role_counts),
            "territory_boundary_counts": dict(boundary_counts),
            "territory_link_relation_counts": dict(link_kind_counts),
        },
        "ontology_note": (
            "Everything is connected by common tuning and resonance dependency. "
            "This module does not break the shared field; it separates how lineages "
            "belong inside it: dominant, secondary, sympathetic or transient. "
            "The goal is territory governance, not resonance isolation."
        ),
    }
    Path(args.out_meta_json).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    txt = [
        "LINEAGE TERRITORY RESOLVER",
        "=" * 72,
        f"lineages_csv      : {args.lineages_csv}",
        f"branch_links_csv  : {args.branch_links_csv}",
        f"branch_nodes_csv  : {args.branch_nodes_csv}",
        "",
        f"lineages          : {len(lineages)}",
        f"branch_links      : {len(links)}",
        f"territory_nodes   : {len(territory_nodes)}",
        f"territory_links   : {len(territory_links)}",
        f"territory_frames  : {len(territory_frame)}",
        "",
        "Territory role counts:",
    ]
    for k in sorted(role_counts):
        txt.append(f"  {k}: {role_counts[k]}")
    txt.append("")
    txt.append("Territory boundary counts:")
    for k in sorted(boundary_counts):
        txt.append(f"  {k}: {boundary_counts[k]}")
    txt.append("")
    txt.append("Territory link relation counts:")
    for k in sorted(link_kind_counts):
        txt.append(f"  {k}: {link_kind_counts[k]}")
    txt.extend([
        "",
        "Principle:",
        "  Resonance commonality is not an error.",
        "  Common tuning makes the field connected.",
        "  The task is to identify belonging:",
        "  what is born from what, what continues in what,",
        "  and what is only sympathetic or transient.",
        "",
    ])
    Path(args.out_summary_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_summary_txt).write_text("\n".join(txt), encoding="utf-8")

    print("lineage territory resolver complete")
    print(json.dumps(meta["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
