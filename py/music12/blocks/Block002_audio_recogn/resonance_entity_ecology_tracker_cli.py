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


# ============================================================
# Token helpers
# ============================================================

def _split_token_micro(token: str) -> tuple[str, str]:
    token = str(token or "").strip()
    if "'" not in token:
        return token, ""
    coarse, micro = token.split("'", 1)
    return coarse, micro or "-"


def _token_coarse(token: str) -> str:
    coarse, _micro = _split_token_micro(token)
    return coarse


def _tokens(raw: str) -> Set[str]:
    return {x.strip() for x in str(raw or "").split() if x.strip()}


def _token_list(raw: str) -> List[str]:
    return [x.strip() for x in str(raw or "").split() if x.strip()]


def _tokens_json_or_space(raw_json: str, raw_space: str = "") -> Set[str]:
    raw_json = str(raw_json or "").strip()

    if raw_json:
        try:
            data = json.loads(raw_json)
            if isinstance(data, list):
                return {str(x).strip() for x in data if str(x).strip()}
        except Exception:
            pass

    return _tokens(raw_space)


def _row_micro_tokens(row: Dict[str, Any]) -> Set[str]:
    micro = _tokens_json_or_space(
        str(row.get("token_union_micro_json", "")),
        str(row.get("token_union_micro", "")),
    )
    if micro:
        return micro
    return _tokens(row.get("token_union", ""))


def _row_coarse_tokens(row: Dict[str, Any]) -> Set[str]:
    coarse = _tokens_json_or_space(
        str(row.get("token_union_coarse_json", "")),
        str(row.get("token_union_coarse", "")),
    )
    if coarse:
        return coarse
    return {_token_coarse(t) for t in _row_micro_tokens(row)}


def _row_roots_micro(row: Dict[str, Any]) -> List[str]:
    raw = str(row.get("observed_roots_micro", "")).strip()
    if raw:
        return _token_list(raw)
    return _token_list(row.get("observed_roots", ""))


def _row_roots_coarse(row: Dict[str, Any]) -> List[str]:
    raw = str(row.get("observed_roots_coarse", "")).strip()
    if raw:
        return _token_list(raw)
    return [_token_coarse(t) for t in _row_roots_micro(row)]


def _root_hint_micro(row: Dict[str, Any]) -> str:
    return (
        str(row.get("root_hint_micro_not_identity", "")).strip()
        or str(row.get("root_hint_not_identity", "")).strip()
    )


def _root_hint_coarse(row: Dict[str, Any]) -> str:
    return (
        str(row.get("root_hint_coarse_not_identity", "")).strip()
        or (_token_coarse(_root_hint_micro(row)) if _root_hint_micro(row) else "")
    )


# ============================================================
# Ecology loading
# ============================================================

def _entity_id(row: Dict[str, Any]) -> str:
    return str(
        row.get(
            "trajectory_entity_id",
            row.get("stable_entity_id", row.get("entity_id", "")),
        )
    ).strip()


def _load_overlap_ecology(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load overlap ecology from an overlap CSV.

    Old model:
        entity_id -> set(neighbor_ids)

    New model:
        entity_id -> {
            neighbors: set[str],
            weighted_neighbors: dict[str, float],
            stable_neighbors: set[str],
            transient_neighbors: set[str],
        }

    If the overlap CSV contains no score/duration fields, every edge is kept
    with weight 1.0 and treated as stable enough for compatibility.
    """
    rows = _load_csv(path)

    edge_weights: Dict[tuple[str, str], float] = defaultdict(float)
    edge_counts: Dict[tuple[str, str], int] = defaultdict(int)

    for r in rows:
        a = str(r.get("entity_a", r.get("entity_id_a", ""))).strip()
        b = str(r.get("entity_b", r.get("entity_id_b", ""))).strip()

        if not a or not b or a == b:
            continue

        # Prefer structured overlap/coexistence columns when present.
        weight = 0.0
        for key in (
            "micro_topology_jaccard",
            "topology_jaccard",
            "overlap_score",
            "co_presence_score",
            "coexistence_score",
        ):
            if key in r and str(r.get(key, "")).strip() != "":
                weight = max(weight, _safe_float(r.get(key), 0.0))

        if weight <= 0.0:
            weight = 1.0

        edge = tuple(sorted((a, b)))
        edge_weights[edge] += weight
        edge_counts[edge] += 1

    ecology: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "neighbors": set(),
            "weighted_neighbors": {},
            "stable_neighbors": set(),
            "transient_neighbors": set(),
        }
    )

    for (a, b), weight_sum in edge_weights.items():
        count = edge_counts[(a, b)]
        weight_avg = weight_sum / max(count, 1)

        # A simple first distinction:
        # - repeated relation or non-trivial score => stable
        # - one weak encounter => transient
        is_stable = count >= 2 or weight_avg >= 0.20

        for src, dst in ((a, b), (b, a)):
            ecology[src]["neighbors"].add(dst)
            ecology[src]["weighted_neighbors"][dst] = weight_avg
            if is_stable:
                ecology[src]["stable_neighbors"].add(dst)
            else:
                ecology[src]["transient_neighbors"].add(dst)

    return ecology


# ============================================================
# Similarity / scoring
# ============================================================

def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def _structured_token_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, float]:
    micro = _jaccard(a.get("micro_tokens", set()), b.get("micro_tokens", set()))
    coarse = _jaccard(a.get("coarse_tokens", set()), b.get("coarse_tokens", set()))
    combined = 0.70 * micro + 0.30 * coarse

    return {
        "micro": micro,
        "coarse": coarse,
        "combined": combined,
    }


def _weighted_neighbor_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, float]:
    """
    Ecology context is supportive evidence only.

    We use:
      - stable neighbor overlap
      - all neighbor overlap
      - approximate weighted overlap

    But ecology is never allowed to replace direct topology continuity.
    """
    a_stable = a.get("stable_neighbors", set())
    b_stable = b.get("stable_neighbors", set())
    a_all = a.get("neighbors", set())
    b_all = b.get("neighbors", set())

    stable_sim = _jaccard(a_stable, b_stable)
    all_sim = _jaccard(a_all, b_all)

    aw = a.get("weighted_neighbors", {})
    bw = b.get("weighted_neighbors", {})

    common = set(aw) & set(bw)
    union = set(aw) | set(bw)

    if not union:
        weighted_sim = 0.0
    else:
        numerator = sum(min(float(aw.get(k, 0.0)), float(bw.get(k, 0.0))) for k in common)
        denominator = sum(max(float(aw.get(k, 0.0)), float(bw.get(k, 0.0))) for k in union)
        weighted_sim = numerator / denominator if denominator > 0 else 0.0

    combined = 0.55 * stable_sim + 0.25 * weighted_sim + 0.20 * all_sim

    return {
        "stable": stable_sim,
        "all": all_sim,
        "weighted": weighted_sim,
        "combined": combined,
    }


def _signature(row: Dict[str, Any], ecology: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    eid = _entity_id(row)
    eco = ecology.get(
        eid,
        {
            "neighbors": set(),
            "weighted_neighbors": {},
            "stable_neighbors": set(),
            "transient_neighbors": set(),
        },
    )

    return {
        "id": eid,
        "birth_frame": _safe_int(row.get("birth_frame"), 0),
        "end_frame": _safe_int(row.get("end_frame"), 0),
        "duration_frames": _safe_int(row.get("duration_frames"), 0),
        "frame_count": _safe_int(row.get("frame_count"), 0),

        "micro_tokens": _row_micro_tokens(row),
        "coarse_tokens": _row_coarse_tokens(row),

        # Backward-compatible alias.
        "tokens": _row_micro_tokens(row),

        "signatures": _tokens(row.get("topology_signatures", "")),
        "roots_micro": set(_row_roots_micro(row)),
        "roots_coarse": set(_row_roots_coarse(row)),

        # Backward-compatible alias.
        "roots": set(_row_roots_micro(row)),

        "neighbors": set(eco.get("neighbors", set())),
        "stable_neighbors": set(eco.get("stable_neighbors", set())),
        "transient_neighbors": set(eco.get("transient_neighbors", set())),
        "weighted_neighbors": dict(eco.get("weighted_neighbors", {})),

        "mean_score": _safe_float(row.get("mean_family_score"), 0.0),
        "coherence": _safe_float(row.get("mean_topology_coherence"), 0.0),
        "micro_coherence": _safe_float(row.get("mean_micro_topology_coherence"), 0.0),
        "coarse_coherence": _safe_float(row.get("mean_coarse_topology_coherence"), 0.0),
        "trajectory_coherence": _safe_float(row.get("trajectory_pairwise_topology_coherence"), 0.0),
    }


def _ecology_score(
    a: Dict[str, Any],
    b: Dict[str, Any],
    max_gap: int,
) -> Dict[str, float]:
    gap = b["birth_frame"] - a["end_frame"]
    if gap < 0 or gap > max_gap:
        return {
            "score": -999.0,
            "direct_topology": 0.0,
            "micro_token_sim": 0.0,
            "coarse_token_sim": 0.0,
            "topology_signature_sim": 0.0,
            "root_sim": 0.0,
            "neighbor_sim": 0.0,
            "stable_neighbor_sim": 0.0,
            "coherence": 0.0,
            "gap": float(gap),
        }

    token_sim = _structured_token_similarity(a, b)
    topo_sim = _jaccard(a["signatures"], b["signatures"])

    root_micro = _jaccard(a["roots_micro"], b["roots_micro"])
    root_coarse = _jaccard(a["roots_coarse"], b["roots_coarse"])
    root_sim = 0.70 * root_micro + 0.30 * root_coarse

    neighbor_sim = _weighted_neighbor_similarity(a, b)

    coherence = max(
        min(a.get("coherence", 0.0), b.get("coherence", 0.0)),
        min(a.get("trajectory_coherence", 0.0), b.get("trajectory_coherence", 0.0)),
    )

    direct_topology = 0.62 * token_sim["combined"] + 0.23 * topo_sim + 0.15 * root_sim

    # Ecology is context, not identity. Keep neighbor contribution small.
    score = 0.0
    score += token_sim["combined"] * 0.38
    score += token_sim["micro"] * 0.12
    score += topo_sim * 0.18
    score += root_sim * 0.12
    score += neighbor_sim["combined"] * 0.10
    score += coherence * 0.10
    score -= gap * 0.012

    return {
        "score": score,
        "direct_topology": direct_topology,
        "micro_token_sim": token_sim["micro"],
        "coarse_token_sim": token_sim["coarse"],
        "topology_signature_sim": topo_sim,
        "root_sim": root_sim,
        "neighbor_sim": neighbor_sim["combined"],
        "stable_neighbor_sim": neighbor_sim["stable"],
        "coherence": coherence,
        "gap": float(gap),
    }


def _mean_pairwise_group_coherence(group: List[Dict[str, Any]]) -> Dict[str, float]:
    if len(group) <= 1:
        return {
            "combined": 1.0,
            "micro": 1.0,
            "coarse": 1.0,
            "ecology": 1.0,
        }

    sigs = [_signature(r, {}) for r in group]

    combined_vals: list[float] = []
    micro_vals: list[float] = []
    coarse_vals: list[float] = []

    for i, a in enumerate(sigs):
        for b in sigs[i + 1:]:
            sim = _structured_token_similarity(a, b)
            combined_vals.append(sim["combined"])
            micro_vals.append(sim["micro"])
            coarse_vals.append(sim["coarse"])

    def avg(xs: list[float]) -> float:
        return sum(xs) / max(len(xs), 1)

    return {
        "combined": avg(combined_vals),
        "micro": avg(micro_vals),
        "coarse": avg(coarse_vals),
        "ecology": 0.0,
    }


def _would_keep_group_coherent(
    group: List[Dict[str, Any]],
    candidate: Dict[str, Any],
    *,
    min_group_coherence: float,
    max_coherence_drop: float,
) -> bool:
    before = _mean_pairwise_group_coherence(group)
    after = _mean_pairwise_group_coherence(group + [candidate])

    if after["combined"] < min_group_coherence:
        return False

    if before["combined"] - after["combined"] > max_coherence_drop:
        return False

    return True


# ============================================================
# Merge summarization
# ============================================================

def _mode(xs: List[str]) -> str:
    counts: Dict[str, int] = {}
    for x in xs:
        if x:
            counts[x] = counts.get(x, 0) + 1
    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0] if counts else ""


def _merge_group(group: List[Dict[str, Any]], eid: int) -> Dict[str, Any]:
    group = sorted(group, key=lambda r: _safe_int(r.get("birth_frame"), 0))

    birth = min(_safe_int(r.get("birth_frame"), 0) for r in group)
    end = max(_safe_int(r.get("end_frame"), 0) for r in group)

    source_ids = [_entity_id(r) for r in group]

    token_union_micro: Set[str] = set()
    token_union_coarse: Set[str] = set()
    signatures: Set[str] = set()

    roots_micro: List[str] = []
    roots_coarse: List[str] = []

    total_score = 0.0
    total_frames = 0
    max_score = 0.0

    coherence_sum = 0.0
    micro_coherence_sum = 0.0
    coarse_coherence_sum = 0.0

    for r in group:
        fc = max(_safe_int(r.get("frame_count"), 0), 1)
        mean_score = _safe_float(r.get("mean_family_score"), 0.0)

        total_score += mean_score * fc
        total_frames += fc
        max_score = max(max_score, _safe_float(r.get("max_family_score"), 0.0))

        coherence_sum += _safe_float(r.get("mean_topology_coherence"), 0.0)
        micro_coherence_sum += _safe_float(r.get("mean_micro_topology_coherence"), 0.0)
        coarse_coherence_sum += _safe_float(r.get("mean_coarse_topology_coherence"), 0.0)

        token_union_micro |= _row_micro_tokens(r)
        token_union_coarse |= _row_coarse_tokens(r)
        signatures |= _tokens(r.get("topology_signatures", ""))

        roots_micro.extend(_row_roots_micro(r))
        roots_coarse.extend(_row_roots_coarse(r))

    group_coherence = _mean_pairwise_group_coherence(group)
    n = max(len(group), 1)

    root_hint_micro = _mode(roots_micro)
    root_hint_coarse = _mode(roots_coarse)

    return {
        "ecology_entity_id": eid,
        "source_trajectory_entity_ids": " ".join(source_ids),
        "birth_frame": birth,
        "end_frame": end,
        "duration_frames": end - birth + 1,
        "frame_count": total_frames,
        "segment_count": len(group),

        "mean_family_score": f"{(total_score / max(total_frames, 1)):.9f}",
        "max_family_score": f"{max_score:.9f}",

        "mean_topology_coherence": f"{(coherence_sum / n):.9f}",
        "mean_micro_topology_coherence": f"{(micro_coherence_sum / n):.9f}",
        "mean_coarse_topology_coherence": f"{(coarse_coherence_sum / n):.9f}",

        "ecology_pairwise_topology_coherence": f"{group_coherence['combined']:.9f}",
        "ecology_pairwise_micro_coherence": f"{group_coherence['micro']:.9f}",
        "ecology_pairwise_coarse_coherence": f"{group_coherence['coarse']:.9f}",

        "token_union_micro_count": len(token_union_micro),
        "token_union_coarse_count": len(token_union_coarse),

        # Backward-compatible alias.
        "token_union_count": len(token_union_micro),

        "topology_signature_count": len(signatures),

        "root_hint_micro_not_identity": root_hint_micro,
        "root_hint_coarse_not_identity": root_hint_coarse,

        # Backward-compatible alias.
        "root_hint_not_identity": root_hint_micro,

        "observed_roots_micro": " ".join(roots_micro[:260]),
        "observed_roots_coarse": " ".join(roots_coarse[:260]),

        # Backward-compatible alias.
        "observed_roots": " ".join(roots_micro[:260]),

        "topology_signatures": " ".join(sorted(signatures)[:180]),

        "token_union_micro": " ".join(sorted(token_union_micro)[:300]),
        "token_union_coarse": " ".join(sorted(token_union_coarse)[:300]),

        # Backward-compatible alias.
        "token_union": " ".join(sorted(token_union_micro)[:300]),
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Track resonance entity identity through ecological context "
            "without allowing ecology to replace direct topology continuity."
        )
    )

    ap.add_argument("--trajectory_entities_csv", required=True)
    ap.add_argument("--entity_overlap_csv", required=True)

    ap.add_argument("--out_ecology_entities_csv", required=True)
    ap.add_argument("--out_mapping_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--max_gap_frames", type=int, default=24)
    ap.add_argument("--min_ecology_score", type=float, default=0.22)

    # Anti-crowd protection: ecology cannot preserve identity alone.
    ap.add_argument("--min_direct_topology", type=float, default=0.10)
    ap.add_argument("--min_micro_token_sim", type=float, default=0.04)
    ap.add_argument("--min_coarse_token_sim", type=float, default=0.12)

    ap.add_argument("--min_group_coherence", type=float, default=0.12)
    ap.add_argument("--max_coherence_drop", type=float, default=0.36)

    ap.add_argument("--min_frames", type=int, default=4)

    args = ap.parse_args()

    rows = _load_csv(Path(args.trajectory_entities_csv))
    ecology = _load_overlap_ecology(Path(args.entity_overlap_csv))

    rows.sort(
        key=lambda r: (
            _safe_int(r.get("birth_frame"), 0),
            _safe_int(_entity_id(r), 0),
        )
    )

    sigs = [_signature(r, ecology) for r in rows]

    used = set()
    groups: List[List[Dict[str, Any]]] = []

    for i, row in enumerate(rows):
        if i in used:
            continue

        group = [row]
        used.add(i)
        current_i = i

        while True:
            best_j = None
            best_score = -999.0

            for j, cand in enumerate(rows):
                if j in used:
                    continue

                details = _ecology_score(
                    sigs[current_i],
                    sigs[j],
                    args.max_gap_frames,
                )

                if details["score"] < args.min_ecology_score:
                    continue

                # Direct topology continuity is mandatory.
                # Neighbor similarity alone must not create identity continuity.
                if details["direct_topology"] < args.min_direct_topology:
                    continue

                if (
                    details["micro_token_sim"] < args.min_micro_token_sim
                    and details["coarse_token_sim"] < args.min_coarse_token_sim
                ):
                    continue

                if not _would_keep_group_coherent(
                    group,
                    cand,
                    min_group_coherence=args.min_group_coherence,
                    max_coherence_drop=args.max_coherence_drop,
                ):
                    continue

                if details["score"] > best_score:
                    best_score = details["score"]
                    best_j = j

            if best_j is None:
                break

            group.append(rows[best_j])
            used.add(best_j)
            current_i = best_j

        groups.append(group)

    ecology_rows = []
    mapping_rows = []

    for eid, group in enumerate(groups, start=1):
        merged = _merge_group(group, eid)

        if _safe_int(merged["frame_count"], 0) < args.min_frames:
            continue

        ecology_rows.append(merged)

        for src in group:
            mapping_rows.append({
                "ecology_entity_id": eid,
                "source_trajectory_entity_id": _entity_id(src),
                "source_birth_frame": src.get("birth_frame", ""),
                "source_end_frame": src.get("end_frame", ""),

                "source_root_hint_micro_not_identity": _root_hint_micro(src),
                "source_root_hint_coarse_not_identity": _root_hint_coarse(src),

                # Backward-compatible alias.
                "source_root_hint_not_identity": _root_hint_micro(src),
            })

    ecology_rows.sort(
        key=lambda r: (
            _safe_int(r.get("birth_frame"), 0),
            _safe_int(r.get("ecology_entity_id"), 0),
        )
    )

    out_ecology = Path(args.out_ecology_entities_csv)
    out_mapping = Path(args.out_mapping_csv)
    out_meta = Path(args.out_meta_json)
    out_txt = Path(args.out_summary_txt)

    out_ecology.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "ecology_entity_id",
        "source_trajectory_entity_ids",
        "birth_frame",
        "end_frame",
        "duration_frames",
        "frame_count",
        "segment_count",

        "mean_family_score",
        "max_family_score",

        "mean_topology_coherence",
        "mean_micro_topology_coherence",
        "mean_coarse_topology_coherence",

        "ecology_pairwise_topology_coherence",
        "ecology_pairwise_micro_coherence",
        "ecology_pairwise_coarse_coherence",

        "token_union_micro_count",
        "token_union_coarse_count",
        "token_union_count",

        "topology_signature_count",

        "root_hint_micro_not_identity",
        "root_hint_coarse_not_identity",
        "root_hint_not_identity",

        "observed_roots_micro",
        "observed_roots_coarse",
        "observed_roots",

        "topology_signatures",

        "token_union_micro",
        "token_union_coarse",
        "token_union",
    ]

    with out_ecology.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(ecology_rows)

    with out_mapping.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "ecology_entity_id",
                "source_trajectory_entity_id",
                "source_birth_frame",
                "source_end_frame",
                "source_root_hint_micro_not_identity",
                "source_root_hint_coarse_not_identity",
                "source_root_hint_not_identity",
            ],
        )
        w.writeheader()
        w.writerows(mapping_rows)

    segment_distribution: Dict[int, int] = {}
    for r in ecology_rows:
        n = _safe_int(r.get("segment_count"), 0)
        segment_distribution[n] = segment_distribution.get(n, 0) + 1

    meta = {
        "stage": "resonance_entity_ecology_tracker",
        "semantic_version": "structured_micro_coarse_context_v2",
        "inputs": {
            "trajectory_entities_csv": args.trajectory_entities_csv,
            "entity_overlap_csv": args.entity_overlap_csv,
        },
        "outputs": {
            "ecology_entities_csv": args.out_ecology_entities_csv,
            "mapping_csv": args.out_mapping_csv,
            "meta_json": args.out_meta_json,
            "summary_txt": args.out_summary_txt,
        },
        "parameters": {
            "max_gap_frames": args.max_gap_frames,
            "min_ecology_score": args.min_ecology_score,
            "min_direct_topology": args.min_direct_topology,
            "min_micro_token_sim": args.min_micro_token_sim,
            "min_coarse_token_sim": args.min_coarse_token_sim,
            "min_group_coherence": args.min_group_coherence,
            "max_coherence_drop": args.max_coherence_drop,
            "min_frames": args.min_frames,
            "similarity_model": {
                "token_similarity": "micro 0.70 + coarse 0.30",
                "neighbor_weight_in_score": 0.10,
                "rule": (
                    "Ecology is contextual evidence only. Neighbor similarity alone "
                    "cannot preserve identity; direct topology continuity is mandatory."
                ),
            },
        },
        "result": {
            "input_trajectory_entities": len(rows),
            "ecology_entities": len(ecology_rows),
            "mapping_rows": len(mapping_rows),
            "segment_distribution": segment_distribution,
        },
        "ontology_note": (
            "Entity identity is not only internal topology, but ecology cannot override "
            "direct resonance continuity. Micro/coarse topology is preserved separately. "
            "Neighbor overlap is supportive context, not identity."
        ),
    }

    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    txt = [
        "RESONANCE ENTITY ECOLOGY TRACKER",
        "=" * 72,
        f"trajectory_entities_csv : {args.trajectory_entities_csv}",
        f"entity_overlap_csv      : {args.entity_overlap_csv}",
        "",
        f"input_trajectory_entities : {len(rows)}",
        f"ecology_entities          : {len(ecology_rows)}",
        f"mapping_rows              : {len(mapping_rows)}",
        "",
        "Segment distribution:",
    ]

    for k in sorted(segment_distribution):
        txt.append(f"  {k}: {segment_distribution[k]}")

    txt.extend([
        "",
        "Principle:",
        "  Entity identity is not only internal topology.",
        "  Identity also persists through ecological context:",
        "  neighbors, overlap field and causal acoustic scene position.",
        "  But ecology is contextual evidence, not identity replacement.",
        "  Direct topology continuity is mandatory.",
        "",
    ])

    out_txt.write_text("\n".join(txt), encoding="utf-8")

    print("resonance entity ecology tracker complete")
    print(json.dumps(meta["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
