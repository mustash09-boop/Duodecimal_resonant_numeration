# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
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
    return micro if micro else _tokens(row.get("token_union", ""))


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
    return _token_list(raw if raw else row.get("observed_roots", ""))


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


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def _structured_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, float]:
    micro_j = _jaccard(a.get("micro_tokens", set()), b.get("micro_tokens", set()))
    coarse_j = _jaccard(a.get("coarse_tokens", set()), b.get("coarse_tokens", set()))
    return {
        "micro_jaccard": micro_j,
        "coarse_jaccard": coarse_j,
        "combined": 0.70 * micro_j + 0.30 * coarse_j,
    }


def _trajectory_signature(row: Dict[str, Any]) -> Dict[str, Any]:
    micro_tokens = _row_micro_tokens(row)
    coarse_tokens = _row_coarse_tokens(row)
    return {
        "micro_tokens": micro_tokens,
        "coarse_tokens": coarse_tokens,
        "tokens": micro_tokens,  # backward-compatible alias
        "roots_micro": _row_roots_micro(row),
        "roots_coarse": _row_roots_coarse(row),
        "roots": _row_roots_micro(row),  # backward-compatible alias
        "signatures": _token_list(row.get("topology_signatures", "")),
        "birth_frame": _safe_int(row.get("birth_frame"), 0),
        "end_frame": _safe_int(row.get("end_frame"), 0),
        "duration_frames": _safe_int(row.get("duration_frames"), 0),
        "frame_count": _safe_int(row.get("frame_count"), 0),
        "mean_score": _safe_float(row.get("mean_family_score"), 0.0),
        "coherence": _safe_float(row.get("mean_topology_coherence"), 0.0),
        "micro_coherence": _safe_float(row.get("mean_micro_topology_coherence"), 0.0),
        "coarse_coherence": _safe_float(row.get("mean_coarse_topology_coherence"), 0.0),
        "group_coherence": _safe_float(row.get("group_pairwise_topology_coherence"), 0.0),
    }


def _ordered_flow_similarity(a_items: List[str], b_items: List[str], tail: int = 8) -> float:
    ar = a_items[-tail:]
    br = b_items[:tail]
    if not ar or not br:
        return 0.0
    return len(set(ar) & set(br)) / max(len(set(ar) | set(br)), 1)


def _root_flow_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, float]:
    micro = _ordered_flow_similarity(a["roots_micro"], b["roots_micro"])
    coarse = _ordered_flow_similarity(a["roots_coarse"], b["roots_coarse"])
    return {"micro": micro, "coarse": coarse, "combined": 0.70 * micro + 0.30 * coarse}


def _signature_flow_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    asig = set(a["signatures"][-8:])
    bsig = set(b["signatures"][:8])
    if not asig or not bsig:
        return 0.0
    return len(asig & bsig) / max(len(asig | bsig), 1)


def _trajectory_score(a: Dict[str, Any], b: Dict[str, Any], max_gap: int) -> Dict[str, float]:
    gap = b["birth_frame"] - a["end_frame"]
    if gap < 0 or gap > max_gap:
        return {"score": -999.0, "micro_token_sim": 0.0, "coarse_token_sim": 0.0, "combined_token_sim": 0.0, "root_flow": 0.0, "signature_flow": 0.0, "coherence": 0.0, "gap": float(gap)}

    token_sim = _structured_similarity(a, b)
    root_flow = _root_flow_similarity(a, b)
    sig_flow = _signature_flow_similarity(a, b)
    coherence = max(
        min(a.get("coherence", 0.0), b.get("coherence", 0.0)),
        min(a.get("group_coherence", 0.0), b.get("group_coherence", 0.0)),
    )

    score = (
        token_sim["combined"] * 0.42
        + token_sim["micro_jaccard"] * 0.12
        + root_flow["combined"] * 0.20
        + sig_flow * 0.16
        + coherence * 0.10
        - gap * 0.015
    )

    return {
        "score": score,
        "micro_token_sim": token_sim["micro_jaccard"],
        "coarse_token_sim": token_sim["coarse_jaccard"],
        "combined_token_sim": token_sim["combined"],
        "root_flow": root_flow["combined"],
        "signature_flow": sig_flow,
        "coherence": coherence,
        "gap": float(gap),
    }


def _mean_pairwise_group_coherence(group: List[Dict[str, Any]]) -> Dict[str, float]:
    if len(group) <= 1:
        return {"combined": 1.0, "micro": 1.0, "coarse": 1.0}

    sigs = [_trajectory_signature(r) for r in group]
    combined_vals: list[float] = []
    micro_vals: list[float] = []
    coarse_vals: list[float] = []

    for i, a in enumerate(sigs):
        for b in sigs[i + 1:]:
            sim = _structured_similarity(a, b)
            combined_vals.append(sim["combined"])
            micro_vals.append(sim["micro_jaccard"])
            coarse_vals.append(sim["coarse_jaccard"])

    def avg(xs: list[float]) -> float:
        return sum(xs) / max(len(xs), 1)

    return {"combined": avg(combined_vals), "micro": avg(micro_vals), "coarse": avg(coarse_vals)}


def _would_keep_group_coherent(group: List[Dict[str, Any]], candidate: Dict[str, Any], *, min_group_coherence: float, max_coherence_drop: float) -> bool:
    before = _mean_pairwise_group_coherence(group)
    after = _mean_pairwise_group_coherence(group + [candidate])
    if after["combined"] < min_group_coherence:
        return False
    if before["combined"] - after["combined"] > max_coherence_drop:
        return False
    return True


def _mode(xs: List[str]) -> str:
    counts: Dict[str, int] = {}
    for x in xs:
        if x:
            counts[x] = counts.get(x, 0) + 1
    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0] if counts else ""


def _merge_group(group: List[Dict[str, Any]], tid: int) -> Dict[str, Any]:
    group = sorted(group, key=lambda r: _safe_int(r.get("birth_frame"), 0))
    birth = min(_safe_int(r.get("birth_frame"), 0) for r in group)
    end = max(_safe_int(r.get("end_frame"), 0) for r in group)
    source_ids = [str(r.get("stable_entity_id", r.get("entity_id", ""))) for r in group]

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

    n = max(len(group), 1)
    group_coherence = _mean_pairwise_group_coherence(group)

    return {
        "trajectory_entity_id": tid,
        "source_stable_entity_ids": " ".join(source_ids),
        "birth_frame": birth,
        "end_frame": end,
        "duration_frames": end - birth + 1,
        "frame_count": total_frames,
        "segment_count": len(group),
        "mean_family_score": f"{total_score / max(total_frames, 1):.9f}",
        "max_family_score": f"{max_score:.9f}",
        "mean_topology_coherence": f"{coherence_sum / n:.9f}",
        "mean_micro_topology_coherence": f"{micro_coherence_sum / n:.9f}",
        "mean_coarse_topology_coherence": f"{coarse_coherence_sum / n:.9f}",
        "trajectory_pairwise_topology_coherence": f"{group_coherence['combined']:.9f}",
        "trajectory_pairwise_micro_coherence": f"{group_coherence['micro']:.9f}",
        "trajectory_pairwise_coarse_coherence": f"{group_coherence['coarse']:.9f}",
        "token_union_micro_count": len(token_union_micro),
        "token_union_coarse_count": len(token_union_coarse),
        "token_union_count": len(token_union_micro),
        "topology_signature_count": len(signatures),
        "root_hint_micro_not_identity": _mode(roots_micro),
        "root_hint_coarse_not_identity": _mode(roots_coarse),
        "root_hint_not_identity": _mode(roots_micro),
        "observed_roots_micro": " ".join(roots_micro[:240]),
        "observed_roots_coarse": " ".join(roots_coarse[:240]),
        "observed_roots": " ".join(roots_micro[:240]),
        "topology_signatures": " ".join(sorted(signatures)[:160]),
        "token_union_micro": " ".join(sorted(token_union_micro)[:260]),
        "token_union_coarse": " ".join(sorted(token_union_coarse)[:260]),
        "token_union": " ".join(sorted(token_union_micro)[:260]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Trajectory-aware persistence for structured resonance entities.")
    ap.add_argument("--stable_entities_csv", required=True)
    ap.add_argument("--out_trajectory_entities_csv", required=True)
    ap.add_argument("--out_mapping_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)
    ap.add_argument("--max_gap_frames", type=int, default=18)
    ap.add_argument("--min_trajectory_score", type=float, default=0.24)
    ap.add_argument("--min_micro_token_sim", type=float, default=0.05)
    ap.add_argument("--min_coarse_token_sim", type=float, default=0.15)
    ap.add_argument("--min_group_coherence", type=float, default=0.14)
    ap.add_argument("--max_coherence_drop", type=float, default=0.35)
    ap.add_argument("--min_frames", type=int, default=4)
    args = ap.parse_args()

    rows = _load_csv(Path(args.stable_entities_csv))
    rows.sort(key=lambda r: (_safe_int(r.get("birth_frame"), 0), _safe_int(r.get("stable_entity_id", r.get("entity_id", 0)))))

    used = set()
    groups: List[List[Dict[str, Any]]] = []
    signatures = [_trajectory_signature(r) for r in rows]

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

                details = _trajectory_score(signatures[current_i], signatures[j], args.max_gap_frames)

                if details["score"] < args.min_trajectory_score:
                    continue
                if details["micro_token_sim"] < args.min_micro_token_sim and details["coarse_token_sim"] < args.min_coarse_token_sim:
                    continue
                if not _would_keep_group_coherent(group, cand, min_group_coherence=args.min_group_coherence, max_coherence_drop=args.max_coherence_drop):
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

    trajectory_rows = []
    mapping_rows = []

    for tid, group in enumerate(groups, start=1):
        merged = _merge_group(group, tid)
        if _safe_int(merged.get("frame_count"), 0) < args.min_frames:
            continue

        trajectory_rows.append(merged)

        for src in group:
            mapping_rows.append({
                "trajectory_entity_id": tid,
                "source_stable_entity_id": src.get("stable_entity_id", src.get("entity_id", "")),
                "source_birth_frame": src.get("birth_frame", ""),
                "source_end_frame": src.get("end_frame", ""),
                "source_root_hint_micro_not_identity": _root_hint_micro(src),
                "source_root_hint_coarse_not_identity": _root_hint_coarse(src),
                "source_root_hint_not_identity": _root_hint_micro(src),
            })

    trajectory_rows.sort(key=lambda r: (_safe_int(r.get("birth_frame"), 0), _safe_int(r.get("trajectory_entity_id"), 0)))

    out_traj = Path(args.out_trajectory_entities_csv)
    out_map = Path(args.out_mapping_csv)
    out_meta = Path(args.out_meta_json)
    out_txt = Path(args.out_summary_txt)
    out_traj.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "trajectory_entity_id", "source_stable_entity_ids", "birth_frame", "end_frame",
        "duration_frames", "frame_count", "segment_count", "mean_family_score", "max_family_score",
        "mean_topology_coherence", "mean_micro_topology_coherence", "mean_coarse_topology_coherence",
        "trajectory_pairwise_topology_coherence", "trajectory_pairwise_micro_coherence", "trajectory_pairwise_coarse_coherence",
        "token_union_micro_count", "token_union_coarse_count", "token_union_count",
        "topology_signature_count", "root_hint_micro_not_identity", "root_hint_coarse_not_identity", "root_hint_not_identity",
        "observed_roots_micro", "observed_roots_coarse", "observed_roots",
        "topology_signatures", "token_union_micro", "token_union_coarse", "token_union",
    ]

    with out_traj.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(trajectory_rows)

    with out_map.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "trajectory_entity_id", "source_stable_entity_id", "source_birth_frame", "source_end_frame",
            "source_root_hint_micro_not_identity", "source_root_hint_coarse_not_identity", "source_root_hint_not_identity",
        ])
        w.writeheader()
        w.writerows(mapping_rows)

    segment_distribution: Dict[int, int] = {}
    for r in trajectory_rows:
        n = _safe_int(r.get("segment_count"), 0)
        segment_distribution[n] = segment_distribution.get(n, 0) + 1

    meta = {
        "stage": "resonance_entity_trajectory_persistence",
        "semantic_version": "structured_micro_coarse_trajectory_v2",
        "inputs": {"stable_entities_csv": args.stable_entities_csv},
        "outputs": {
            "trajectory_entities_csv": args.out_trajectory_entities_csv,
            "mapping_csv": args.out_mapping_csv,
            "meta_json": args.out_meta_json,
            "summary_txt": args.out_summary_txt,
        },
        "parameters": {
            "max_gap_frames": args.max_gap_frames,
            "min_trajectory_score": args.min_trajectory_score,
            "min_micro_token_sim": args.min_micro_token_sim,
            "min_coarse_token_sim": args.min_coarse_token_sim,
            "min_group_coherence": args.min_group_coherence,
            "max_coherence_drop": args.max_coherence_drop,
            "min_frames": args.min_frames,
            "similarity_model": {
                "micro_token_weight": 0.70,
                "coarse_token_weight": 0.30,
                "rule": "Micro identity is primary; coarse topology and root flow support trajectory persistence but cannot replace micro continuity.",
            },
        },
        "result": {
            "input_stable_entities": len(rows),
            "trajectory_entities": len(trajectory_rows),
            "mapping_rows": len(mapping_rows),
            "segment_distribution": segment_distribution,
        },
        "ontology_note": (
            "Trajectory identity may persist through continuous topology evolution, "
            "but micro/coarse topology is preserved separately and greedy local links "
            "are rejected when they damage whole-trajectory coherence."
        ),
    }
    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    txt = [
        "RESONANCE ENTITY TRAJECTORY PERSISTENCE",
        "=" * 72,
        f"stable_entities_csv : {args.stable_entities_csv}",
        "",
        f"input_stable_entities : {len(rows)}",
        f"trajectory_entities   : {len(trajectory_rows)}",
        f"mapping_rows          : {len(mapping_rows)}",
        "",
        "Segment distribution:",
    ]
    for k in sorted(segment_distribution):
        txt.append(f"  {k}: {segment_distribution[k]}")
    txt.extend([
        "",
        "Principle:",
        "  Entity identity may persist through continuous topology evolution.",
        "  Similarity is not only snapshot overlap, but trajectory flow.",
        "  Micro identity and coarse ownership topology are preserved separately.",
        "  Greedy trajectory links are rejected if they damage whole-trajectory coherence.",
        "",
    ])
    out_txt.write_text("\n".join(txt), encoding="utf-8")

    print("resonance entity trajectory persistence complete")
    print(json.dumps(meta["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
