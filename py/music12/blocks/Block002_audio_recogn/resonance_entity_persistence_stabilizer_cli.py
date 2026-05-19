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


def _root_micro(row: Dict[str, Any]) -> str:
    return str(row.get("root_hint_micro_not_identity", "")).strip() or str(row.get("root_hint_not_identity", "")).strip()


def _root_coarse(row: Dict[str, Any]) -> str:
    return str(row.get("root_hint_coarse_not_identity", "")).strip() or (_token_coarse(_root_micro(row)) if _root_micro(row) else "")


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def _entity_signature(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "micro_tokens": _row_micro_tokens(row),
        "coarse_tokens": _row_coarse_tokens(row),
        "birth_frame": _safe_int(row.get("birth_frame"), 0),
        "end_frame": _safe_int(row.get("end_frame"), 0),
    }


def _structured_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, float]:
    micro_j = _jaccard(a.get("micro_tokens", set()), b.get("micro_tokens", set()))
    coarse_j = _jaccard(a.get("coarse_tokens", set()), b.get("coarse_tokens", set()))
    combined = 0.70 * micro_j + 0.30 * coarse_j
    return {"micro_jaccard": micro_j, "coarse_jaccard": coarse_j, "combined": combined}


def _gap(a: Dict[str, Any], b: Dict[str, Any]) -> int:
    return _safe_int(b.get("birth_frame"), 0) - _safe_int(a.get("end_frame"), 0)


def _can_merge_pair(
    a: Dict[str, Any],
    b: Dict[str, Any],
    *,
    max_gap: int,
    min_union_jaccard: float,
    min_micro_jaccard: float,
    min_coarse_jaccard: float,
) -> bool:
    gap = _gap(a, b)
    if gap < 0 or gap > max_gap:
        return False

    sim = _structured_similarity(_entity_signature(a), _entity_signature(b))
    if sim["combined"] < min_union_jaccard:
        return False

    if sim["micro_jaccard"] < min_micro_jaccard and sim["coarse_jaccard"] < min_coarse_jaccard:
        return False

    return True


def _merge_candidate_score(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    sim = _structured_similarity(_entity_signature(a), _entity_signature(b))
    gap = max(0, _gap(a, b))
    score_b = _safe_float(b.get("mean_family_score"), 0.0)
    return 0.70 * sim["combined"] + 0.20 * sim["micro_jaccard"] + 0.10 * sim["coarse_jaccard"] + 0.01 * score_b - 0.015 * gap


def _mean_pairwise_group_coherence(group: List[Dict[str, Any]]) -> Dict[str, float]:
    if len(group) <= 1:
        return {"combined": 1.0, "micro": 1.0, "coarse": 1.0}

    combined_vals: list[float] = []
    micro_vals: list[float] = []
    coarse_vals: list[float] = []

    sigs = [_entity_signature(r) for r in group]

    for i, a in enumerate(sigs):
        for b in sigs[i + 1:]:
            sim = _structured_similarity(a, b)
            combined_vals.append(sim["combined"])
            micro_vals.append(sim["micro_jaccard"])
            coarse_vals.append(sim["coarse_jaccard"])

    def avg(xs: list[float]) -> float:
        return sum(xs) / max(len(xs), 1)

    return {"combined": avg(combined_vals), "micro": avg(micro_vals), "coarse": avg(coarse_vals)}


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


def _merge_group(group: List[Dict[str, Any]], stable_id: int) -> Dict[str, Any]:
    group = sorted(group, key=lambda r: _safe_int(r.get("birth_frame"), 0))

    birth = min(_safe_int(r.get("birth_frame"), 0) for r in group)
    end = max(_safe_int(r.get("end_frame"), 0) for r in group)

    source_ids = [str(r.get("entity_id", "")) for r in group]

    token_union_micro: Set[str] = set()
    token_union_coarse: Set[str] = set()
    signatures: Set[str] = set()
    roots_micro: List[str] = []
    roots_coarse: List[str] = []

    total_score = 0.0
    total_frames = 0
    max_score = 0.0
    topology_coherence_sum = 0.0
    micro_coherence_sum = 0.0
    coarse_coherence_sum = 0.0

    for r in group:
        frame_count = max(_safe_int(r.get("frame_count"), 0), 1)
        mean_score = _safe_float(r.get("mean_family_score"), 0.0)

        total_score += mean_score * frame_count
        total_frames += frame_count
        max_score = max(max_score, _safe_float(r.get("max_family_score"), 0.0))

        topology_coherence_sum += _safe_float(r.get("mean_topology_coherence"), 0.0)
        micro_coherence_sum += _safe_float(r.get("mean_micro_topology_coherence"), 0.0)
        coarse_coherence_sum += _safe_float(r.get("mean_coarse_topology_coherence"), 0.0)

        token_union_micro |= _row_micro_tokens(r)
        token_union_coarse |= _row_coarse_tokens(r)
        signatures |= _tokens(r.get("topology_signatures", ""))

        rm = _root_micro(r)
        rc = _root_coarse(r)
        if rm:
            roots_micro.append(rm)
        if rc:
            roots_coarse.append(rc)

    mean_score = total_score / max(total_frames, 1)
    n = max(len(group), 1)
    group_coherence = _mean_pairwise_group_coherence(group)

    def mode(xs: List[str]) -> str:
        counts: Dict[str, int] = {}
        for x in xs:
            if x:
                counts[x] = counts.get(x, 0) + 1
        return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0] if counts else ""

    root_hint_micro = mode(roots_micro)
    root_hint_coarse = mode(roots_coarse)

    return {
        "stable_entity_id": stable_id,
        "source_entity_ids": " ".join(source_ids),
        "birth_frame": birth,
        "end_frame": end,
        "duration_frames": end - birth + 1,
        "frame_count": total_frames,
        "segment_count": len(group),
        "mean_family_score": f"{mean_score:.9f}",
        "max_family_score": f"{max_score:.9f}",
        "mean_topology_coherence": f"{topology_coherence_sum / n:.9f}",
        "mean_micro_topology_coherence": f"{micro_coherence_sum / n:.9f}",
        "mean_coarse_topology_coherence": f"{coarse_coherence_sum / n:.9f}",
        "group_pairwise_topology_coherence": f"{group_coherence['combined']:.9f}",
        "group_pairwise_micro_coherence": f"{group_coherence['micro']:.9f}",
        "group_pairwise_coarse_coherence": f"{group_coherence['coarse']:.9f}",
        "token_union_micro_count": len(token_union_micro),
        "token_union_coarse_count": len(token_union_coarse),
        "token_union_count": len(token_union_micro),
        "topology_signature_count": len(signatures),
        "root_hint_micro_not_identity": root_hint_micro,
        "root_hint_coarse_not_identity": root_hint_coarse,
        "root_hint_not_identity": root_hint_micro,
        "observed_roots_micro": " ".join(roots_micro[:160]),
        "observed_roots_coarse": " ".join(roots_coarse[:160]),
        "observed_roots": " ".join(roots_micro[:160]),
        "topology_signatures": " ".join(sorted(signatures)[:100]),
        "token_union_micro": " ".join(sorted(token_union_micro)[:220]),
        "token_union_coarse": " ".join(sorted(token_union_coarse)[:220]),
        "token_union": " ".join(sorted(token_union_micro)[:220]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stabilize resonance entities through structured topology evolution."
    )

    ap.add_argument("--entities_csv", required=True)

    ap.add_argument("--out_stable_entities_csv", required=True)
    ap.add_argument("--out_mapping_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--max_merge_gap_frames", type=int, default=10)
    ap.add_argument("--min_union_jaccard", type=float, default=0.22)
    ap.add_argument("--min_micro_jaccard", type=float, default=0.08)
    ap.add_argument("--min_coarse_jaccard", type=float, default=0.18)
    ap.add_argument("--min_group_coherence", type=float, default=0.16)
    ap.add_argument("--max_coherence_drop", type=float, default=0.32)
    ap.add_argument("--min_stable_frames", type=int, default=4)

    args = ap.parse_args()

    rows = _load_csv(Path(args.entities_csv))
    rows.sort(key=lambda r: (_safe_int(r.get("birth_frame"), 0), _safe_int(r.get("entity_id"), 0)))

    used = set()
    groups: List[List[Dict[str, Any]]] = []

    for i, r in enumerate(rows):
        if i in used:
            continue

        group = [r]
        used.add(i)
        current = r

        while True:
            best_j = None
            best_score = -999.0

            for j, cand in enumerate(rows):
                if j in used:
                    continue

                if not _can_merge_pair(
                    current,
                    cand,
                    max_gap=args.max_merge_gap_frames,
                    min_union_jaccard=args.min_union_jaccard,
                    min_micro_jaccard=args.min_micro_jaccard,
                    min_coarse_jaccard=args.min_coarse_jaccard,
                ):
                    continue

                if not _would_keep_group_coherent(
                    group,
                    cand,
                    min_group_coherence=args.min_group_coherence,
                    max_coherence_drop=args.max_coherence_drop,
                ):
                    continue

                score = _merge_candidate_score(current, cand)

                if score > best_score:
                    best_score = score
                    best_j = j

            if best_j is None:
                break

            nxt = rows[best_j]
            group.append(nxt)
            used.add(best_j)
            current = nxt

        groups.append(group)

    stable_rows = []
    mapping_rows = []

    for sid, group in enumerate(groups, start=1):
        merged = _merge_group(group, sid)

        if _safe_int(merged["frame_count"], 0) < args.min_stable_frames:
            continue

        stable_rows.append(merged)

        for src in group:
            mapping_rows.append({
                "stable_entity_id": sid,
                "source_entity_id": src.get("entity_id", ""),
                "source_birth_frame": src.get("birth_frame", ""),
                "source_end_frame": src.get("end_frame", ""),
                "source_root_hint_micro_not_identity": _root_micro(src),
                "source_root_hint_coarse_not_identity": _root_coarse(src),
                "source_root_hint_not_identity": _root_micro(src),
            })

    stable_rows.sort(key=lambda r: (_safe_int(r.get("birth_frame"), 0), _safe_int(r.get("stable_entity_id"), 0)))

    out_stable = Path(args.out_stable_entities_csv)
    out_mapping = Path(args.out_mapping_csv)
    out_meta = Path(args.out_meta_json)
    out_txt = Path(args.out_summary_txt)

    out_stable.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "stable_entity_id", "source_entity_ids", "birth_frame", "end_frame",
        "duration_frames", "frame_count", "segment_count",
        "mean_family_score", "max_family_score",
        "mean_topology_coherence", "mean_micro_topology_coherence", "mean_coarse_topology_coherence",
        "group_pairwise_topology_coherence", "group_pairwise_micro_coherence", "group_pairwise_coarse_coherence",
        "token_union_micro_count", "token_union_coarse_count", "token_union_count",
        "topology_signature_count",
        "root_hint_micro_not_identity", "root_hint_coarse_not_identity", "root_hint_not_identity",
        "observed_roots_micro", "observed_roots_coarse", "observed_roots",
        "topology_signatures", "token_union_micro", "token_union_coarse", "token_union",
    ]

    with out_stable.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(stable_rows)

    with out_mapping.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "stable_entity_id", "source_entity_id", "source_birth_frame", "source_end_frame",
                "source_root_hint_micro_not_identity", "source_root_hint_coarse_not_identity",
                "source_root_hint_not_identity",
            ],
        )
        w.writeheader()
        w.writerows(mapping_rows)

    segment_distribution: Dict[int, int] = {}
    for r in stable_rows:
        n = _safe_int(r.get("segment_count"), 0)
        segment_distribution[n] = segment_distribution.get(n, 0) + 1

    meta = {
        "stage": "resonance_entity_persistence_stabilizer",
        "semantic_version": "structured_micro_coarse_group_coherence_v2",
        "inputs": {"entities_csv": args.entities_csv},
        "outputs": {
            "stable_entities_csv": args.out_stable_entities_csv,
            "mapping_csv": args.out_mapping_csv,
            "meta_json": args.out_meta_json,
            "summary_txt": args.out_summary_txt,
        },
        "parameters": {
            "max_merge_gap_frames": args.max_merge_gap_frames,
            "min_union_jaccard": args.min_union_jaccard,
            "min_micro_jaccard": args.min_micro_jaccard,
            "min_coarse_jaccard": args.min_coarse_jaccard,
            "min_group_coherence": args.min_group_coherence,
            "max_coherence_drop": args.max_coherence_drop,
            "min_stable_frames": args.min_stable_frames,
            "similarity_model": {
                "micro_jaccard_weight": 0.70,
                "coarse_jaccard_weight": 0.30,
                "rule": "Micro identity is primary; coarse topology supports continuity; candidate merges must preserve group-level coherence.",
            },
        },
        "result": {
            "input_entities": len(rows),
            "stable_entities": len(stable_rows),
            "mapping_rows": len(mapping_rows),
            "segment_distribution": segment_distribution,
        },
        "ontology_note": (
            "This stabilizer preserves micro/coarse topology separately and rejects merges "
            "that would reduce global group coherence too much. Observed roots remain hints, not identity."
        ),
    }

    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    txt = [
        "RESONANCE ENTITY PERSISTENCE STABILIZER",
        "=" * 72,
        f"entities_csv    : {args.entities_csv}",
        "",
        f"input_entities  : {len(rows)}",
        f"stable_entities : {len(stable_rows)}",
        f"mapping_rows    : {len(mapping_rows)}",
        "",
        "Segment distribution:",
    ]

    for k in sorted(segment_distribution):
        txt.append(f"  {k}: {segment_distribution[k]}")

    txt.extend([
        "",
        "Principle:",
        "  A resonance entity may evolve in topology without losing identity.",
        "  Micro identity and coarse ownership neighborhood are preserved separately.",
        "  This module rejects local greedy merges that damage whole-group coherence.",
        "",
    ])

    out_txt.write_text("\n".join(txt), encoding="utf-8")

    print("resonance entity persistence stabilizer complete")
    print(json.dumps(meta["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
