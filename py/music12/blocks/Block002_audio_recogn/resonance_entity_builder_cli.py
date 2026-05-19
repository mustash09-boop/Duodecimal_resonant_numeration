# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import hashlib
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


def _parse_token_sets(row: Dict[str, Any]) -> tuple[Set[str], Set[str]]:
    micro_tokens: Set[str] = set()
    coarse_tokens: Set[str] = set()

    micro_tokens |= _tokens_json_or_space(
        str(row.get("family_members_micro_json", "")),
        str(row.get("family_members_micro", "")),
    )
    coarse_tokens |= _tokens_json_or_space(
        str(row.get("family_members_coarse_json", "")),
        str(row.get("family_members_coarse", "")),
    )

    root_micro = str(row.get("family_root_note_micro", "")).strip() or str(row.get("family_root_note", "")).strip()
    root_coarse = str(row.get("family_root_note_coarse", "")).strip()

    if root_micro:
        micro_tokens.add(root_micro)
        coarse_tokens.add(root_coarse or _token_coarse(root_micro))
    elif root_coarse:
        coarse_tokens.add(root_coarse)

    legacy_members = _tokens(row.get("family_members", ""))
    if legacy_members:
        micro_tokens |= legacy_members
        coarse_tokens |= {_token_coarse(t) for t in legacy_members}
        
    root_micro_members = _tokens(row.get("root_micro_members", ""))

    if root_micro_members:
        micro_tokens |= root_micro_members
        coarse_tokens |= {_token_coarse(t) for t in root_micro_members}

    if micro_tokens and not coarse_tokens:
        coarse_tokens = {_token_coarse(t) for t in micro_tokens}

    if coarse_tokens and not micro_tokens:
        # Legacy coarse-only input. Kept visible through identical micro/coarse count.
        micro_tokens = set(coarse_tokens)

    return micro_tokens, coarse_tokens


def _token_structure(raw_tokens: Set[str]) -> list[dict[str, str]]:
    out = []
    for token in sorted(raw_tokens):
        coarse, micro = _split_token_micro(token)
        out.append({"token_micro": token, "token_coarse": coarse, "micro_suffix": micro or "-"})
    return out


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def _topology_tags(raw: str) -> Set[str]:
    """
    Semantic topology tags are whitespace-separated lineage classes.

    Important:
      This is intentionally NOT a hash/fingerprint comparison.
      Tags are meant to preserve family resemblance across topology evolution.
    """
    return {x.strip() for x in str(raw or "").split() if x.strip()}


def _topology_tag_similarity(a_raw: str, b_raw: str) -> float:
    return _jaccard(_topology_tags(a_raw), _topology_tags(b_raw))


def _combined_overlap_similarity(token_sim: Dict[str, float], tag_jaccard: float) -> float:
    """
    Overlap is not identity and not mere token equality.

    token_sim["combined"] preserves direct micro/coarse token continuity.
    tag_jaccard preserves topology lineage similarity.

    The semantic topology tags are weighted higher here because the overlap table
    feeds ecology and influence; otherwise real coexistence can become all zero.
    """
    return 0.42 * token_sim["combined"] + 0.58 * tag_jaccard


def _structured_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, float]:
    micro_j = _jaccard(a.get("micro_tokens", set()), b.get("micro_tokens", set()))
    coarse_j = _jaccard(a.get("coarse_tokens", set()), b.get("coarse_tokens", set()))
    combined = 0.68 * micro_j + 0.32 * coarse_j
    return {"micro_jaccard": micro_j, "coarse_jaccard": coarse_j, "combined": combined}


def _topology_signature(
    micro_tokens: Set[str],
    coarse_tokens: Set[str],
    max_items: int = 12,
) -> str:

    micro_count = len(micro_tokens)
    coarse_count = len(coarse_tokens)

    tags = []

    # =========================================================
    # Density
    # =========================================================

    if micro_count <= 1:
        tags.append("micro_sparse")
    elif micro_count <= 3:
        tags.append("micro_medium")
    else:
        tags.append("micro_dense")

    if coarse_count <= 1:
        tags.append("coarse_sparse")
    elif coarse_count <= 3:
        tags.append("coarse_medium")
    else:
        tags.append("coarse_dense")

    # =========================================================
    # Spread
    # =========================================================

    coarse_octaves = set()
    coarse_degrees = set()

    for token in coarse_tokens:
        coarse, _micro = _split_token_micro(token)

        if "." in coarse:
            octave, degree = coarse.split(".", 1)
            coarse_octaves.add(octave)
            coarse_degrees.add(degree)

    if len(coarse_octaves) <= 1:
        tags.append("single_octave")
    elif len(coarse_octaves) <= 2:
        tags.append("dual_octave")
    else:
        tags.append("wide_octave")

    if len(coarse_degrees) <= 2:
        tags.append("narrow_degree")
    elif len(coarse_degrees) <= 5:
        tags.append("medium_degree")
    else:
        tags.append("wide_degree")

    # =========================================================
    # Root behavior
    # =========================================================

    root_like = 0

    for token in micro_tokens:
        if "'" in token:
            _coarse, micro = token.split("'", 1)

            if micro in ("", "-", "i", "a"):
                root_like += 1

    if root_like >= max(1, micro_count // 2):
        tags.append("root_stable")
    else:
        tags.append("root_migrating")

    # =========================================================
    # Micro diversity
    # =========================================================

    micro_suffixes = set()

    for token in micro_tokens:
        _coarse, micro = _split_token_micro(token)
        micro_suffixes.add(micro or "-")

    if len(micro_suffixes) <= 1:
        tags.append("micro_unified")
    elif len(micro_suffixes) <= 3:
        tags.append("micro_varied")
    else:
        tags.append("micro_complex")

    # =========================================================
    # Harmonic family style
    # =========================================================

    if micro_count > coarse_count * 2:
        tags.append("harmonic_expanded")

    if coarse_count > micro_count:
        tags.append("coarse_clustered")

    # =========================================================
    # Final semantic signature
    # =========================================================

    return " ".join(sorted(set(tags)))


def _group_families_by_frame(rows: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        out[_safe_int(r.get("frame_index"), 0)].append(r)
    for frame in out:
        out[frame].sort(key=lambda r: -_safe_float(r.get("family_score"), 0.0))
    return out


def _family_signature(row: Dict[str, Any]) -> Dict[str, Any]:
    micro_tokens, coarse_tokens = _parse_token_sets(row)

    observed_root_micro = str(row.get("family_root_note_micro", "")).strip() or str(row.get("family_root_note", "")).strip()
    observed_root_coarse = str(row.get("family_root_note_coarse", "")).strip() or (
        _token_coarse(observed_root_micro) if observed_root_micro else ""
    )

    if observed_root_micro:
        micro_tokens.add(observed_root_micro)
    if observed_root_coarse:
        coarse_tokens.add(observed_root_coarse)

    return {
        "frame_index": _safe_int(row.get("frame_index"), 0),
        "observed_root_micro": observed_root_micro,
        "observed_root_coarse": observed_root_coarse,
        "family_score": _safe_float(row.get("family_score"), 0.0),
        "evidence_count": _safe_int(row.get("evidence_count"), 0),
        "root_micro_count": _safe_int(row.get("root_micro_count"), 0),
        "root_micro_diversity": _safe_int(row.get("root_micro_diversity"), 0),
        "micro_tokens": micro_tokens,
        "coarse_tokens": coarse_tokens,
        "tokens": micro_tokens,
        "token_structure": _token_structure(micro_tokens),
        "topology_signature": _topology_signature(micro_tokens, coarse_tokens),
        "topology_tags": _topology_signature(micro_tokens, coarse_tokens),
    }


def _can_continue_entity(entity: Dict[str, Any], sig: Dict[str, Any], frame: int, max_gap_frames: int, min_topology_jaccard: float) -> bool:
    gap = frame - _safe_int(entity.get("last_frame"), 0)
    if gap < 0 or gap > max_gap_frames:
        return False
    sim = _structured_similarity(
        {"micro_tokens": entity["last_micro_tokens"], "coarse_tokens": entity["last_coarse_tokens"]},
        sig,
    )
    return sim["combined"] >= min_topology_jaccard


def _entity_match_cost(entity: Dict[str, Any], sig: Dict[str, Any], frame: int) -> float:
    gap = frame - _safe_int(entity.get("last_frame"), 0)
    sim = _structured_similarity(
        {"micro_tokens": entity["last_micro_tokens"], "coarse_tokens": entity["last_coarse_tokens"]},
        sig,
    )
    score = _safe_float(sig.get("family_score"), 0.0)
    return (1.0 - sim["combined"]) * 3.0 + gap * 0.55 - score * 0.04


def _summarize_entity(entity: Dict[str, Any]) -> Dict[str, Any]:
    scores = entity["scores"]
    frames = entity["frames"]

    token_union_micro: Set[str] = set()
    token_union_coarse: Set[str] = set()

    for tset in entity["micro_token_sets"]:
        token_union_micro |= tset
    for tset in entity["coarse_token_sets"]:
        token_union_coarse |= tset

    mean_score = sum(scores) / max(len(scores), 1)
    max_score = max(scores) if scores else 0.0

    def _mode(xs: list[str]) -> str:
        counts: Dict[str, int] = {}
        for x in xs:
            if x:
                counts[x] = counts.get(x, 0) + 1
        return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0] if counts else ""

    roots_micro = entity["observed_roots_micro"]
    roots_coarse = entity["observed_roots_coarse"]

    micro_vals = []
    coarse_vals = []
    combined_vals = []

    prev_micro = None
    prev_coarse = None
    for micro_set, coarse_set in zip(entity["micro_token_sets"], entity["coarse_token_sets"]):
        if prev_micro is not None and prev_coarse is not None:
            sim = _structured_similarity(
                {"micro_tokens": prev_micro, "coarse_tokens": prev_coarse},
                {"micro_tokens": micro_set, "coarse_tokens": coarse_set},
            )
            micro_vals.append(sim["micro_jaccard"])
            coarse_vals.append(sim["coarse_jaccard"])
            combined_vals.append(sim["combined"])
        prev_micro = micro_set
        prev_coarse = coarse_set

    def _avg(xs: list[float]) -> float:
        return sum(xs) / max(len(xs), 1) if xs else 1.0

    topology_signatures = sorted(set(entity["topology_signatures"]))
    token_structure = _token_structure(token_union_micro)

    return {
        "entity_id": entity["entity_id"],
        "birth_frame": min(frames),
        "end_frame": max(frames),
        "duration_frames": max(frames) - min(frames) + 1,
        "frame_count": len(frames),
        "mean_family_score": f"{mean_score:.9f}",
        "max_family_score": f"{max_score:.9f}",
        "mean_topology_coherence": f"{_avg(combined_vals):.9f}",
        "mean_micro_topology_coherence": f"{_avg(micro_vals):.9f}",
        "mean_coarse_topology_coherence": f"{_avg(coarse_vals):.9f}",
        "token_union_micro_count": len(token_union_micro),
        "token_union_coarse_count": len(token_union_coarse),
        "token_union_count": len(token_union_micro),
        "topology_signature_count": len(topology_signatures),
        "root_hint_micro_not_identity": _mode(roots_micro),
        "root_hint_coarse_not_identity": _mode(roots_coarse),
        "root_hint_not_identity": _mode(roots_micro),
        "observed_roots_micro": " ".join(roots_micro),
        "observed_roots_coarse": " ".join(roots_coarse),
        "observed_roots": " ".join(roots_micro),
        "topology_signatures": " ".join(topology_signatures[:30]),
        "token_union_micro": " ".join(sorted(token_union_micro)[:120]),
        "token_union_coarse": " ".join(sorted(token_union_coarse)[:120]),
        "token_union": " ".join(sorted(token_union_micro)[:120]),
        "token_structure_json": json.dumps(token_structure[:240], ensure_ascii=False),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build resonance entities from structured micro/coarse topology, not note identity.")
    ap.add_argument("--micro_family_csv", required=True)
    ap.add_argument("--out_entities_csv", required=True)
    ap.add_argument("--out_entity_frames_csv", required=True)
    ap.add_argument("--out_overlap_csv", required=True)
    ap.add_argument("--out_readable_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)
    ap.add_argument("--max_families_per_frame", type=int, default=12)
    ap.add_argument("--max_gap_frames", type=int, default=4)
    ap.add_argument("--min_topology_jaccard", type=float, default=0.28)
    ap.add_argument("--min_birth_score", type=float, default=0.25)
    ap.add_argument("--min_entity_frames", type=int, default=4)
    args = ap.parse_args()

    rows = _load_csv(Path(args.micro_family_csv))
    by_frame = _group_families_by_frame(rows)

    active_entities: Dict[int, Dict[str, Any]] = {}
    finished_entities: List[Dict[str, Any]] = []
    entity_frame_rows: List[Dict[str, Any]] = []
    overlap_rows: List[Dict[str, Any]] = []
    readable_rows: List[Dict[str, Any]] = []
    next_entity_id = 1

    for frame in sorted(by_frame):
        sigs = [_family_signature(r) for r in by_frame[frame][: args.max_families_per_frame]]
        updated_entities: Set[int] = set()

        for sig in sigs:
            if _safe_float(sig["family_score"], 0.0) < args.min_birth_score:
                continue

            best_eid = None
            best_cost = 999999.0

            for eid, ent in active_entities.items():
                if eid in updated_entities:
                    continue
                if not _can_continue_entity(ent, sig, frame, args.max_gap_frames, args.min_topology_jaccard):
                    continue
                cost = _entity_match_cost(ent, sig, frame)
                if cost < best_cost:
                    best_cost = cost
                    best_eid = eid

            if best_eid is None:
                eid = next_entity_id
                next_entity_id += 1
                ent = {
                    "entity_id": eid,
                    "birth_frame": frame,
                    "last_frame": frame,
                    "frames": [frame],
                    "scores": [sig["family_score"]],
                    "observed_roots_micro": [sig["observed_root_micro"]],
                    "observed_roots_coarse": [sig["observed_root_coarse"]],
                    "micro_token_sets": [set(sig["micro_tokens"])],
                    "coarse_token_sets": [set(sig["coarse_tokens"])],
                    "last_micro_tokens": set(sig["micro_tokens"]),
                    "last_coarse_tokens": set(sig["coarse_tokens"]),
                    "topology_signatures": [sig["topology_signature"]],
                    "last_topology_signature": sig["topology_signature"],
                }
                active_entities[eid] = ent
                updated_entities.add(eid)
            else:
                eid = best_eid
                ent = active_entities[eid]
                ent["last_frame"] = frame
                ent["frames"].append(frame)
                ent["scores"].append(sig["family_score"])
                ent["observed_roots_micro"].append(sig["observed_root_micro"])
                ent["observed_roots_coarse"].append(sig["observed_root_coarse"])
                ent["micro_token_sets"].append(set(sig["micro_tokens"]))
                ent["coarse_token_sets"].append(set(sig["coarse_tokens"]))
                ent["last_micro_tokens"] = set(sig["micro_tokens"])
                ent["last_coarse_tokens"] = set(sig["coarse_tokens"])
                ent["topology_signatures"].append(sig["topology_signature"])
                ent["last_topology_signature"] = sig["topology_signature"]
                updated_entities.add(eid)

            entity_frame_rows.append({
                "entity_id": eid,
                "frame_index": frame,
                "observed_root_micro_not_identity": sig["observed_root_micro"],
                "observed_root_coarse_not_identity": sig["observed_root_coarse"],
                "observed_root_not_identity": sig["observed_root_micro"],
                "family_score": f"{sig['family_score']:.9f}",
                "evidence_count": sig["evidence_count"],
                "root_micro_count": sig["root_micro_count"],
                "root_micro_diversity": sig["root_micro_diversity"],
                "topology_signature": sig["topology_signature"],
                "token_micro_count": len(sig["micro_tokens"]),
                "token_coarse_count": len(sig["coarse_tokens"]),
                "token_count": len(sig["micro_tokens"]),
                "tokens_micro": " ".join(sorted(sig["micro_tokens"])[:80]),
                "tokens_coarse": " ".join(sorted(sig["coarse_tokens"])[:80]),
                "tokens": " ".join(sorted(sig["micro_tokens"])[:80]),
                "token_structure_json": json.dumps(sig["token_structure"][:120], ensure_ascii=False),
            })

        current_ids = sorted(updated_entities)

        for i, a in enumerate(current_ids):
            for b in current_ids[i + 1:]:
                ent_a = active_entities[a]
                ent_b = active_entities[b]
                sim = _structured_similarity(
                    {"micro_tokens": ent_a["last_micro_tokens"], "coarse_tokens": ent_a["last_coarse_tokens"]},
                    {"micro_tokens": ent_b["last_micro_tokens"], "coarse_tokens": ent_b["last_coarse_tokens"]},
                )

                tag_jaccard = _topology_tag_similarity(
                    ent_a.get("last_topology_signature", ""),
                    ent_b.get("last_topology_signature", ""),
                )

                combined_overlap = _combined_overlap_similarity(sim, tag_jaccard)

                overlap_rows.append({
                    "frame_index": frame,
                    "entity_a": a,
                    "entity_b": b,

                    # Semantic overlap continuity:
                    # token continuity + topology lineage tags.
                    "topology_jaccard": f"{combined_overlap:.9f}",

                    # Explicit components for diagnostics.
                    "token_topology_jaccard": f"{sim['combined']:.9f}",
                    "topology_tag_jaccard": f"{tag_jaccard:.9f}",

                    "micro_topology_jaccard": f"{sim['micro_jaccard']:.9f}",
                    "coarse_topology_jaccard": f"{sim['coarse_jaccard']:.9f}",

                    "entity_a_topology_tags": ent_a.get("last_topology_signature", ""),
                    "entity_b_topology_tags": ent_b.get("last_topology_signature", ""),
                    
                    # =========================================================
                    # Diagnostic micro previews
                    # =========================================================

                    "entity_a_micro_count": len(ent_a["last_micro_tokens"]),
                    "entity_b_micro_count": len(ent_b["last_micro_tokens"]),

                    "entity_a_micro_preview": " ".join(
                        sorted(ent_a["last_micro_tokens"])[:24]
                    ),

                    "entity_b_micro_preview": " ".join(
                        sorted(ent_b["last_micro_tokens"])[:24]
                    ),

                    "entity_a_root_hint_micro": ent_a["observed_roots_micro"][-1],
                    "entity_b_root_hint_micro": ent_b["observed_roots_micro"][-1],
                    "entity_a_root_hint_coarse": ent_a["observed_roots_coarse"][-1],
                    "entity_b_root_hint_coarse": ent_b["observed_roots_coarse"][-1],
                    "entity_a_root_hint": ent_a["observed_roots_micro"][-1],
                    "entity_b_root_hint": ent_b["observed_roots_micro"][-1],
                })

        readable_rows.append({
            "frame_index": frame,
            "active_entity_count": len(current_ids),
            "entities": " | ".join(
                (
                    f"ENTITY_{eid:06d}"
                    f":ROOT={active_entities[eid]['observed_roots_micro'][-1]}"
                    f":TOKENS={len(active_entities[eid]['last_micro_tokens'])}"
                    f":MICRO={','.join(sorted(list(active_entities[eid]['last_micro_tokens']))[:8])}"
                    f":SCORE={active_entities[eid]['scores'][-1]:.3f}"
                )
                for eid in current_ids
            ),
        })

        to_close = []
        for eid, ent in active_entities.items():
            if frame - _safe_int(ent.get("last_frame"), frame) > args.max_gap_frames:
                to_close.append(eid)
        for eid in to_close:
            finished_entities.append(active_entities.pop(eid))

    finished_entities.extend(active_entities.values())

    entity_rows = []
    for ent in finished_entities:
        if len(ent["frames"]) < args.min_entity_frames:
            continue
        entity_rows.append(_summarize_entity(ent))

    entity_rows.sort(key=lambda r: (_safe_int(r["birth_frame"]), _safe_int(r["entity_id"])))

    out_entities = Path(args.out_entities_csv)
    out_frames = Path(args.out_entity_frames_csv)
    out_overlap = Path(args.out_overlap_csv)
    out_readable = Path(args.out_readable_csv)
    out_meta = Path(args.out_meta_json)
    out_txt = Path(args.out_summary_txt)
    out_entities.parent.mkdir(parents=True, exist_ok=True)

    entity_fields = [
        "entity_id", "birth_frame", "end_frame", "duration_frames", "frame_count",
        "mean_family_score", "max_family_score",
        "mean_topology_coherence", "mean_micro_topology_coherence", "mean_coarse_topology_coherence",
        "token_union_micro_count", "token_union_coarse_count", "token_union_count",
        "topology_signature_count",
        "root_hint_micro_not_identity", "root_hint_coarse_not_identity", "root_hint_not_identity",
        "observed_roots_micro", "observed_roots_coarse", "observed_roots",
        "topology_signatures", "token_union_micro", "token_union_coarse", "token_union", "token_structure_json",
    ]
    with out_entities.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=entity_fields)
        w.writeheader()
        w.writerows(entity_rows)

    frame_fields = [
        "entity_id", "frame_index",
        "observed_root_micro_not_identity", "observed_root_coarse_not_identity", "observed_root_not_identity",
        "family_score", "evidence_count", "root_micro_count", "root_micro_diversity",
        "topology_signature", "token_micro_count", "token_coarse_count", "token_count",
        "tokens_micro", "tokens_coarse", "tokens", "token_structure_json",
    ]
    with out_frames.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=frame_fields)
        w.writeheader()
        w.writerows(entity_frame_rows)

    overlap_fields = [
        "frame_index", "entity_a", "entity_b",
        "topology_jaccard",
        "token_topology_jaccard",
        "topology_tag_jaccard",
        "micro_topology_jaccard",
        "coarse_topology_jaccard",
        "entity_a_topology_tags",
        "entity_b_topology_tags",
        "entity_a_micro_count",
        "entity_b_micro_count",
        "entity_a_micro_preview",
        "entity_b_micro_preview",
        "entity_a_root_hint_micro", "entity_b_root_hint_micro",
        "entity_a_root_hint_coarse", "entity_b_root_hint_coarse",
        "entity_a_root_hint", "entity_b_root_hint",
    ]
    with out_overlap.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=overlap_fields)
        w.writeheader()
        w.writerows(overlap_rows)

    with out_readable.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame_index", "active_entity_count", "entities"])
        w.writeheader()
        w.writerows(readable_rows)

    active_distribution: Dict[int, int] = {}
    for r in readable_rows:
        n = _safe_int(r.get("active_entity_count"), 0)
        active_distribution[n] = active_distribution.get(n, 0) + 1

    meta = {
        "stage": "resonance_entity_builder",
        "semantic_version": "structured_micro_coarse_topology_tags_overlap_v3",
        "inputs": {"micro_family_csv": args.micro_family_csv},
        "outputs": {
            "entities_csv": args.out_entities_csv,
            "entity_frames_csv": args.out_entity_frames_csv,
            "overlap_csv": args.out_overlap_csv,
            "readable_csv": args.out_readable_csv,
            "meta_json": args.out_meta_json,
            "summary_txt": args.out_summary_txt,
        },
        "parameters": {
            "max_families_per_frame": args.max_families_per_frame,
            "max_gap_frames": args.max_gap_frames,
            "min_topology_jaccard": args.min_topology_jaccard,
            "min_birth_score": args.min_birth_score,
            "min_entity_frames": args.min_entity_frames,
            "similarity_model": {
                "micro_jaccard_weight": 0.68,
                "coarse_jaccard_weight": 0.32,
                "identity_rule": "micro identity is primary; coarse topology supports continuity but is not identity",
            },
        },
        "result": {
            "input_family_rows": len(rows),
            "entity_frame_rows": len(entity_frame_rows),
            "entities_kept": len(entity_rows),
            "overlap_rows": len(overlap_rows),
            "readable_frames": len(readable_rows),
            "active_entity_distribution": active_distribution,
        },
        "ontology_note": (
            "Entity identity is built from structured resonance topology. "
            "Micro and coarse token sets are preserved separately. Semantic topology tags are used for overlap lineage. "
            "Observed roots remain hints, not identity."
        ),
    }
    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    txt = [
        "RESONANCE ENTITY BUILDER",
        "=" * 72,
        f"micro_family_csv   : {args.micro_family_csv}",
        "",
        f"input_family_rows  : {len(rows)}",
        f"entity_frame_rows  : {len(entity_frame_rows)}",
        f"entities_kept      : {len(entity_rows)}",
        f"overlap_rows       : {len(overlap_rows)}",
        "",
        "Active entity distribution:",
    ]
    for k in sorted(active_distribution):
        txt.append(f"  {k}: {active_distribution[k]}")
    txt.extend([
        "",
        "Principle:",
        "  Entity identity is built from structured resonance topology, not from note_token.",
        "  Micro identity is preserved separately from coarse ownership neighborhood.",
        "  observed_root is kept only as a hint and must not be treated as identity.",
        "",
    ])
    out_txt.write_text("\n".join(txt), encoding="utf-8")

    print("resonance entity builder complete")
    print(json.dumps(meta["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
