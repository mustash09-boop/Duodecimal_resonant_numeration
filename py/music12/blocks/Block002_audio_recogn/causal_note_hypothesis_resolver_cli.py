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
        return int(x)
    except Exception:
        return default


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _tokens(raw: str) -> Set[str]:
    return {x.strip() for x in str(raw or "").split() if x.strip()}


def _normalize_note(s: str) -> str:
    s = str(s or "").strip()

    if not s:
        return ""

    if "'" in s:
        return s.split("'", 1)[0] + "'-"

    if s.endswith("-"):
        return s[:-1] + "'-"

    return s + "'-"


def _degree(note: str) -> str:
    try:
        return _normalize_note(note).split(".", 1)[1].split("'", 1)[0]
    except Exception:
        return ""


def _root_hint(row: Dict[str, Any]) -> str:
    return _normalize_note(row.get("root_hint_not_identity", ""))


def _candidate_notes_from_entity(row: Dict[str, Any]) -> Set[str]:
    candidates = set()

    root = _root_hint(row)

    if root:
        candidates.add(root)

    for tok in _tokens(row.get("token_union", "")):
        n = _normalize_note(tok)
        if n:
            candidates.add(n)

    for tok in _tokens(row.get("observed_roots", "")):
        n = _normalize_note(tok)
        if n:
            candidates.add(n)

    return candidates


def _build_lineage_map(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out = {}

    for r in rows:
        note = _normalize_note(r.get("candidate_note", ""))

        out[note] = {
            "h1": _safe_float(r.get("h1_present_percent", 0.0)),
            "h3": _safe_float(r.get("h3_present_percent", 0.0)),
            "h5": _safe_float(r.get("h5_present_percent", 0.0)),
            "h7": _safe_float(r.get("h7_present_percent", 0.0)),
            "h9": _safe_float(r.get("h9_present_percent", 0.0)),
            "h11": _safe_float(r.get("h11_present_percent", 0.0)),
            "h13": _safe_float(r.get("h13_present_percent", 0.0)),
            "odd_score": _safe_float(r.get("odd_lineage_score", 0.0)),
            "deep_score": _safe_float(r.get("deep_lineage_score", 0.0)),
        }

    return out


def _score_hypothesis(
    *,
    candidate: str,
    entity: Dict[str, Any],
    role: Dict[str, Any],
    lineage_map: Dict[str, Dict[str, float]],
) -> float:

    root = _root_hint(entity)

    token_union = {
        _normalize_note(x)
        for x in _tokens(entity.get("token_union", ""))
    }

    observed_roots = [
        _normalize_note(x)
        for x in str(entity.get("observed_roots", "")).split()
        if x.strip()
    ]

    ownership = _safe_float(role.get("ownership_strength"), 0.0)
    carrying = _safe_float(role.get("carrying_strength"), 0.0)
    feeding = _safe_float(role.get("feeding_strength"), 0.0)
    masking = _safe_float(role.get("masking_strength"), 0.0)

    mean_score = _safe_float(entity.get("mean_family_score"), 0.0)
    coherence = _safe_float(entity.get("mean_topology_coherence"), 0.0)
    duration = _safe_int(entity.get("duration_frames"), 0)

    ownership_role = str(role.get("ownership_role", "")).strip()

    lineage = lineage_map.get(candidate, {})

    h1 = lineage.get("h1", 0.0)
    h3 = lineage.get("h3", 0.0)
    h5 = lineage.get("h5", 0.0)
    h7 = lineage.get("h7", 0.0)
    h9 = lineage.get("h9", 0.0)
    h11 = lineage.get("h11", 0.0)
    h13 = lineage.get("h13", 0.0)

    odd_score = lineage.get("odd_score", 0.0)
    deep_score = lineage.get("deep_score", 0.0)

    score = 0.0

    # ---------------------------------------------------------
    # root / token evidence
    # ---------------------------------------------------------

    if candidate == root:
        score += 0.14

    if _degree(candidate) == _degree(root) and candidate != root:
        score += 0.04

    if candidate in token_union:
        score += 0.16

    root_hits = sum(
        1
        for r in observed_roots
        if r == candidate
    )

    same_degree_hits = sum(
        1
        for r in observed_roots
        if _degree(r) == _degree(candidate)
    )

    score += min(root_hits, 10) * 0.025
    score += min(same_degree_hits, 10) * 0.008

    # ---------------------------------------------------------
    # ownership dynamics
    # ---------------------------------------------------------

    score += ownership * 0.52
    score += carrying * 0.035
    score += feeding * 0.07

    score -= masking * 0.30

    # ---------------------------------------------------------
    # ecology evidence
    # ---------------------------------------------------------

    score += min(mean_score / 2.5, 1.0) * 0.10
    score += min(coherence, 1.0) * 0.08

    # ---------------------------------------------------------
    # harmonic genealogy
    # ---------------------------------------------------------

    score += min(h3 / 100.0, 1.0) * 0.16
    score += min(h5 / 100.0, 1.0) * 0.24
    score += min(h7 / 100.0, 1.0) * 0.18
    score += min(h9 / 100.0, 1.0) * 0.24
    score += min(h11 / 100.0, 1.0) * 0.28
    score += min(h13 / 100.0, 1.0) * 0.24

    score += min(odd_score / 100.0, 1.0) * 0.40
    score += min(deep_score / 100.0, 1.0) * 0.46

    # ---------------------------------------------------------
    # anti-resonance-basin logic
    # ---------------------------------------------------------

    duration_ratio = min(duration / 180.0, 1.0)

    if duration > 72:
        score -= duration_ratio * 0.18

    if duration > 120:
        score -= 0.14

    if ownership_role == "RESONANCE_CARRIER":
        score -= 0.22

        if duration > 48:
            score -= 0.12

        if carrying > ownership:
            score -= 0.10

    if ownership_role == "MASKING_FIELD":
        score -= 0.35

    # ---------------------------------------------------------
    # deep-lineage collapse detection
    # ---------------------------------------------------------

    early_bloom = (h1 + h3) / 2.0
    deep_lineage = (h9 + h11 + h13) / 3.0

    if early_bloom > 45.0 and deep_lineage < 8.0:
        score -= 0.34

    if early_bloom > 55.0 and deep_lineage < 4.0:
        score -= 0.48

    if h11 < 1.0 and h13 < 1.0 and h1 > 40.0:
        score -= 0.22

    # ---------------------------------------------------------
    # excessive harmonic inertia penalty
    # ---------------------------------------------------------

    if root_hits > 12 and duration > 90:
        score -= 0.10

    if same_degree_hits > 16 and duration > 120:
        score -= 0.14

    return max(score, 0.0)


def main() -> None:

    ap = argparse.ArgumentParser(
        description="Build competing causal note hypotheses for ownership entities."
    )

    ap.add_argument("--ecology_entities_csv", required=True)
    ap.add_argument("--ownership_roles_csv", required=True)

    ap.add_argument("--lineage_support_csv", required=True)

    ap.add_argument("--out_hypotheses_csv", required=True)
    ap.add_argument("--out_resolved_events_csv", required=True)
    ap.add_argument("--out_frame_notes_csv", required=True)
    ap.add_argument("--out_readable_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--min_hypothesis_score", type=float, default=0.42)
    ap.add_argument("--min_confidence", type=float, default=0.36)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--fps", type=float, default=60.0)

    args = ap.parse_args()

    entities = _load_csv(Path(args.ecology_entities_csv))
    roles = _load_csv(Path(args.ownership_roles_csv))
    lineage_rows = _load_csv(Path(args.lineage_support_csv))

    lineage_map = _build_lineage_map(lineage_rows)

    role_map = {
        str(r.get("entity_id", "")).strip(): r
        for r in roles
    }

    hypothesis_rows = []
    resolved_rows = []
    frame_rows = []

    status_counts = defaultdict(int)

    for e in entities:

        eid = str(e.get("ecology_entity_id", "")).strip()

        role = role_map.get(eid, {})

        ownership_role = str(role.get("ownership_role", "")).strip()

        candidates = _candidate_notes_from_entity(e)

        scored = []

        for cand in candidates:

            score = _score_hypothesis(
                candidate=cand,
                entity=e,
                role=role,
                lineage_map=lineage_map,
            )

            if score >= args.min_hypothesis_score:
                scored.append((cand, score))

        scored.sort(key=lambda x: (-x[1], x[0]))
        scored = scored[: args.top_k]

        total = sum(s for _, s in scored)

        best_note = scored[0][0] if scored else ""
        best_score = scored[0][1] if scored else 0.0

        confidence = best_score / max(total, 1e-9)

        if not scored:
            status = "NO_HYPOTHESIS"

        elif confidence >= args.min_confidence and ownership_role in {
            "PRIMARY_OWNER",
            "RESONANCE_CARRIER",
        }:
            status = "RESOLVED"

        elif ownership_role in {"MASKING_FIELD"}:
            status = "MASKING_NOT_NOTE"

        else:
            status = "AMBIGUOUS"

        status_counts[status] += 1

        for rank, (note, score) in enumerate(scored, start=1):

            hypothesis_rows.append({
                "entity_id": eid,
                "rank": rank,
                "candidate_note": note,
                "hypothesis_score": f"{score:.9f}",
                "hypothesis_probability": f"{(score / max(total, 1e-9)):.9f}",
                "ownership_role": ownership_role,
                "birth_frame": e.get("birth_frame", ""),
                "end_frame": e.get("end_frame", ""),
                "duration_frames": e.get("duration_frames", ""),
            })

    # ---------------------------------------------------------
    # resolved rows
    # ---------------------------------------------------------

    for e in entities:

        eid = str(e.get("ecology_entity_id", "")).strip()

        role = role_map.get(eid, {})

        ownership_role = str(role.get("ownership_role", "")).strip()

        entity_hypotheses = [
            r
            for r in hypothesis_rows
            if r["entity_id"] == eid
        ]

        entity_hypotheses.sort(
            key=lambda x: -_safe_float(x["hypothesis_score"])
        )

        total = sum(
            _safe_float(x["hypothesis_score"])
            for x in entity_hypotheses
        )

        best_note = (
            entity_hypotheses[0]["candidate_note"]
            if entity_hypotheses
            else ""
        )

        best_score = (
            _safe_float(entity_hypotheses[0]["hypothesis_score"])
            if entity_hypotheses
            else 0.0
        )

        confidence = best_score / max(total, 1e-9)

        if not entity_hypotheses:
            status = "NO_HYPOTHESIS"

        elif (
            confidence >= args.min_confidence
            and ownership_role in {
                "PRIMARY_OWNER",
                "RESONANCE_CARRIER",
            }
        ):
            status = "RESOLVED"

        elif ownership_role in {"MASKING_FIELD"}:
            status = "MASKING_NOT_NOTE"

        else:
            status = "AMBIGUOUS"

        resolved_rows.append({
            "entity_id": eid,
            "resolved_note": best_note,
            "resolution_status": status,
            "resolution_confidence": f"{confidence:.9f}",
            "best_hypothesis_score": f"{best_score:.9f}",
            "ownership_role": ownership_role,
            "birth_frame": e.get("birth_frame", ""),
            "end_frame": e.get("end_frame", ""),
            "duration_frames": e.get("duration_frames", ""),
            "alternatives": " | ".join(
                f"{x['candidate_note']}:{_safe_float(x['hypothesis_score']):.3f}"
                for x in entity_hypotheses[:8]
            ),
            "root_hint_not_identity": e.get(
                "root_hint_not_identity",
                "",
            ),
        })

        if status == "RESOLVED" and best_note:

            start = _safe_int(e.get("birth_frame"), 0)
            end = _safe_int(e.get("end_frame"), 0)

            for frame in range(start, end + 1):

                frame_rows.append({
                    "frame_index": frame,
                    "time_sec": f"{(frame / max(args.fps, 1e-9)):.9f}",
                    "entity_id": eid,
                    "note_token": best_note,
                    "note_confidence": f"{confidence:.9f}",
                    "ownership_role": ownership_role,
                    "alternatives": " | ".join(
                        f"{x['candidate_note']}:{_safe_float(x['hypothesis_score']):.3f}"
                        for x in entity_hypotheses[:3]
                    ),
                })

    # ---------------------------------------------------------
    # readable frames
    # ---------------------------------------------------------

    by_frame: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for r in frame_rows:
        by_frame[_safe_int(r["frame_index"])].append(r)

    readable_rows = []

    for frame in sorted(by_frame):

        items = sorted(
            by_frame[frame],
            key=lambda x: -_safe_float(x.get("note_confidence"), 0.0),
        )

        readable_rows.append({
            "frame_index": frame,
            "time_sec": f"{(frame / max(args.fps, 1e-9)):.9f}",
            "active_note_count": len(items),
            "notes": " | ".join(
                f"{r['note_token']}:{_safe_float(r['note_confidence']):.3f}[E{r['entity_id']}/{r['ownership_role']}]"
                for r in items[:12]
            ),
        })

    # ---------------------------------------------------------
    # outputs
    # ---------------------------------------------------------

    out_hyp = Path(args.out_hypotheses_csv)
    out_res = Path(args.out_resolved_events_csv)
    out_frame = Path(args.out_frame_notes_csv)
    out_readable = Path(args.out_readable_csv)
    out_meta = Path(args.out_meta_json)
    out_txt = Path(args.out_summary_txt)

    out_hyp.parent.mkdir(parents=True, exist_ok=True)

    hyp_fields = [
        "entity_id",
        "rank",
        "candidate_note",
        "hypothesis_score",
        "hypothesis_probability",
        "ownership_role",
        "birth_frame",
        "end_frame",
        "duration_frames",
    ]

    with out_hyp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hyp_fields)
        w.writeheader()
        w.writerows(hypothesis_rows)

    res_fields = [
        "entity_id",
        "resolved_note",
        "resolution_status",
        "resolution_confidence",
        "best_hypothesis_score",
        "ownership_role",
        "birth_frame",
        "end_frame",
        "duration_frames",
        "alternatives",
        "root_hint_not_identity",
    ]

    with out_res.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=res_fields)
        w.writeheader()
        w.writerows(resolved_rows)

    frame_fields = [
        "frame_index",
        "time_sec",
        "entity_id",
        "note_token",
        "note_confidence",
        "ownership_role",
        "alternatives",
    ]

    with out_frame.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=frame_fields)
        w.writeheader()
        w.writerows(frame_rows)

    with out_readable.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "frame_index",
                "time_sec",
                "active_note_count",
                "notes",
            ],
        )
        w.writeheader()
        w.writerows(readable_rows)

    active_distribution = defaultdict(int)

    for r in readable_rows:
        active_distribution[
            _safe_int(r["active_note_count"])
        ] += 1

    meta = {
        "stage": "causal_note_hypothesis_resolver_lineage",
        "inputs": {
            "ecology_entities_csv": args.ecology_entities_csv,
            "ownership_roles_csv": args.ownership_roles_csv,
            "lineage_support_csv": args.lineage_support_csv,
        },
        "outputs": {
            "hypotheses_csv": args.out_hypotheses_csv,
            "resolved_events_csv": args.out_resolved_events_csv,
            "frame_notes_csv": args.out_frame_notes_csv,
            "readable_csv": args.out_readable_csv,
            "meta_json": args.out_meta_json,
            "summary_txt": args.out_summary_txt,
        },
        "result": {
            "input_entities": len(entities),
            "hypotheses": len(hypothesis_rows),
            "resolved_events": len(resolved_rows),
            "frame_note_rows": len(frame_rows),
            "readable_frames": len(readable_rows),
            "status_counts": dict(status_counts),
            "active_distribution": dict(active_distribution),
        },
    }

    out_meta.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    txt = [
        "CAUSAL NOTE HYPOTHESIS RESOLVER WITH LINEAGE",
        "=" * 72,
        f"input_entities  : {len(entities)}",
        f"hypotheses      : {len(hypothesis_rows)}",
        f"resolved_events : {len(resolved_rows)}",
        "",
        "Principle:",
        "  Note identity is validated by harmonic genealogy.",
        "  Deep odd harmonics dominate over early resonance bloom.",
        "",
        "Status counts:",
    ]

    for k in sorted(status_counts):
        txt.append(f"  {k}: {status_counts[k]}")

    out_txt.write_text(
        "\n".join(txt),
        encoding="utf-8",
    )

    print("causal note hypothesis resolver with lineage support complete")

    print(json.dumps(meta["result"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()