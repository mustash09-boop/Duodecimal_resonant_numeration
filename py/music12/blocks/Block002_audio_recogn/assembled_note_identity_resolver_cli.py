# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
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


def _normalize_note(token: Any) -> str:
    s = str(token or "").strip()
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


def _note_candidates_from_tokens(tokens: Set[str]) -> Counter:
    c = Counter()
    for t in tokens:
        n = _normalize_note(t)
        if n:
            c[n] += 1
    return c


def _score_candidate(note: str, row: Dict[str, Any]) -> Dict[str, Any]:
    core = _tokens(row.get("core_chain_tokens", ""))
    box = _tokens(row.get("box_resonance_tokens", ""))
    echo = _tokens(row.get("secondary_echo_tokens", ""))
    carrier = _tokens(row.get("carrier_tokens", ""))
    masking = _tokens(row.get("masking_tokens", ""))

    assembly_conf = _safe_float(row.get("assembly_confidence"), 0.0)
    linked_secondary_count = _safe_int(row.get("linked_secondary_count"), 0)

    core_notes = _note_candidates_from_tokens(core)
    box_notes = _note_candidates_from_tokens(box)
    echo_notes = _note_candidates_from_tokens(echo)
    carrier_notes = _note_candidates_from_tokens(carrier)
    masking_notes = _note_candidates_from_tokens(masking)

    candidate = _normalize_note(note)
    cand_degree = _degree(candidate)

    core_hits = core_notes[candidate]
    box_hits = box_notes[candidate]
    echo_hits = echo_notes[candidate]
    carrier_hits = carrier_notes[candidate]
    masking_hits = masking_notes[candidate]

    same_degree_core = sum(v for n, v in core_notes.items() if _degree(n) == cand_degree)
    same_degree_box = sum(v for n, v in box_notes.items() if _degree(n) == cand_degree)
    same_degree_echo = sum(v for n, v in echo_notes.items() if _degree(n) == cand_degree)

    hinted = 1.0 if candidate == _normalize_note(row.get("candidate_note_not_final", "")) else 0.0

    score = 0.0
    score += hinted * 0.22
    score += min(core_hits, 12) * 0.055
    score += min(same_degree_core, 18) * 0.018
    score += min(box_hits, 12) * 0.020
    score += min(same_degree_box, 18) * 0.007
    score += min(carrier_hits, 12) * 0.018
    score -= min(echo_hits, 12) * 0.022
    score -= min(masking_hits, 12) * 0.030
    score -= min(same_degree_echo, 18) * 0.006
    score += assembly_conf * 0.28
    score += min(linked_secondary_count / 8.0, 1.0) * 0.07

    return {
        "note": candidate,
        "score": max(score, 0.0),
        "core_hits": core_hits,
        "box_hits": box_hits,
        "echo_hits": echo_hits,
        "carrier_hits": carrier_hits,
        "masking_hits": masking_hits,
        "same_degree_core": same_degree_core,
        "same_degree_box": same_degree_box,
        "same_degree_echo": same_degree_echo,
        "hinted": hinted,
    }


def _candidate_pool(row: Dict[str, Any]) -> Set[str]:
    pool = set()

    hinted = _normalize_note(row.get("candidate_note_not_final", ""))
    if hinted:
        pool.add(hinted)

    for field in [
        "core_chain_tokens",
        "box_resonance_tokens",
        "secondary_echo_tokens",
        "carrier_tokens",
        "masking_tokens",
    ]:
        for t in _tokens(row.get(field, "")):
            n = _normalize_note(t)
            if n:
                pool.add(n)

    return pool


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Resolve final note identity from assembled polyphonic resonance structures."
    )

    ap.add_argument("--assembled_notes_csv", required=True)
    ap.add_argument("--assembled_frame_notes_csv", required=True)

    ap.add_argument("--out_final_note_events_csv", required=True)
    ap.add_argument("--out_final_frame_notes_csv", required=True)
    ap.add_argument("--out_note_hypotheses_csv", required=True)
    ap.add_argument("--out_note_conflicts_csv", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--min_final_confidence", type=float, default=0.34)
    ap.add_argument("--min_margin", type=float, default=0.08)
    ap.add_argument("--top_k", type=int, default=6)
    ap.add_argument("--fps", type=float, default=60.0)

    args = ap.parse_args()

    rows = _load_csv(Path(args.assembled_notes_csv))

    event_rows = []
    frame_rows = []
    hyp_rows = []
    conflict_rows = []

    status_counts = defaultdict(int)

    for r in rows:
        assembled_id = str(r.get("assembled_id", "")).strip()
        primary_id = str(r.get("primary_entity_id", "")).strip()

        pool = _candidate_pool(r)
        scored = []

        for cand in pool:
            s = _score_candidate(cand, r)
            scored.append(s)

        scored.sort(key=lambda x: (-x["score"], x["note"]))
        scored = scored[: args.top_k]

        total = sum(x["score"] for x in scored)
        best = scored[0] if scored else None
        second = scored[1] if len(scored) > 1 else None

        if best is None or total <= 0:
            status = "NO_NOTE_IDENTITY"
            final_note = ""
            confidence = 0.0
            margin = 0.0
        else:
            final_note = best["note"]
            confidence = best["score"] / max(total, 1e-9)
            margin = best["score"] - (second["score"] if second else 0.0)

            if confidence >= args.min_final_confidence and margin >= args.min_margin:
                status = "FINAL_NOTE_RESOLVED"
            elif confidence >= args.min_final_confidence:
                status = "FINAL_NOTE_AMBIGUOUS"
            else:
                status = "FINAL_NOTE_WEAK"

        status_counts[status] += 1

        birth = _safe_int(r.get("birth_frame"), 0)
        end = _safe_int(r.get("end_frame"), birth)

        for rank, s in enumerate(scored, start=1):
            hyp_rows.append({
                "assembled_id": assembled_id,
                "primary_entity_id": primary_id,
                "rank": rank,
                "candidate_note": s["note"],
                "hypothesis_score": f"{s['score']:.9f}",
                "hypothesis_probability": f"{(s['score'] / max(total, 1e-9)):.9f}",
                "core_hits": s["core_hits"],
                "box_hits": s["box_hits"],
                "echo_hits": s["echo_hits"],
                "carrier_hits": s["carrier_hits"],
                "masking_hits": s["masking_hits"],
                "same_degree_core": s["same_degree_core"],
                "same_degree_box": s["same_degree_box"],
                "same_degree_echo": s["same_degree_echo"],
                "hinted": f"{s['hinted']:.1f}",
            })

        event_rows.append({
            "assembled_id": assembled_id,
            "primary_entity_id": primary_id,
            "final_note": final_note,
            "final_note_status": status,
            "final_note_confidence": f"{confidence:.9f}",
            "note_margin": f"{margin:.9f}",
            "birth_frame": birth,
            "end_frame": end,
            "duration_frames": end - birth + 1,
            "assembly_confidence": r.get("assembly_confidence", ""),
            "linked_secondary_count": r.get("linked_secondary_count", ""),
            "alternatives": " | ".join(f"{x['note']}:{x['score']:.3f}" for x in scored[:5]),
            "core_chain_tokens": r.get("core_chain_tokens", ""),
            "box_resonance_tokens": r.get("box_resonance_tokens", ""),
            "secondary_echo_tokens": r.get("secondary_echo_tokens", ""),
            "carrier_tokens": r.get("carrier_tokens", ""),
            "masking_tokens": r.get("masking_tokens", ""),
        })

        if status in {"FINAL_NOTE_AMBIGUOUS", "FINAL_NOTE_WEAK"}:
            conflict_rows.append({
                "assembled_id": assembled_id,
                "primary_entity_id": primary_id,
                "final_note": final_note,
                "status": status,
                "confidence": f"{confidence:.9f}",
                "margin": f"{margin:.9f}",
                "alternatives": " | ".join(f"{x['note']}:{x['score']:.3f}" for x in scored[:8]),
            })

        if status == "FINAL_NOTE_RESOLVED":
            for frame in range(birth, end + 1):
                frame_rows.append({
                    "frame_index": frame,
                    "time_sec": f"{frame / max(args.fps, 1e-9):.9f}",
                    "assembled_id": assembled_id,
                    "primary_entity_id": primary_id,
                    "note_token": final_note,
                    "note_confidence": f"{confidence:.9f}",
                    "linked_secondary_count": r.get("linked_secondary_count", ""),
                })

    _write_csv(
        Path(args.out_final_note_events_csv),
        event_rows,
        [
            "assembled_id",
            "primary_entity_id",
            "final_note",
            "final_note_status",
            "final_note_confidence",
            "note_margin",
            "birth_frame",
            "end_frame",
            "duration_frames",
            "assembly_confidence",
            "linked_secondary_count",
            "alternatives",
            "core_chain_tokens",
            "box_resonance_tokens",
            "secondary_echo_tokens",
            "carrier_tokens",
            "masking_tokens",
        ],
    )

    _write_csv(
        Path(args.out_final_frame_notes_csv),
        frame_rows,
        [
            "frame_index",
            "time_sec",
            "assembled_id",
            "primary_entity_id",
            "note_token",
            "note_confidence",
            "linked_secondary_count",
        ],
    )

    _write_csv(
        Path(args.out_note_hypotheses_csv),
        hyp_rows,
        [
            "assembled_id",
            "primary_entity_id",
            "rank",
            "candidate_note",
            "hypothesis_score",
            "hypothesis_probability",
            "core_hits",
            "box_hits",
            "echo_hits",
            "carrier_hits",
            "masking_hits",
            "same_degree_core",
            "same_degree_box",
            "same_degree_echo",
            "hinted",
        ],
    )

    _write_csv(
        Path(args.out_note_conflicts_csv),
        conflict_rows,
        [
            "assembled_id",
            "primary_entity_id",
            "final_note",
            "status",
            "confidence",
            "margin",
            "alternatives",
        ],
    )

    summary = {
        "input_assembled": len(rows),
        "final_note_events": len(event_rows),
        "final_frame_rows": len(frame_rows),
        "hypotheses": len(hyp_rows),
        "conflicts": len(conflict_rows),
        "status_counts": dict(status_counts),
    }

    Path(args.out_summary_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_summary_txt).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()