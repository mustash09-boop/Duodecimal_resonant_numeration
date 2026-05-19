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
    return {
        x.strip()
        for x in str(raw or "").replace("|", " ").replace(",", " ").split()
        if x.strip()
    }


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


def _note_counter(tokens: Set[str]) -> Counter:
    c = Counter()
    for t in tokens:
        n = _normalize_note(t)
        if n:
            c[n] += 1
    return c


def _degree_counter_from_notes(notes: Counter) -> Counter:
    c = Counter()
    for n, v in notes.items():
        d = _degree(n)
        if d:
            c[d] += v
    return c


def _candidate_pool(row: Dict[str, Any]) -> Set[str]:
    pool = set()

    for field in [
        "resolved_note",
        "excitation_core_tokens",
        "instrument_body_tokens",
        "secondary_field_tokens",
    ]:
        raw = row.get(field, "")
        if field == "resolved_note":
            n = _normalize_note(raw)
            if n:
                pool.add(n)
        else:
            for t in _tokens(raw):
                n = _normalize_note(t)
                if n:
                    pool.add(n)

    return pool


def _score_attractor(candidate: str, row: Dict[str, Any]) -> Dict[str, Any]:
    cand = _normalize_note(candidate)
    cand_degree = _degree(cand)

    core = _tokens(row.get("excitation_core_tokens", ""))
    body = _tokens(row.get("instrument_body_tokens", ""))
    secondary = _tokens(row.get("secondary_field_tokens", ""))

    core_notes = _note_counter(core)
    body_notes = _note_counter(body)
    secondary_notes = _note_counter(secondary)

    core_degree = _degree_counter_from_notes(core_notes)
    body_degree = _degree_counter_from_notes(body_notes)
    secondary_degree = _degree_counter_from_notes(secondary_notes)

    identity_strength = _safe_float(row.get("identity_strength"), 0.0)
    anatomy_confidence = _safe_float(row.get("anatomy_confidence"), 0.0)

    core_strength = _safe_float(row.get("core_strength"), 0.0)
    body_strength = _safe_float(row.get("body_strength"), 0.0)
    secondary_strength = _safe_float(row.get("secondary_strength"), 0.0)

    field_confidence = _safe_float(row.get("field_confidence"), 0.0)
    persistence_ratio = _safe_float(row.get("secondary_persistence_ratio"), 0.0)
    linked_secondary_count = _safe_int(row.get("linked_secondary_count"), 0)
    delayed_secondary_count = _safe_int(row.get("delayed_secondary_count"), 0)

    behavior = str(row.get("field_behavior", "")).strip()

    hinted = 1.0 if cand == _normalize_note(row.get("resolved_note", "")) else 0.0

    score = 0.0

    score += hinted * 0.18

    score += min(core_notes[cand], 12) * 0.060
    score += min(core_degree[cand_degree], 24) * 0.018

    score += min(body_notes[cand], 12) * 0.020
    score += min(body_degree[cand_degree], 24) * 0.009

    score += min(secondary_notes[cand], 12) * 0.010
    score += min(secondary_degree[cand_degree], 24) * 0.004

    score += identity_strength * 0.18
    score += anatomy_confidence * 0.14
    score += field_confidence * 0.16

    score += core_strength * 0.12
    score += body_strength * 0.06
    score += secondary_strength * 0.03

    score += min(persistence_ratio / 3.0, 1.0) * 0.07
    score += min(linked_secondary_count / 10.0, 1.0) * 0.04
    score += min(delayed_secondary_count / 6.0, 1.0) * 0.03

    if behavior == "DELAYED_RETURNING_BODY_FIELD":
        score += 0.045
    elif behavior == "LONG_BODY_PERSISTENCE":
        score += 0.060
    elif behavior == "MASKING_DOMINANT_FIELD":
        score -= 0.080
    elif behavior == "DRY_EXCITATION_DOMINANT":
        score += 0.020

    return {
        "candidate_note": cand,
        "score": max(score, 0.0),
        "core_note_hits": core_notes[cand],
        "body_note_hits": body_notes[cand],
        "secondary_note_hits": secondary_notes[cand],
        "core_degree_hits": core_degree[cand_degree],
        "body_degree_hits": body_degree[cand_degree],
        "secondary_degree_hits": secondary_degree[cand_degree],
        "hinted": hinted,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Resolve resonance attractors inside persistent body fields."
    )

    ap.add_argument("--body_anatomy_csv", required=True)
    ap.add_argument("--field_persistence_csv", required=True)

    ap.add_argument("--out_attractor_events_csv", required=True)
    ap.add_argument("--out_attractor_hypotheses_csv", required=True)
    ap.add_argument("--out_attractor_frame_csv", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--min_attractor_confidence", type=float, default=0.34)
    ap.add_argument("--min_margin", type=float, default=0.065)
    ap.add_argument("--top_k", type=int, default=6)
    ap.add_argument("--fps", type=float, default=60.0)

    args = ap.parse_args()

    body_rows = _load_csv(Path(args.body_anatomy_csv))
    field_rows = _load_csv(Path(args.field_persistence_csv))

    field_map = {
        str(r.get("identity_id", "")).strip(): r
        for r in field_rows
    }

    events = []
    hypotheses = []
    frames = []

    status_counts = defaultdict(int)
    behavior_counts = defaultdict(int)

    for body in body_rows:
        iid = str(body.get("identity_id", "")).strip()
        field = field_map.get(iid, {})

        row = dict(body)
        row.update(field)

        pool = _candidate_pool(row)
        scored = []

        for cand in pool:
            scored.append(_score_attractor(cand, row))

        scored.sort(key=lambda x: (-x["score"], x["candidate_note"]))
        scored = scored[: args.top_k]

        total = sum(x["score"] for x in scored)
        best = scored[0] if scored else None
        second = scored[1] if len(scored) > 1 else None

        if best is None or total <= 0:
            attractor_note = ""
            confidence = 0.0
            margin = 0.0
            status = "NO_ATTRACTOR"
        else:
            attractor_note = best["candidate_note"]
            confidence = best["score"] / max(total, 1e-9)
            margin = best["score"] - (second["score"] if second else 0.0)

            if confidence >= args.min_attractor_confidence and margin >= args.min_margin:
                status = "ATTRACTOR_RESOLVED"
            elif confidence >= args.min_attractor_confidence:
                status = "ATTRACTOR_AMBIGUOUS"
            else:
                status = "ATTRACTOR_WEAK"

        status_counts[status] += 1

        behavior = str(row.get("field_behavior", "")).strip()
        behavior_counts[behavior] += 1

        birth = _safe_int(row.get("birth_frame"), 0)
        end = _safe_int(row.get("end_frame"), birth)

        for rank, h in enumerate(scored, start=1):
            hypotheses.append({
                "identity_id": iid,
                "rank": rank,
                "candidate_note": h["candidate_note"],
                "attractor_score": f"{h['score']:.9f}",
                "attractor_probability": f"{(h['score'] / max(total, 1e-9)):.9f}",
                "core_note_hits": h["core_note_hits"],
                "body_note_hits": h["body_note_hits"],
                "secondary_note_hits": h["secondary_note_hits"],
                "core_degree_hits": h["core_degree_hits"],
                "body_degree_hits": h["body_degree_hits"],
                "secondary_degree_hits": h["secondary_degree_hits"],
                "hinted": f"{h['hinted']:.1f}",
            })

        events.append({
            "identity_id": iid,
            "attractor_note": attractor_note,
            "attractor_status": status,
            "attractor_confidence": f"{confidence:.9f}",
            "attractor_margin": f"{margin:.9f}",
            "birth_frame": birth,
            "end_frame": end,
            "duration_frames": end - birth + 1,
            "identity_status": row.get("identity_status", ""),
            "identity_strength": row.get("identity_strength", ""),
            "anatomy_status": row.get("anatomy_status", ""),
            "anatomy_confidence": row.get("anatomy_confidence", ""),
            "field_behavior": behavior,
            "field_confidence": row.get("field_confidence", ""),
            "secondary_persistence_ratio": row.get("secondary_persistence_ratio", ""),
            "linked_secondary_count": row.get("linked_secondary_count", ""),
            "delayed_secondary_count": row.get("delayed_secondary_count", ""),
            "alternatives": " | ".join(
                f"{x['candidate_note']}:{x['score']:.3f}" for x in scored
            ),
            "excitation_core_tokens": row.get("excitation_core_tokens", ""),
            "instrument_body_tokens": row.get("instrument_body_tokens", ""),
            "secondary_field_tokens": row.get("secondary_field_tokens", ""),
        })

        if status == "ATTRACTOR_RESOLVED":
            for frame in range(birth, end + 1):
                frames.append({
                    "frame_index": frame,
                    "time_sec": f"{frame / max(args.fps, 1e-9):.9f}",
                    "identity_id": iid,
                    "note_token": attractor_note,
                    "attractor_confidence": f"{confidence:.9f}",
                    "field_behavior": behavior,
                })

    _write_csv(
        Path(args.out_attractor_events_csv),
        events,
        [
            "identity_id",
            "attractor_note",
            "attractor_status",
            "attractor_confidence",
            "attractor_margin",
            "birth_frame",
            "end_frame",
            "duration_frames",
            "identity_status",
            "identity_strength",
            "anatomy_status",
            "anatomy_confidence",
            "field_behavior",
            "field_confidence",
            "secondary_persistence_ratio",
            "linked_secondary_count",
            "delayed_secondary_count",
            "alternatives",
            "excitation_core_tokens",
            "instrument_body_tokens",
            "secondary_field_tokens",
        ],
    )

    _write_csv(
        Path(args.out_attractor_hypotheses_csv),
        hypotheses,
        [
            "identity_id",
            "rank",
            "candidate_note",
            "attractor_score",
            "attractor_probability",
            "core_note_hits",
            "body_note_hits",
            "secondary_note_hits",
            "core_degree_hits",
            "body_degree_hits",
            "secondary_degree_hits",
            "hinted",
        ],
    )

    _write_csv(
        Path(args.out_attractor_frame_csv),
        frames,
        [
            "frame_index",
            "time_sec",
            "identity_id",
            "note_token",
            "attractor_confidence",
            "field_behavior",
        ],
    )

    summary = {
        "input_identities": len(body_rows),
        "attractor_events": len(events),
        "hypotheses": len(hypotheses),
        "frame_rows": len(frames),
        "status_counts": dict(status_counts),
        "field_behavior_counts": dict(behavior_counts),
    }

    Path(args.out_summary_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_summary_txt).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()