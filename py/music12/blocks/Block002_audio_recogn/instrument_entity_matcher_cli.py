# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


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
    s = str(raw or "").replace("|", " ").replace(",", " ")
    return {x.strip() for x in s.split() if x.strip()}


def _normalize_note(token: Any) -> str:
    s = str(token or "").strip()
    if not s:
        return ""
    if "'" in s:
        return s.split("'", 1)[0] + "'-"
    if s.endswith("-"):
        return s[:-1] + "'-"
    return s + "'-"


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def _first_existing(row: Dict[str, Any], names: List[str], default: Any = "") -> Any:
    for n in names:
        if n in row and str(row.get(n, "")).strip() != "":
            return row.get(n)
    return default


def _instrument_arg(raw: str) -> Tuple[str, Path]:
    if "=" not in raw:
        raise ValueError("Instrument passport must be NAME=PATH")
    name, path = raw.split("=", 1)
    return name.strip(), Path(path.strip().strip('"'))


def _load_passport_notes(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = _load_csv(path)
    out: Dict[str, Dict[str, Any]] = {}

    for r in rows:
        note = _normalize_note(_first_existing(
            r,
            ["expected", "note", "note_token", "passport_note", "musical_note"],
            "",
        ))
        if not note:
            continue

        token_fields = [
            "box_resonance_tokens",
            "box_tokens",
            "note_box_tokens",
            "resonance_tokens",
            "harmonic_tokens",
            "strong_tokens",
            "persistent_tokens",
            "echo_tokens",
            "secondary_tokens",
            "dense_tokens",
            "token_union",
        ]

        passport_tokens: Set[str] = set()
        for f in token_fields:
            passport_tokens |= _tokens(r.get(f, ""))

        out[note] = {
            "note": note,
            "detected": _normalize_note(_first_existing(r, ["detected", "detected_note"], "")),
            "root_hz": _safe_float(_first_existing(r, ["root Hz", "root_hz"], 0.0)),
            "delta_cents": _safe_float(_first_existing(r, ["Δ cents", "delta_cents", "cents"], 0.0)),
            "theory_match": _safe_float(_first_existing(r, ["theory match", "theory_match"], 0.0)),
            "clean_rows": _safe_int(_first_existing(r, ["clean rows", "clean_rows"], 0)),
            "removed_box": _safe_int(_first_existing(r, ["removed box", "removed_box"], 0)),
            "spiral_points": _safe_int(_first_existing(r, ["spiral points", "spiral_points"], 0)),
            "passport_tokens": passport_tokens,
            "raw": r,
        }

    return out


def _load_available_notes(path: Path | None) -> Dict[str, Set[str]]:
    if not path or not path.exists():
        return {}

    rows = _load_csv(path)
    out: Dict[str, Set[str]] = defaultdict(set)

    for r in rows:
        inst = str(_first_existing(r, ["instrument", "instrument_name", "name"], "")).strip()
        note = _normalize_note(_first_existing(r, ["note", "note_token", "expected"], ""))

        if inst and note:
            out[inst].add(note)

    return out


def _load_audit(path: Path | None) -> Dict[str, float]:
    if not path or not path.exists():
        return {}

    rows = _load_csv(path)
    out: Dict[str, float] = {}

    for r in rows:
        inst = str(_first_existing(r, ["instrument", "instrument_name", "name"], "")).strip()
        if not inst:
            continue

        missing = _safe_float(_first_existing(r, ["missing", "total_missing", "root_missing", "clean_missing"], 0.0))
        total = _safe_float(_first_existing(r, ["total", "total_notes", "note_count"], 88.0), 88.0)

        trust = 1.0 - min(missing / max(total, 1.0), 1.0)
        out[inst] = max(0.0, min(trust, 1.0))

    return out


def _entity_id(row: Dict[str, Any]) -> str:
    return str(_first_existing(
        row,
        ["entity_id", "ecology_entity_id", "trajectory_entity_id", "stable_entity_id"],
        "",
    )).strip()


def _score_instrument(
    *,
    note: str,
    entity: Dict[str, Any],
    passport: Dict[str, Any] | None,
    playable: bool,
    completeness: float,
) -> Dict[str, Any]:
    if passport is None:
        return {
            "score": 0.0,
            "box_similarity": 0.0,
            "reason": "NO_NOTE_PASSPORT",
        }

    entity_tokens = _tokens(entity.get("token_union", "")) | _tokens(entity.get("observed_roots", ""))
    passport_tokens = set(passport.get("passport_tokens", set()))

    box_similarity = _jaccard(entity_tokens, passport_tokens)

    theory_match = _safe_float(passport.get("theory_match", 0.0))
    clean_rows = _safe_int(passport.get("clean_rows", 0))
    spiral_points = _safe_int(passport.get("spiral_points", 0))
    delta_cents = abs(_safe_float(passport.get("delta_cents", 0.0)))

    data_strength = min((clean_rows + spiral_points) / 20000.0, 1.0)
    tuning_score = max(0.0, 1.0 - min(delta_cents / 35.0, 1.0))

    score = 0.0
    score += 0.35 if playable else -0.45
    score += box_similarity * 0.35
    score += theory_match * 0.15
    score += data_strength * 0.10
    score += tuning_score * 0.08
    score *= completeness

    reason = []
    reason.append("PLAYABLE" if playable else "OUT_OF_RANGE")
    if box_similarity > 0:
        reason.append("BOX_MATCH")
    if theory_match > 0:
        reason.append("THEORY_MATCH")
    reason.append(f"TRUST={completeness:.2f}")

    return {
        "score": max(score, 0.0),
        "box_similarity": box_similarity,
        "reason": ",".join(reason),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Resolve instrument identity for already resolved musical note entities."
    )

    ap.add_argument("--musical_events_csv", required=True)
    ap.add_argument("--ecology_entities_csv", required=True)

    ap.add_argument(
        "--instrument_passport",
        action="append",
        required=True,
        help="NAME=path_to_instrument_passport_notes.csv",
    )

    ap.add_argument("--available_notes_csv", default="")
    ap.add_argument("--audit_missing_csv", default="")

    ap.add_argument("--out_note_instrument_events_csv", required=True)
    ap.add_argument("--out_frame_note_instruments_csv", required=True)
    ap.add_argument("--out_conflicts_csv", required=True)
    ap.add_argument("--out_summary_txt", required=True)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--min_instrument_score", type=float, default=0.18)

    args = ap.parse_args()

    musical_events = _load_csv(Path(args.musical_events_csv))
    ecology_entities = _load_csv(Path(args.ecology_entities_csv))

    entity_map = {_entity_id(e): e for e in ecology_entities}

    passports: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for raw in args.instrument_passport:
        name, path = _instrument_arg(raw)
        passports[name] = _load_passport_notes(path)

    available = _load_available_notes(Path(args.available_notes_csv) if args.available_notes_csv else None)
    audit = _load_audit(Path(args.audit_missing_csv) if args.audit_missing_csv else None)

    event_rows = []
    frame_rows = []
    conflict_rows = []

    instrument_counts = defaultdict(int)
    status_counts = defaultdict(int)

    for ev in musical_events:
        eid = str(ev.get("entity_id", "")).strip()
        note = _normalize_note(_first_existing(ev, ["musical_note", "scene_note", "note_token"], ""))

        if not eid or not note:
            continue

        entity = entity_map.get(eid, {})
        candidates = []

        for inst, pass_notes in passports.items():
            playable_set = available.get(inst, set())
            playable = True if not playable_set else note in playable_set
            completeness = audit.get(inst, 1.0)

            passport = pass_notes.get(note)

            scored = _score_instrument(
                note=note,
                entity=entity,
                passport=passport,
                playable=playable,
                completeness=completeness,
            )

            candidates.append({
                "instrument": inst,
                "score": scored["score"],
                "box_similarity": scored["box_similarity"],
                "reason": scored["reason"],
            })

        candidates.sort(key=lambda x: (-x["score"], x["instrument"]))

        best = candidates[0] if candidates else {
            "instrument": "",
            "score": 0.0,
            "box_similarity": 0.0,
            "reason": "NO_INSTRUMENTS",
        }

        second = candidates[1] if len(candidates) > 1 else None

        margin = best["score"] - (second["score"] if second else 0.0)

        if best["score"] < args.min_instrument_score:
            status = "INSTRUMENT_UNRESOLVED"
        elif second and margin < 0.08:
            status = "INSTRUMENT_AMBIGUOUS"
        else:
            status = "INSTRUMENT_RESOLVED"

        status_counts[status] += 1
        if status == "INSTRUMENT_RESOLVED":
            instrument_counts[best["instrument"]] += 1

        start = _safe_int(ev.get("birth_frame"), 0)
        end = _safe_int(ev.get("end_frame"), start)

        event_rows.append({
            "entity_id": eid,
            "note_token": note,
            "instrument": best["instrument"],
            "instrument_score": f"{best['score']:.9f}",
            "box_similarity": f"{best['box_similarity']:.9f}",
            "instrument_status": status,
            "birth_frame": start,
            "end_frame": end,
            "duration_frames": end - start + 1,
            "musical_confidence": ev.get("musical_confidence", ""),
            "musical_status": ev.get("musical_status", ""),
            "match_reason": best["reason"],
            "alternatives": " | ".join(
                f"{c['instrument']}:{c['score']:.3f}" for c in candidates[:6]
            ),
        })

        if status == "INSTRUMENT_AMBIGUOUS":
            conflict_rows.append({
                "entity_id": eid,
                "note_token": note,
                "birth_frame": start,
                "end_frame": end,
                "best": f"{best['instrument']}:{best['score']:.3f}",
                "second": f"{second['instrument']}:{second['score']:.3f}" if second else "",
                "margin": f"{margin:.9f}",
                "alternatives": " | ".join(
                    f"{c['instrument']}:{c['score']:.3f}/{c['reason']}" for c in candidates[:8]
                ),
            })

        if status != "INSTRUMENT_UNRESOLVED":
            for frame in range(start, end + 1):
                frame_rows.append({
                    "frame_index": frame,
                    "time_sec": f"{frame / max(args.fps, 1e-9):.9f}",
                    "entity_id": eid,
                    "note_token": note,
                    "instrument": best["instrument"],
                    "instrument_score": f"{best['score']:.9f}",
                    "instrument_status": status,
                })

    _write_csv(
        Path(args.out_note_instrument_events_csv),
        event_rows,
        [
            "entity_id",
            "note_token",
            "instrument",
            "instrument_score",
            "box_similarity",
            "instrument_status",
            "birth_frame",
            "end_frame",
            "duration_frames",
            "musical_confidence",
            "musical_status",
            "match_reason",
            "alternatives",
        ],
    )

    _write_csv(
        Path(args.out_frame_note_instruments_csv),
        frame_rows,
        [
            "frame_index",
            "time_sec",
            "entity_id",
            "note_token",
            "instrument",
            "instrument_score",
            "instrument_status",
        ],
    )

    _write_csv(
        Path(args.out_conflicts_csv),
        conflict_rows,
        [
            "entity_id",
            "note_token",
            "birth_frame",
            "end_frame",
            "best",
            "second",
            "margin",
            "alternatives",
        ],
    )

    summary = {
        "input_musical_events": len(musical_events),
        "output_events": len(event_rows),
        "output_frame_rows": len(frame_rows),
        "conflicts": len(conflict_rows),
        "status_counts": dict(status_counts),
        "instrument_counts": dict(instrument_counts),
        "instruments_loaded": sorted(passports.keys()),
    }

    Path(args.out_summary_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_summary_txt).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()