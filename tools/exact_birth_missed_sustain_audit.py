# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return default


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _normalize_note(note: str) -> str:
    s = str(note or "").strip()
    if not s:
        return ""
    if "'" in s:
        return s.split("'", 1)[0] + "'-"
    if s.endswith("-"):
        return s[:-1] + "'-"
    return s + "'-"


def _pitch_class(note: str) -> str:
    try:
        return _normalize_note(note).split(".", 1)[1].split("'", 1)[0]
    except Exception:
        return ""


def _octave(note: str) -> str:
    try:
        return _normalize_note(note).split(".", 1)[0]
    except Exception:
        return ""


def _build_by_frame(rows: Iterable[Dict[str, Any]], field: str = "frame_index") -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[_safe_int(row.get(field), 0)].append(row)
    return out


def _dominant_family_row(rows: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    if not rows:
        return None
    return min(rows, key=lambda r: _safe_int(r.get("family_rank"), 999999))


def _note_counts(rows: List[Dict[str, Any]]) -> Counter[str]:
    c: Counter[str] = Counter()
    for row in rows:
        note = _normalize_note(row.get("note_token", ""))
        if note:
            c[note] += 1
    return c


def _classify_event(
    *,
    event: Dict[str, Any],
    detected_by_frame: Dict[int, List[Dict[str, Any]]],
    families_by_frame: Dict[int, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    note = _normalize_note(event.get("expected_note", ""))
    target_pc = _pitch_class(note)
    target_oct = _octave(note)

    birth_frame = _safe_int(event.get("birth_exact_frame"), -1)
    start_frame = _safe_int(event.get("start_frame"), 0)
    end_frame = _safe_int(event.get("end_frame"), 0)
    if birth_frame < 0:
        birth_frame = start_frame

    exact_hit_frames: List[int] = []
    miss_frames: List[int] = []
    dominant_note_counts: Counter[str] = Counter()
    candidate_kind_counts: Counter[str] = Counter()
    causal_role_counts: Counter[str] = Counter()
    collapse_votes: Counter[str] = Counter()
    dominant_family_exact_frames = 0
    dominant_family_pc_frames = 0
    same_pc_detected_frames = 0
    wrong_octave_detected_frames = 0

    for frame in range(birth_frame, end_frame + 1):
        detected_rows = detected_by_frame.get(frame, [])
        notes_here = {_normalize_note(r.get("note_token", "")) for r in detected_rows}
        notes_here.discard("")

        if note in notes_here:
            exact_hit_frames.append(frame)
            continue

        miss_frames.append(frame)
        if notes_here:
            dominant_note, _ = _note_counts(detected_rows).most_common(1)[0]
            dominant_note_counts[dominant_note] += 1

        frame_has_pc = False
        frame_has_wrong_oct = False
        for row in detected_rows:
            n = _normalize_note(row.get("note_token", ""))
            if not n:
                continue
            candidate_kind_counts[str(row.get("candidate_kind", "")).strip() or "untyped"] += 1
            causal_role_counts[str(row.get("causal_role", "")).strip() or "untyped"] += 1
            if _pitch_class(n) == target_pc:
                frame_has_pc = True
                if _octave(n) != target_oct:
                    frame_has_wrong_oct = True

        if frame_has_pc:
            same_pc_detected_frames += 1
        if frame_has_wrong_oct:
            wrong_octave_detected_frames += 1

        dominant_family = _dominant_family_row(families_by_frame.get(frame, []))
        family_root = _normalize_note((dominant_family or {}).get("family_root_note_micro", ""))
        if family_root == note:
            dominant_family_exact_frames += 1
            collapse_votes["SAME_DEGREE_DRIFT"] += 2
        elif family_root and _pitch_class(family_root) == target_pc:
            dominant_family_pc_frames += 1
            collapse_votes["OCTAVE_ROOT_CONFUSION"] += 1

        if frame_has_wrong_oct:
            collapse_votes["OCTAVE_ROOT_CONFUSION"] += 2

        if any("bridge" in role for role in causal_role_counts if causal_role_counts[role] > 0):
            pass

    companion_votes = sum(
        count for kind, count in candidate_kind_counts.items() if kind == "STRUCTURAL_COMPANION"
    )
    bridge_votes = sum(
        count for role, count in causal_role_counts.items() if "bridge" in role.lower()
    )

    if companion_votes:
        collapse_votes["COMPANION_TAKEOVER"] += companion_votes
    if bridge_votes:
        collapse_votes["BRIDGE_TAKEOVER"] += bridge_votes

    tail_status = str(event.get("tail_status", "")).strip()
    tail_exact_frames = _safe_int(event.get("tail_exact_frames"), 0)
    tail_pc_frames = _safe_int(event.get("tail_pc_frames"), 0)
    exact_sustain_hits = _safe_int(event.get("exact_sustain_hits"), 0)
    duration_frames = max(_safe_int(event.get("duration_frames"), 1), 1)

    if tail_status != "NO_TAIL" and exact_sustain_hits <= max(2, duration_frames // 6):
        collapse_votes["EARLY_TAIL_CAPTURE"] += 3 + tail_exact_frames + min(tail_pc_frames, 2)

    if same_pc_detected_frames and not wrong_octave_detected_frames and dominant_family_exact_frames:
        collapse_votes["SAME_DEGREE_DRIFT"] += same_pc_detected_frames

    collapse_type = "UNRESOLVED"
    if collapse_votes:
        collapse_type = sorted(
            collapse_votes.items(),
            key=lambda item: (-item[1], item[0]),
        )[0][0]

    dominant_note_after_birth = ""
    if dominant_note_counts:
        dominant_note_after_birth = dominant_note_counts.most_common(1)[0][0]

    return {
        "event_id": event.get("event_id", ""),
        "event_index": event.get("event_index", ""),
        "expected_note": note,
        "birth_frame": birth_frame,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "duration_frames": duration_frames,
        "exact_sustain_hits": exact_sustain_hits,
        "miss_frames_after_birth": len(miss_frames),
        "last_exact_frame": exact_hit_frames[-1] if exact_hit_frames else -1,
        "dominant_note_after_birth": dominant_note_after_birth,
        "dominant_candidate_kind": candidate_kind_counts.most_common(1)[0][0] if candidate_kind_counts else "",
        "dominant_causal_role": causal_role_counts.most_common(1)[0][0] if causal_role_counts else "",
        "same_pc_detected_frames": same_pc_detected_frames,
        "wrong_octave_detected_frames": wrong_octave_detected_frames,
        "dominant_family_exact_frames": dominant_family_exact_frames,
        "dominant_family_pc_frames": dominant_family_pc_frames,
        "tail_status": tail_status,
        "tail_exact_frames": tail_exact_frames,
        "tail_pc_frames": tail_pc_frames,
        "collapse_type": collapse_type,
        "collapse_votes_json": json.dumps(dict(collapse_votes), ensure_ascii=False, sort_keys=True),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Inspect EXACT_BIRTH + MISSED_SUSTAIN events and classify how the note collapses after birth."
    )
    ap.add_argument("--event-audit-csv", required=True)
    ap.add_argument("--frame-notes-csv", required=True)
    ap.add_argument("--micro-families-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--include-late-exact-birth", action="store_true")
    args = ap.parse_args()

    event_rows = _load_csv(Path(args.event_audit_csv))
    detected_rows = _load_csv(Path(args.frame_notes_csv))
    family_rows = _load_csv(Path(args.micro_families_csv))

    detected_by_frame = _build_by_frame(detected_rows, "frame_index")
    families_by_frame = _build_by_frame(family_rows, "frame_index")

    target_events: List[Dict[str, Any]] = []
    for row in event_rows:
        onset_status = str(row.get("onset_status", "")).strip()
        sustain_status = str(row.get("sustain_status", "")).strip()
        if sustain_status != "MISSED_SUSTAIN":
            continue
        if onset_status == "EXACT_BIRTH" or (
            args.include_late_exact_birth and onset_status == "LATE_EXACT_BIRTH"
        ):
            target_events.append(row)

    breakdown_rows = [
        _classify_event(
            event=row,
            detected_by_frame=detected_by_frame,
            families_by_frame=families_by_frame,
        )
        for row in target_events
    ]

    collapse_counts: Counter[str] = Counter()
    dominant_note_counts: Counter[str] = Counter()
    for row in breakdown_rows:
        collapse_counts[row["collapse_type"]] += 1
        if row["dominant_note_after_birth"]:
            dominant_note_counts[row["dominant_note_after_birth"]] += 1

    out_csv = Path(args.out_csv)
    out_txt = Path(args.out_summary_txt)
    out_json = Path(args.out_meta_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "event_id",
        "event_index",
        "expected_note",
        "birth_frame",
        "start_frame",
        "end_frame",
        "duration_frames",
        "exact_sustain_hits",
        "miss_frames_after_birth",
        "last_exact_frame",
        "dominant_note_after_birth",
        "dominant_candidate_kind",
        "dominant_causal_role",
        "same_pc_detected_frames",
        "wrong_octave_detected_frames",
        "dominant_family_exact_frames",
        "dominant_family_pc_frames",
        "tail_status",
        "tail_exact_frames",
        "tail_pc_frames",
        "collapse_type",
        "collapse_votes_json",
    ]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(breakdown_rows)

    summary = {
        "stage": "exact_birth_missed_sustain_audit",
        "inputs": {
            "event_audit_csv": args.event_audit_csv,
            "frame_notes_csv": args.frame_notes_csv,
            "micro_families_csv": args.micro_families_csv,
        },
        "parameters": {
            "include_late_exact_birth": bool(args.include_late_exact_birth),
        },
        "result": {
            "events": len(breakdown_rows),
            "collapse_counts": dict(collapse_counts),
            "dominant_notes_after_birth": dict(dominant_note_counts.most_common(20)),
        },
    }

    lines = [
        "EXACT_BIRTH + MISSED_SUSTAIN BREAKDOWN",
        "=" * 72,
        f"events                : {len(breakdown_rows)}",
        "",
        "Collapse types:",
    ]
    for key, value in sorted(collapse_counts.items()):
        lines.append(f"  {key}: {value}")
    lines.extend([
        "",
        "Dominant notes after birth:",
    ])
    for key, value in dominant_note_counts.most_common(15):
        lines.append(f"  {key}: {value}")

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
