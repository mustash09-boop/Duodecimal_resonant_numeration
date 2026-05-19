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
        return int(x)
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


def _build_detected_index(rows: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[_safe_int(row.get("frame_index"), 0)].append(row)
    return out


def _frame_notes(rows: List[Dict[str, Any]]) -> Set[str]:
    return {
        _normalize_note(row.get("note_token", ""))
        for row in rows
        if _normalize_note(row.get("note_token", ""))
    }


def _first_match_frame(
    detected_by_frame: Dict[int, List[Dict[str, Any]]],
    frame_from: int,
    frame_to: int,
    target_note: str,
    pitch_class_only: bool = False,
) -> int:
    target_pc = _pitch_class(target_note)
    for frame in range(frame_from, frame_to + 1):
        notes = _frame_notes(detected_by_frame.get(frame, []))
        if not notes:
            continue
        if pitch_class_only:
            if target_pc and target_pc in {_pitch_class(n) for n in notes}:
                return frame
        else:
            if target_note in notes:
                return frame
    return -1


def _classify_onset(
    *,
    start_frame: int,
    birth_exact_frame: int,
    birth_pc_frame: int,
    onset_slack_frames: int,
) -> str:
    if birth_exact_frame >= 0:
        if birth_exact_frame <= start_frame + onset_slack_frames:
            return "EXACT_BIRTH"
        return "LATE_EXACT_BIRTH"
    if birth_pc_frame >= 0:
        if birth_pc_frame <= start_frame + onset_slack_frames:
            return "PITCHCLASS_BIRTH"
        return "LATE_PITCHCLASS_BIRTH"
    return "MISSED_BIRTH"


def _classify_sustain(exact_ratio: float, pc_ratio: float) -> str:
    if exact_ratio >= 0.60:
        return "STRONG_SUSTAIN"
    if exact_ratio >= 0.25:
        return "PARTIAL_SUSTAIN"
    if pc_ratio >= 0.25:
        return "PITCHCLASS_SUSTAIN_ONLY"
    return "MISSED_SUSTAIN"


def _classify_tail(tail_exact_frames: int, tail_pc_frames: int, false_carry_threshold_frames: int) -> str:
    if tail_exact_frames >= false_carry_threshold_frames:
        return "LONG_FALSE_CARRY"
    if tail_exact_frames > 0:
        return "SHORT_CARRY"
    if tail_pc_frames > 0:
        return "PITCHCLASS_TAIL"
    return "NO_TAIL"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit note detection at the MIDI event level using birth / sustain / tail windows."
    )
    ap.add_argument("--detected-frame-notes-csv", required=True)
    ap.add_argument("--reference-events-csv", required=True)
    ap.add_argument("--out-event-audit-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--detected-duration-sec", type=float, required=True)
    ap.add_argument("--reference-duration-sec", type=float, required=True)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--onset-slack-frames", type=int, default=2)
    ap.add_argument("--tail-window-frames", type=int, default=8)
    ap.add_argument("--false-carry-threshold-frames", type=int, default=4)
    args = ap.parse_args()

    detected_rows = _load_csv(Path(args.detected_frame_notes_csv))
    reference_rows = _load_csv(Path(args.reference_events_csv))

    detected_by_frame = _build_detected_index(detected_rows)
    tempo_ratio = args.detected_duration_sec / max(args.reference_duration_sec, 1e-9)

    event_rows: List[Dict[str, Any]] = []
    onset_counts: Counter[str] = Counter()
    sustain_counts: Counter[str] = Counter()
    tail_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()

    for ref in reference_rows:
        note = _normalize_note(
            ref.get("expected_note_token")
            or ref.get("note_token")
            or ref.get("token")
        )
        if not note:
            continue

        ref_start = _safe_float(ref.get("time_start_sec", ref.get("start_sec", 0.0)))
        ref_end = _safe_float(ref.get("time_end_sec", ref.get("end_sec", 0.0)))
        real_start = ref_start * tempo_ratio
        real_end = ref_end * tempo_ratio

        start_frame = int(round(real_start * args.fps))
        end_frame = int(round(real_end * args.fps))
        if end_frame < start_frame:
            end_frame = start_frame

        onset_from = max(0, start_frame - 1)
        onset_to = start_frame + args.onset_slack_frames

        birth_exact_frame = _first_match_frame(
            detected_by_frame, onset_from, end_frame, note, pitch_class_only=False
        )
        birth_pc_frame = _first_match_frame(
            detected_by_frame, onset_from, end_frame, note, pitch_class_only=True
        )

        onset_status = _classify_onset(
            start_frame=start_frame,
            birth_exact_frame=birth_exact_frame,
            birth_pc_frame=birth_pc_frame,
            onset_slack_frames=args.onset_slack_frames,
        )
        onset_counts[onset_status] += 1

        exact_sustain_hits = 0
        pc_sustain_hits = 0
        target_pc = _pitch_class(note)

        for frame in range(start_frame, end_frame + 1):
            notes = _frame_notes(detected_by_frame.get(frame, []))
            if note in notes:
                exact_sustain_hits += 1
            if target_pc and target_pc in {_pitch_class(n) for n in notes}:
                pc_sustain_hits += 1

        duration_frames = max(end_frame - start_frame + 1, 1)
        exact_sustain_ratio = exact_sustain_hits / duration_frames
        pc_sustain_ratio = pc_sustain_hits / duration_frames
        sustain_status = _classify_sustain(exact_sustain_ratio, pc_sustain_ratio)
        sustain_counts[sustain_status] += 1

        tail_exact_frames = 0
        tail_pc_frames = 0
        tail_from = end_frame + 1
        tail_to = end_frame + args.tail_window_frames
        for frame in range(tail_from, tail_to + 1):
            notes = _frame_notes(detected_by_frame.get(frame, []))
            if note in notes:
                tail_exact_frames += 1
            if target_pc and target_pc in {_pitch_class(n) for n in notes}:
                tail_pc_frames += 1

        tail_status = _classify_tail(
            tail_exact_frames, tail_pc_frames, args.false_carry_threshold_frames
        )
        tail_counts[tail_status] += 1

        if onset_status == "MISSED_BIRTH":
            error_counts["missed_birth"] += 1
        elif onset_status == "LATE_EXACT_BIRTH":
            error_counts["late_birth"] += 1
        elif onset_status == "PITCHCLASS_BIRTH":
            error_counts["pitchclass_only_birth"] += 1
        elif onset_status == "LATE_PITCHCLASS_BIRTH":
            error_counts["late_pitchclass_birth"] += 1

        if sustain_status == "MISSED_SUSTAIN":
            error_counts["missed_sustain"] += 1
        elif sustain_status == "PITCHCLASS_SUSTAIN_ONLY":
            error_counts["octave_or_root_confusion"] += 1

        if tail_status == "LONG_FALSE_CARRY":
            error_counts["false_carry"] += 1
        elif tail_status == "PITCHCLASS_TAIL":
            error_counts["pitchclass_tail"] += 1

        event_rows.append({
            "event_id": ref.get("event_id", ""),
            "event_index": ref.get("event_index", ""),
            "expected_note": note,
            "reference_start_sec": f"{ref_start:.9f}",
            "reference_end_sec": f"{ref_end:.9f}",
            "real_start_sec": f"{real_start:.9f}",
            "real_end_sec": f"{real_end:.9f}",
            "start_frame": start_frame,
            "end_frame": end_frame,
            "duration_frames": duration_frames,
            "birth_exact_frame": birth_exact_frame,
            "birth_pc_frame": birth_pc_frame,
            "onset_status": onset_status,
            "exact_sustain_hits": exact_sustain_hits,
            "pc_sustain_hits": pc_sustain_hits,
            "exact_sustain_ratio": f"{exact_sustain_ratio:.9f}",
            "pc_sustain_ratio": f"{pc_sustain_ratio:.9f}",
            "sustain_status": sustain_status,
            "tail_exact_frames": tail_exact_frames,
            "tail_pc_frames": tail_pc_frames,
            "tail_status": tail_status,
        })

    out_csv = Path(args.out_event_audit_csv)
    out_txt = Path(args.out_summary_txt)
    out_json = Path(args.out_meta_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "event_id",
            "event_index",
            "expected_note",
            "reference_start_sec",
            "reference_end_sec",
            "real_start_sec",
            "real_end_sec",
            "start_frame",
            "end_frame",
            "duration_frames",
            "birth_exact_frame",
            "birth_pc_frame",
            "onset_status",
            "exact_sustain_hits",
            "pc_sustain_hits",
            "exact_sustain_ratio",
            "pc_sustain_ratio",
            "sustain_status",
            "tail_exact_frames",
            "tail_pc_frames",
            "tail_status",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(event_rows)

    summary = {
        "stage": "event_level_note_audit",
        "inputs": {
            "detected_frame_notes_csv": args.detected_frame_notes_csv,
            "reference_events_csv": args.reference_events_csv,
        },
        "parameters": {
            "detected_duration_sec": args.detected_duration_sec,
            "reference_duration_sec": args.reference_duration_sec,
            "tempo_ratio": tempo_ratio,
            "fps": args.fps,
            "onset_slack_frames": args.onset_slack_frames,
            "tail_window_frames": args.tail_window_frames,
            "false_carry_threshold_frames": args.false_carry_threshold_frames,
        },
        "result": {
            "events": len(event_rows),
            "onset_counts": dict(onset_counts),
            "sustain_counts": dict(sustain_counts),
            "tail_counts": dict(tail_counts),
            "error_counts": dict(error_counts),
        },
    }

    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "EVENT-LEVEL NOTE AUDIT",
        "=" * 72,
        f"events                : {len(event_rows)}",
        f"detected_frame_notes  : {args.detected_frame_notes_csv}",
        "",
        "Onset status:",
    ]
    for key, value in sorted(onset_counts.items()):
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("Sustain status:")
    for key, value in sorted(sustain_counts.items()):
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("Tail status:")
    for key, value in sorted(tail_counts.items()):
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("Error classes:")
    for key, value in sorted(error_counts.items()):
        lines.append(f"  {key}: {value}")

    out_txt.write_text("\n".join(lines), encoding="utf-8")

    print("event-level note audit complete")
    print(json.dumps(summary["result"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
