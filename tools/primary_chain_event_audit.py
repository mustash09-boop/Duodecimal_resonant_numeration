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


def _build_frame_index(rows: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[_safe_int(row.get("frame_index"), 0)].append(row)
    return out


def _frame_notes(rows: List[Dict[str, Any]]) -> Set[str]:
    out: Set[str] = set()
    for row in rows:
        note = _normalize_note(row.get("selected_note_token", row.get("note_token", "")))
        if note:
            out.add(note)
    return out


def _classify_onset(start_frame: int, birth_exact_frame: int, birth_pc_frame: int, onset_slack_frames: int) -> str:
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
        description="Event-level audit for primary_note_chain output with polyphonic overlap preserved."
    )
    ap.add_argument("--chain-frames-csv", required=True)
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

    chain_rows = _load_csv(Path(args.chain_frames_csv))
    ref_rows = _load_csv(Path(args.reference_events_csv))
    detected_by_frame = _build_frame_index(chain_rows)

    tempo_ratio = args.detected_duration_sec / max(args.reference_duration_sec, 1e-9)

    event_rows: List[Dict[str, Any]] = []
    onset_counts: Counter[str] = Counter()
    sustain_counts: Counter[str] = Counter()
    tail_counts: Counter[str] = Counter()

    for ref in ref_rows:
        note = _normalize_note(
            ref.get("expected_note_token")
            or ref.get("expected_note")
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

        target_pc = _pitch_class(note)
        onset_from = max(0, start_frame - 1)

        birth_exact_frame = -1
        birth_pc_frame = -1
        exact_sustain_hits = 0
        pc_sustain_hits = 0
        competing_note_frames = 0
        max_polyphony = 0

        for frame in range(onset_from, end_frame + 1):
            notes = _frame_notes(detected_by_frame.get(frame, []))
            if not notes:
                continue
            max_polyphony = max(max_polyphony, len(notes))
            if len(notes) > 1:
                competing_note_frames += 1
            if birth_exact_frame < 0 and note in notes:
                birth_exact_frame = frame
            if birth_pc_frame < 0 and target_pc and target_pc in {_pitch_class(n) for n in notes}:
                birth_pc_frame = frame
            if frame >= start_frame:
                if note in notes:
                    exact_sustain_hits += 1
                if target_pc and target_pc in {_pitch_class(n) for n in notes}:
                    pc_sustain_hits += 1

        duration_frames = max(end_frame - start_frame + 1, 1)
        exact_ratio = exact_sustain_hits / duration_frames
        pc_ratio = pc_sustain_hits / duration_frames

        onset_status = _classify_onset(start_frame, birth_exact_frame, birth_pc_frame, args.onset_slack_frames)
        sustain_status = _classify_sustain(exact_ratio, pc_ratio)

        tail_exact_frames = 0
        tail_pc_frames = 0
        for frame in range(end_frame + 1, end_frame + args.tail_window_frames + 1):
            notes = _frame_notes(detected_by_frame.get(frame, []))
            if note in notes:
                tail_exact_frames += 1
            if target_pc and target_pc in {_pitch_class(n) for n in notes}:
                tail_pc_frames += 1

        tail_status = _classify_tail(tail_exact_frames, tail_pc_frames, args.false_carry_threshold_frames)

        onset_counts[onset_status] += 1
        sustain_counts[sustain_status] += 1
        tail_counts[tail_status] += 1

        event_rows.append(
            {
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
                "exact_sustain_ratio": f"{exact_ratio:.9f}",
                "pc_sustain_ratio": f"{pc_ratio:.9f}",
                "sustain_status": sustain_status,
                "tail_exact_frames": tail_exact_frames,
                "tail_pc_frames": tail_pc_frames,
                "tail_status": tail_status,
                "competing_note_frames": competing_note_frames,
                "max_polyphony_in_window": max_polyphony,
            }
        )

    out_csv = Path(args.out_event_audit_csv)
    out_txt = Path(args.out_summary_txt)
    out_json = Path(args.out_meta_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

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
        "competing_note_frames",
        "max_polyphony_in_window",
    ]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(event_rows)

    summary = {
        "stage": "primary_chain_event_audit",
        "inputs": {
            "chain_frames_csv": args.chain_frames_csv,
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
        },
    }

    lines = [
        "PRIMARY NOTE CHAIN EVENT AUDIT",
        "=" * 72,
        f"events                : {len(event_rows)}",
        "",
        "Onset status:",
    ]
    for key in sorted(onset_counts):
        lines.append(f"  {key}: {onset_counts[key]}")
    lines.extend(["", "Sustain status:"])
    for key in sorted(sustain_counts):
        lines.append(f"  {key}: {sustain_counts[key]}")
    lines.extend(["", "Tail status:"])
    for key in sorted(tail_counts):
        lines.append(f"  {key}: {tail_counts[key]}")

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
