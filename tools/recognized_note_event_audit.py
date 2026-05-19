from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _load_csv(path: Path) -> list[dict[str, Any]]:
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


def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start) + 1)


def _musical_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if str(row.get("event_kind", "")).strip() in {"notechain_backbone", "event_field_only"}
    ]


def _build_onset_group_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("onset_group_id", "")).strip()
        if key:
            out[key] = row
    return out


def _json_list(value: Any) -> list[str]:
    try:
        raw = json.loads(str(value or "[]"))
        if isinstance(raw, list):
            return [str(x).strip() for x in raw if str(x).strip()]
    except Exception:
        return []
    return []


def _nearby_events(rows: list[dict[str, Any]], start_frame: int, window: int) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        row_start = _safe_int(row.get("start_frame"), 0)
        if abs(row_start - start_frame) <= window:
            out.append(row)
    out.sort(key=lambda r: abs(_safe_int(r.get("start_frame"), 0) - start_frame))
    return out


def _best_chain_match(
    chain_rows: list[dict[str, Any]],
    start_frame: int,
    end_frame: int,
    note: str,
    window: int,
) -> tuple[dict[str, Any] | None, str, int]:
    best_row: dict[str, Any] | None = None
    best_kind = ""
    best_delta = 10**9
    best_priority = 10**9

    note_pc = _pitch_class(note)

    for row in chain_rows:
        row_note = _normalize_note(
            row.get("main_note_token", "")
            or row.get("dominant_note_token", "")
            or row.get("anchor_note_token", "")
        )
        row_start = _safe_int(row.get("start_frame"), 0)
        row_end = _safe_int(row.get("end_frame"), row_start)
        if abs(row_start - start_frame) > window and _overlap(row_start, row_end, start_frame, end_frame) <= 0:
            continue

        if row_note == note:
            kind = "EXACT"
            priority = 0
        elif _pitch_class(row_note) == note_pc:
            kind = "PITCHCLASS"
            priority = 1
        else:
            kind = "WRONG_NOTE"
            priority = 2

        delta = abs(row_start - start_frame)
        overlap = _overlap(row_start, row_end, start_frame, end_frame)
        # Better relation first, then overlap, then time delta.
        score_tuple = (priority, -overlap, delta)
        best_tuple = (best_priority, -_overlap(
            _safe_int(best_row.get("start_frame"), 0),
            _safe_int(best_row.get("end_frame"), _safe_int(best_row.get("start_frame"), 0)),
            start_frame,
            end_frame,
        ) if best_row else 0, best_delta)
        if best_row is None or score_tuple < best_tuple:
            best_row = row
            best_kind = kind
            best_delta = delta
            best_priority = priority

    return best_row, best_kind, best_delta


def _classify_status(
    *,
    match_kind: str,
    start_delta: int,
    exact_proto_nearby: int,
    pitch_proto_nearby: int,
    nearby_event_count: int,
    exact_on_time_frames: int,
    late_window_frames: int,
) -> str:
    if match_kind == "EXACT":
        if start_delta <= exact_on_time_frames:
            return "EXACT_NOTE_ON_TIME"
        if start_delta <= late_window_frames:
            return "EXACT_NOTE_LATE"
        return "EXACT_NOTE_FAR"

    if match_kind == "PITCHCLASS":
        if start_delta <= exact_on_time_frames:
            return "PITCHCLASS_ON_TIME"
        if start_delta <= late_window_frames:
            return "PITCHCLASS_LATE"
        return "PITCHCLASS_FAR"

    if nearby_event_count > 0:
        return "WRONG_NOTE_EVENT"
    if exact_proto_nearby > 0:
        return "PROTO_EXACT_ONLY"
    if pitch_proto_nearby > 0:
        return "PROTO_PITCHCLASS_ONLY"
    return "TOTAL_MISS"


def _classify_breathing_status(
    *,
    strict_status: str,
    match_kind: str,
    midi_note: str,
    best_row: dict[str, Any] | None,
    start_frame: int,
    onset_groups_by_id: dict[str, dict[str, Any]],
    breathing_window_frames: int,
) -> str:
    if best_row is None:
        if strict_status in {"PROTO_EXACT_ONLY", "PROTO_PITCHCLASS_ONLY"}:
            return strict_status
        return "TOTAL_MISS"

    onset_group_id = str(best_row.get("onset_group_id", "")).strip()
    onset_row = onset_groups_by_id.get(onset_group_id, {})
    anchor_frame = _safe_int(onset_row.get("anchor_frame"), _safe_int(best_row.get("onset_anchor_frame"), 0))
    start_min = _safe_int(onset_row.get("start_min_frame"), _safe_int(best_row.get("start_frame"), 0))
    start_max = _safe_int(onset_row.get("start_max_frame"), _safe_int(best_row.get("start_frame"), 0))
    group_notes = [_normalize_note(x) for x in _json_list(onset_row.get("main_note_tokens_json", "[]"))]
    midi_pc = _pitch_class(midi_note)
    group_exact = midi_note in group_notes
    group_pitch = any(_pitch_class(note) == midi_pc for note in group_notes if note)

    in_breath_window = (
        abs(anchor_frame - start_frame) <= breathing_window_frames
        or start_min - breathing_window_frames <= start_frame <= start_max + breathing_window_frames
    )

    if in_breath_window and group_exact:
        return "EXACT_NOTE_IN_BREATH_GROUP"
    if in_breath_window and group_pitch:
        return "PITCHCLASS_IN_BREATH_GROUP"

    if match_kind == "EXACT":
        return "EXACT_NOTE_IN_BREATH_WINDOW" if in_breath_window else "EXACT_NOTE_OUTSIDE_BREATH"
    if match_kind == "PITCHCLASS":
        return "PITCHCLASS_IN_BREATH_WINDOW" if in_breath_window else "PITCHCLASS_OUTSIDE_BREATH"
    if strict_status == "WRONG_NOTE_EVENT":
        return "WRONG_NOTE_IN_BREATH_WINDOW" if in_breath_window else "WRONG_NOTE_OUTSIDE_BREATH"
    return strict_status


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit how many real MIDI note-events are recognized as note-events, where they shift in time, and where they remain unseen."
    )
    ap.add_argument("--midi-events-csv", required=True)
    ap.add_argument("--fused-events-anchored-csv", required=True)
    ap.add_argument("--fused-onset-groups-csv", required=True)
    ap.add_argument("--controlled-sustain-chains-csv", required=True)
    ap.add_argument("--proto-exciters-csv", required=True)
    ap.add_argument("--out-audit-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--match-window-frames", type=int, default=12)
    ap.add_argument("--exact-on-time-frames", type=int, default=2)
    ap.add_argument("--late-window-frames", type=int, default=8)
    ap.add_argument("--breathing-window-frames", type=int, default=5)
    args = ap.parse_args()

    midi_rows = _load_csv(Path(args.midi_events_csv))
    fused_rows = _musical_rows(_load_csv(Path(args.fused_events_anchored_csv)))
    onset_group_rows = _load_csv(Path(args.fused_onset_groups_csv))
    chain_rows = _load_csv(Path(args.controlled_sustain_chains_csv))
    proto_rows = _load_csv(Path(args.proto_exciters_csv))
    onset_groups_by_id = _build_onset_group_by_id(onset_group_rows)

    proto_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in proto_rows:
        proto_by_frame[_safe_int(row.get("start_frame"), 0)].append(row)

    audit_rows: list[dict[str, Any]] = []
    status_counter: Counter[str] = Counter()
    breathing_counter: Counter[str] = Counter()
    polyphony_counter: dict[int, Counter[str]] = defaultdict(Counter)
    breathing_polyphony_counter: dict[int, Counter[str]] = defaultdict(Counter)

    for midi in midi_rows:
        note = _normalize_note(midi.get("expected_note_token", midi.get("note_token", "")))
        note_pc = _pitch_class(note)
        start_frame = _safe_int(midi.get("start_frame60"), 0)
        end_frame = _safe_int(midi.get("end_frame60"), start_frame)
        onset_group = str(midi.get("onset_group", "")).strip()
        onset_polyphony = _safe_int(midi.get("onset_polyphony"), 0)

        nearby_fused = _nearby_events(fused_rows, start_frame, int(args.match_window_frames))
        best_row, match_kind, start_delta = _best_chain_match(
            chain_rows=nearby_fused,
            start_frame=start_frame,
            end_frame=end_frame,
            note=note,
            window=int(args.match_window_frames),
        )

        exact_proto_nearby = 0
        pitch_proto_nearby = 0
        for frame in range(start_frame - int(args.exact_on_time_frames), start_frame + int(args.exact_on_time_frames) + 1):
            for proto in proto_by_frame.get(frame, []):
                proto_note = _normalize_note(
                    proto.get("rescue_group_dominant_note", "")
                    or proto.get("coarse_note", "")
                )
                if not proto_note:
                    continue
                if proto_note == note:
                    exact_proto_nearby += 1
                elif _pitch_class(proto_note) == note_pc:
                    pitch_proto_nearby += 1

        status = _classify_status(
            match_kind=match_kind,
            start_delta=start_delta if best_row is not None else 10**9,
            exact_proto_nearby=exact_proto_nearby,
            pitch_proto_nearby=pitch_proto_nearby,
            nearby_event_count=len(nearby_fused),
            exact_on_time_frames=int(args.exact_on_time_frames),
            late_window_frames=int(args.late_window_frames),
        )
        status_counter[status] += 1
        polyphony_counter[onset_polyphony][status] += 1
        breathing_status = _classify_breathing_status(
            strict_status=status,
            match_kind=match_kind,
            midi_note=note,
            best_row=best_row,
            start_frame=start_frame,
            onset_groups_by_id=onset_groups_by_id,
            breathing_window_frames=int(args.breathing_window_frames),
        )
        breathing_counter[breathing_status] += 1
        breathing_polyphony_counter[onset_polyphony][breathing_status] += 1

        matched_note = _normalize_note(best_row.get("main_note_token", "")) if best_row else ""
        matched_start = _safe_int(best_row.get("start_frame"), 0) if best_row else ""
        matched_kind = str(best_row.get("event_kind", "")) if best_row else ""
        matched_support = str(best_row.get("field_support_kind", "")) if best_row else ""
        matched_onset_group_id = str(best_row.get("onset_group_id", "")).strip() if best_row else ""

        audit_rows.append(
            {
                "event_index": midi.get("event_index", ""),
                "midi_note_token": note,
                "midi_pitch_class": note_pc,
                "start_frame60": start_frame,
                "end_frame60": end_frame,
                "onset_group": onset_group,
                "onset_polyphony": onset_polyphony,
                "strict_status": status,
                "breathing_status": breathing_status,
                "matched_kind": match_kind,
                "matched_event_note": matched_note,
                "matched_event_start_frame": matched_start,
                "start_frame_delta": start_delta if best_row is not None else "",
                "matched_event_kind": matched_kind,
                "matched_field_support_kind": matched_support,
                "matched_onset_group_id": matched_onset_group_id,
                "nearby_musical_event_count": len(nearby_fused),
                "exact_proto_nearby": exact_proto_nearby,
                "pitch_proto_nearby": pitch_proto_nearby,
                "nearby_event_notes_json": json.dumps(
                    [_normalize_note(row.get("main_note_token", "")) for row in nearby_fused],
                    ensure_ascii=False,
                ),
            }
        )

    out_csv = Path(args.out_audit_csv)
    out_summary = Path(args.out_summary_txt)
    out_meta = Path(args.out_meta_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if audit_rows:
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
            w.writeheader()
            w.writerows(audit_rows)

    lines = [
        "RECOGNIZED NOTE EVENT AUDIT",
        "=" * 72,
        f"midi_event_count          : {len(midi_rows)}",
        f"fused_musical_event_count : {len(fused_rows)}",
        "",
        "STRICT STATUS COUNTS",
        "-" * 72,
    ]
    for key in sorted(status_counter):
        lines.append(f"{key:26s}: {status_counter[key]}")
    lines.extend(["", "BREATHING STATUS COUNTS", "-" * 72])
    for key in sorted(breathing_counter):
        lines.append(f"{key:26s}: {breathing_counter[key]}")
    lines.extend(["", "BY ONSET POLYPHONY", "-" * 72])
    for poly in sorted(polyphony_counter):
        lines.append(f"polyphony={poly}")
        for key in sorted(polyphony_counter[poly]):
            lines.append(f"  strict {key:17s}: {polyphony_counter[poly][key]}")
        for key in sorted(breathing_polyphony_counter[poly]):
            lines.append(f"  breath {key:17s}: {breathing_polyphony_counter[poly][key]}")
    out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    meta = {
        "stage": "recognized_note_event_audit",
        "inputs": {
            "midi_events_csv": args.midi_events_csv,
            "fused_events_anchored_csv": args.fused_events_anchored_csv,
            "fused_onset_groups_csv": args.fused_onset_groups_csv,
            "controlled_sustain_chains_csv": args.controlled_sustain_chains_csv,
            "proto_exciters_csv": args.proto_exciters_csv,
        },
        "parameters": {
            "match_window_frames": int(args.match_window_frames),
            "exact_on_time_frames": int(args.exact_on_time_frames),
            "late_window_frames": int(args.late_window_frames),
            "breathing_window_frames": int(args.breathing_window_frames),
        },
        "result": {
            "strict_status_counter": dict(status_counter),
            "breathing_status_counter": dict(breathing_counter),
            "strict_polyphony_counter": {str(k): dict(v) for k, v in polyphony_counter.items()},
            "breathing_polyphony_counter": {str(k): dict(v) for k, v in breathing_polyphony_counter.items()},
        },
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
