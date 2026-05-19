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


def _nearby(rows: list[dict[str, Any]], frame: int, window: int) -> list[dict[str, Any]]:
    return [row for row in rows if abs(_safe_int(row.get("start_frame"), 0) - frame) <= window]


def _window_sweep_counts(
    midi_onset_groups: dict[str, list[dict[str, Any]]],
    midi_rows: list[dict[str, Any]],
    onset_group_rows: list[dict[str, Any]],
    musical_rows: list[dict[str, Any]],
    windows: list[int],
) -> list[dict[str, int]]:
    out: list[dict[str, int]] = []
    for window in windows:
        onset_covered = 0
        for group_rows in midi_onset_groups.values():
            start_frame = min(_safe_int(row.get("start_frame60"), 0) for row in group_rows)
            if any(abs(_safe_int(row.get("anchor_frame"), 0) - start_frame) <= window for row in onset_group_rows):
                onset_covered += 1

        exact_event_covered = 0
        pitch_event_covered = 0
        for row in midi_rows:
            midi_start = _safe_int(row.get("start_frame60"), 0)
            midi_note = _normalize_note(row.get("expected_note_token", row.get("note_token", "")))
            midi_pc = _pitch_class(midi_note)
            if any(
                _normalize_note(r.get("main_note_token", "")) == midi_note
                and abs(_safe_int(r.get("start_frame"), 0) - midi_start) <= window
                for r in musical_rows
            ):
                exact_event_covered += 1
            elif any(
                _pitch_class(r.get("main_note_token", "")) == midi_pc
                and abs(_safe_int(r.get("start_frame"), 0) - midi_start) <= window
                for r in musical_rows
            ):
                pitch_event_covered += 1

        out.append(
            {
                "window": window,
                "onset_covered": onset_covered,
                "exact_event_covered": exact_event_covered,
                "pitch_event_additional": pitch_event_covered,
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit the final gap between MIDI truth and fused musical events / onset groups."
    )
    ap.add_argument("--midi-events-csv", required=True)
    ap.add_argument("--fused-events-anchored-csv", required=True)
    ap.add_argument("--fused-onset-groups-csv", required=True)
    ap.add_argument("--branch-analysis-csv", required=True)
    ap.add_argument("--out-onset-audit-csv", required=True)
    ap.add_argument("--out-event-audit-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--onset-window-frames", type=int, default=3)
    args = ap.parse_args()

    midi_rows = _load_csv(Path(args.midi_events_csv))
    fused_rows = _load_csv(Path(args.fused_events_anchored_csv))
    onset_group_rows = _load_csv(Path(args.fused_onset_groups_csv))
    branch_rows = _load_csv(Path(args.branch_analysis_csv))

    musical_rows = [row for row in fused_rows if str(row.get("event_kind", "")).strip() in {"notechain_backbone", "event_field_only"}]
    residue_rows = [row for row in fused_rows if str(row.get("event_kind", "")).strip() == "ambient_field_residue"]

    midi_onset_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in midi_rows:
        midi_onset_groups[str(row.get("onset_group", "")).strip()].append(row)

    onset_audit_rows: list[dict[str, Any]] = []
    event_audit_rows: list[dict[str, Any]] = []
    onset_status_counter: Counter[str] = Counter()
    event_status_counter: Counter[str] = Counter()
    resolved_group_to_midi: defaultdict[str, list[str]] = defaultdict(list)
    window_sweep = _window_sweep_counts(midi_onset_groups, midi_rows, onset_group_rows, musical_rows, [3, 5, 8, 12, 16, 20])

    # First pass: map MIDI onset groups to nearby resolved onset groups.
    midi_group_matches: dict[str, list[str]] = {}
    for onset_group_id, group_rows in midi_onset_groups.items():
        start_frame = min(_safe_int(row.get("start_frame60"), 0) for row in group_rows)
        nearby_groups = [
            row for row in onset_group_rows
            if abs(_safe_int(row.get("anchor_frame"), 0) - start_frame) <= int(args.onset_window_frames)
        ]
        resolved_ids = [str(row.get("onset_group_id", "")).strip() for row in nearby_groups if str(row.get("onset_group_id", "")).strip()]
        midi_group_matches[onset_group_id] = resolved_ids
        for resolved_id in resolved_ids:
            resolved_group_to_midi[resolved_id].append(onset_group_id)

    for onset_group_id, group_rows in sorted(midi_onset_groups.items(), key=lambda kv: _safe_int(kv[0], 0)):
        start_frame = min(_safe_int(row.get("start_frame60"), 0) for row in group_rows)
        notes = [_normalize_note(row.get("expected_note_token", row.get("note_token", ""))) for row in group_rows]
        resolved_ids = midi_group_matches.get(onset_group_id, [])
        if not resolved_ids:
            status = "MISSING_ONSET_GROUP"
        elif any(len(set(resolved_group_to_midi[rid])) > 1 for rid in resolved_ids):
            status = "OVERMERGED_ONSET_GROUP"
        else:
            status = "ONSET_GROUP_RESOLVED"
        onset_status_counter[status] += 1
        onset_audit_rows.append(
            {
                "midi_onset_group": onset_group_id,
                "start_frame60": start_frame,
                "midi_note_count": len(group_rows),
                "midi_notes_json": json.dumps(notes, ensure_ascii=False),
                "matched_resolved_onset_groups_json": json.dumps(resolved_ids, ensure_ascii=False),
                "status": status,
            }
        )

    # Event-level pass.
    for row in midi_rows:
        midi_start = _safe_int(row.get("start_frame60"), 0)
        midi_note = _normalize_note(row.get("expected_note_token", row.get("note_token", "")))
        midi_pc = _pitch_class(midi_note)
        onset_group_id = str(row.get("onset_group", "")).strip()

        nearby_musical = _nearby(musical_rows, midi_start, int(args.onset_window_frames))
        nearby_residue = _nearby(residue_rows, midi_start, int(args.onset_window_frames))
        nearby_proto = [proto for proto in branch_rows if abs(_safe_int(proto.get("start_frame"), 0) - midi_start) <= int(args.onset_window_frames)]

        exact_musical = [r for r in nearby_musical if _normalize_note(r.get("main_note_token", "")) == midi_note]
        pitch_musical = [r for r in nearby_musical if _pitch_class(r.get("main_note_token", "")) == midi_pc]
        exact_residue = [r for r in nearby_residue if _normalize_note(r.get("main_note_token", "")) == midi_note]
        pitch_residue = [r for r in nearby_residue if _pitch_class(r.get("main_note_token", "")) == midi_pc]
        exact_proto = [r for r in nearby_proto if _normalize_note(r.get("coarse_note", "")) == midi_note]
        pitch_proto = [r for r in nearby_proto if _pitch_class(r.get("coarse_note", "")) == midi_pc]

        if exact_musical:
            status = "EXACT_FUSED_MATCH"
        elif pitch_musical:
            status = "PITCHCLASS_FUSED_ONLY"
        elif exact_residue or pitch_residue:
            status = "AMBIENT_RESCUE_CANDIDATE"
        elif exact_proto or pitch_proto:
            status = "PROTO_ONLY_VISIBLE"
        elif onset_group_id and any(a["midi_onset_group"] == onset_group_id and a["status"] == "OVERMERGED_ONSET_GROUP" for a in onset_audit_rows):
            status = "OVERMERGED_GROUP_EVENT_LOSS"
        else:
            status = "TOTAL_EVENT_MISS"

        event_status_counter[status] += 1
        event_audit_rows.append(
            {
                "event_index": row.get("event_index", ""),
                "midi_onset_group": onset_group_id,
                "start_frame60": midi_start,
                "expected_note_token": midi_note,
                "status": status,
                "nearby_musical_count": len(nearby_musical),
                "nearby_residue_count": len(nearby_residue),
                "nearby_proto_count": len(nearby_proto),
                "nearby_musical_notes_json": json.dumps([str(r.get("main_note_token", "")).strip() for r in nearby_musical], ensure_ascii=False),
                "nearby_residue_notes_json": json.dumps([str(r.get("main_note_token", "")).strip() for r in nearby_residue], ensure_ascii=False),
                "nearby_proto_notes_json": json.dumps([_normalize_note(r.get("coarse_note", "")) for r in nearby_proto], ensure_ascii=False),
            }
        )

    out_onset = Path(args.out_onset_audit_csv)
    out_event = Path(args.out_event_audit_csv)
    out_summary = Path(args.out_summary_txt)
    out_meta = Path(args.out_meta_json)
    out_onset.parent.mkdir(parents=True, exist_ok=True)

    if onset_audit_rows:
        with out_onset.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(onset_audit_rows[0].keys()))
            w.writeheader()
            w.writerows(onset_audit_rows)

    if event_audit_rows:
        with out_event.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(event_audit_rows[0].keys()))
            w.writeheader()
            w.writerows(event_audit_rows)

    summary_lines = [
        "FUSED TARGET GAP AUDIT",
        "=" * 72,
        f"midi_event_count         : {len(midi_rows)}",
        f"midi_onset_group_count   : {len(midi_onset_groups)}",
        "",
        "ONSET STATUS COUNTS",
        "-" * 72,
    ]
    for key in sorted(onset_status_counter):
        summary_lines.append(f"{key:<24}: {onset_status_counter[key]}")
    summary_lines.extend(["", "EVENT STATUS COUNTS", "-" * 72])
    for key in sorted(event_status_counter):
        summary_lines.append(f"{key:<24}: {event_status_counter[key]}")
    summary_lines.extend(["", "WINDOW SWEEP", "-" * 72])
    for row in window_sweep:
        summary_lines.append(
            f"window={row['window']:<2} onset_covered={row['onset_covered']:<3} exact_event_covered={row['exact_event_covered']:<3} pitch_event_additional={row['pitch_event_additional']:<3}"
        )
    out_summary.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    meta = {
        "stage": "fused_target_gap_audit",
        "inputs": {
            "midi_events_csv": args.midi_events_csv,
            "fused_events_anchored_csv": args.fused_events_anchored_csv,
            "fused_onset_groups_csv": args.fused_onset_groups_csv,
            "branch_analysis_csv": args.branch_analysis_csv,
        },
        "parameters": {
            "onset_window_frames": int(args.onset_window_frames),
        },
        "result": {
            "onset_status_counter": dict(onset_status_counter),
            "event_status_counter": dict(event_status_counter),
            "window_sweep": window_sweep,
        },
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
