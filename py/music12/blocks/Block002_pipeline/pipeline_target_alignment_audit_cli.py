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


def _frame_counter(rows: list[dict[str, Any]], note_field: str) -> dict[int, set[str]]:
    out: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        frame_index = _safe_int(row.get("frame_index"), 0)
        token = str(row.get(note_field, "")).strip()
        if token:
            out[frame_index].add(token)
    return out


def _active_entity_counter(rows: list[dict[str, Any]]) -> dict[int, int]:
    out: dict[int, int] = Counter()
    for row in rows:
        start_frame = _safe_int(row.get("start_frame"), 0)
        end_frame = _safe_int(row.get("end_frame"), start_frame)
        for frame_index in range(start_frame, end_frame + 1):
            out[frame_index] += 1
    return dict(out)


def _active_entity_counter_filtered(rows: list[dict[str, Any]], allowed_kinds: set[str]) -> dict[int, int]:
    out: dict[int, int] = Counter()
    for row in rows:
        if str(row.get("event_kind", "")).strip() not in allowed_kinds:
            continue
        start_frame = _safe_int(row.get("start_frame"), 0)
        end_frame = _safe_int(row.get("end_frame"), start_frame)
        for frame_index in range(start_frame, end_frame + 1):
            out[frame_index] += 1
    return dict(out)


def _soft_onset_group_count(starts: list[int], window_frames: int) -> int:
    if not starts:
        return 0
    ordered = sorted(starts)
    count = 1
    anchor = ordered[0]
    for value in ordered[1:]:
        if value - anchor > window_frames:
            count += 1
            anchor = value
    return count


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare current Block002 emitted entity counts and polyphony against exact MIDI meta targets."
    )
    ap.add_argument("--midi-meta-json", required=True)
    ap.add_argument("--notechain-chains-csv", required=True)
    ap.add_argument("--notechain-frames-csv", required=True)
    ap.add_argument("--event-field-entities-csv", required=True)
    ap.add_argument("--event-field-frames-csv", required=True)
    ap.add_argument("--fused-events-csv", default="")
    ap.add_argument("--fused-onset-groups-csv", default="")
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--soft-onset-window-frames", type=int, default=5)
    args = ap.parse_args()

    midi_meta = json.loads(Path(args.midi_meta_json).read_text(encoding="utf-8"))
    notechain_entities = _load_csv(Path(args.notechain_chains_csv))
    notechain_frames = _load_csv(Path(args.notechain_frames_csv))
    event_entities = _load_csv(Path(args.event_field_entities_csv))
    event_frames = _load_csv(Path(args.event_field_frames_csv))
    fused_events = _load_csv(Path(args.fused_events_csv)) if str(args.fused_events_csv).strip() else []
    fused_onset_groups = _load_csv(Path(args.fused_onset_groups_csv)) if str(args.fused_onset_groups_csv).strip() else []

    target_event_count = _safe_int(midi_meta.get("event_count"), 0)
    target_onset_groups = _safe_int(midi_meta.get("unique_onset_groups"), 0)
    target_max_polyphony = _safe_int(midi_meta.get("max_onset_polyphony"), 0)

    notechain_onsets = {_safe_int(row.get("start_frame"), 0) for row in notechain_entities}
    event_onsets = {_safe_int(row.get("start_frame"), 0) for row in event_entities}
    combined_onsets = notechain_onsets | event_onsets
    combined_soft_onset_groups = _soft_onset_group_count(list(combined_onsets), int(args.soft_onset_window_frames))

    notechain_by_frame = _frame_counter(notechain_frames, "selected_note_token")
    event_by_frame = _frame_counter(event_frames, "dominant_note_token")
    notechain_max_polyphony = max((len(tokens) for tokens in notechain_by_frame.values()), default=0)
    event_max_field_density = max((len(tokens) for tokens in event_by_frame.values()), default=0)

    combined_active_entity_counter = _active_entity_counter(notechain_entities)
    event_active_entity_counter = _active_entity_counter(event_entities)
    for frame_index, count in event_active_entity_counter.items():
        combined_active_entity_counter[frame_index] = combined_active_entity_counter.get(frame_index, 0) + count
    combined_max_active_entities = max(combined_active_entity_counter.values(), default=0)

    fused_section_lines: list[str] = []
    fused_observed: dict[str, Any] = {}
    if fused_events:
        fused_musical_rows = [row for row in fused_events if str(row.get("event_kind", "")).strip() in {"notechain_backbone", "event_field_only"}]
        fused_residue_rows = [row for row in fused_events if str(row.get("event_kind", "")).strip() == "ambient_field_residue"]
        fused_onsets = {_safe_int(row.get("start_frame"), 0) for row in fused_musical_rows}
        fused_soft_onset_groups = _soft_onset_group_count(list(fused_onsets), int(args.soft_onset_window_frames))
        fused_active_entities = _active_entity_counter_filtered(fused_events, {"notechain_backbone", "event_field_only"})
        fused_max_active_entities = max(fused_active_entities.values(), default=0)
        fused_section_lines = [
            "",
            f"fused_musical_events     : {len(fused_musical_rows)}",
            f"fused_residue_events     : {len(fused_residue_rows)}",
            f"fused_onset_groups       : {len(fused_onsets)}",
            f"fused_soft_onset_groups  : {fused_soft_onset_groups}",
            f"fused_max_active_units   : {fused_max_active_entities}",
            "",
            f"fused_event_gap_to_target: {len(fused_musical_rows) - target_event_count}",
            f"fused_onset_gap_to_target: {len(fused_onsets) - target_onset_groups}",
            f"fused_soft_gap_to_target : {fused_soft_onset_groups - target_onset_groups}",
        ]
        if fused_onset_groups:
            fused_section_lines.extend(
                [
                    f"resolved_onset_groups    : {len(fused_onset_groups)}",
                    f"resolved_onset_gap       : {len(fused_onset_groups) - target_onset_groups}",
                ]
            )
        fused_observed = {
            "fused_musical_events": len(fused_musical_rows),
            "fused_residue_events": len(fused_residue_rows),
            "fused_onset_groups": len(fused_onsets),
            "fused_soft_onset_groups": fused_soft_onset_groups,
            "fused_max_active_units": fused_max_active_entities,
            "resolved_onset_groups": len(fused_onset_groups),
        }

    lines = [
        "PIPELINE TARGET ALIGNMENT AUDIT",
        "=" * 72,
        f"target_event_count        : {target_event_count}",
        f"target_onset_groups       : {target_onset_groups}",
        f"target_max_polyphony      : {target_max_polyphony}",
        "",
        f"notechain_entities        : {len(notechain_entities)}",
        f"notechain_onset_groups    : {len(notechain_onsets)}",
        f"notechain_max_polyphony   : {notechain_max_polyphony}",
        "",
        f"event_field_entities      : {len(event_entities)}",
        f"event_field_onset_groups  : {len(event_onsets)}",
        f"event_max_field_density   : {event_max_field_density}",
        "",
        f"combined_entities         : {len(notechain_entities) + len(event_entities)}",
        f"combined_onset_groups     : {len(combined_onsets)}",
        f"combined_soft_onset_groups: {combined_soft_onset_groups}",
        f"combined_max_active_units : {combined_max_active_entities}",
        "",
        f"entity_gap_to_target      : {len(notechain_entities) + len(event_entities) - target_event_count}",
        f"onset_gap_to_target       : {len(combined_onsets) - target_onset_groups}",
        f"soft_onset_gap_to_target  : {combined_soft_onset_groups - target_onset_groups}",
        f"polyphony_gap_to_target   : {notechain_max_polyphony - target_max_polyphony}",
    ]
    lines.extend(fused_section_lines)
    Path(args.out_summary_txt).write_text("\n".join(lines) + "\n", encoding="utf-8")

    meta = {
        "stage": "pipeline_target_alignment_audit",
        "inputs": {
            "midi_meta_json": args.midi_meta_json,
            "notechain_chains_csv": args.notechain_chains_csv,
            "notechain_frames_csv": args.notechain_frames_csv,
            "event_field_entities_csv": args.event_field_entities_csv,
            "event_field_frames_csv": args.event_field_frames_csv,
            "fused_events_csv": args.fused_events_csv,
            "fused_onset_groups_csv": args.fused_onset_groups_csv,
        },
        "parameters": {
            "soft_onset_window_frames": int(args.soft_onset_window_frames),
        },
        "target": {
            "event_count": target_event_count,
            "unique_onset_groups": target_onset_groups,
            "max_onset_polyphony": target_max_polyphony,
        },
        "observed": {
            "notechain_entities": len(notechain_entities),
            "notechain_onset_groups": len(notechain_onsets),
            "notechain_max_polyphony": notechain_max_polyphony,
            "event_field_entities": len(event_entities),
            "event_field_onset_groups": len(event_onsets),
            "event_max_field_density": event_max_field_density,
            "combined_entities": len(notechain_entities) + len(event_entities),
            "combined_onset_groups": len(combined_onsets),
            "combined_soft_onset_groups": combined_soft_onset_groups,
            "combined_max_active_units": combined_max_active_entities,
            **fused_observed,
        },
    }
    Path(args.out_meta_json).write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
