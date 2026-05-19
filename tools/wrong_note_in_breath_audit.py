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


def _octave(note: str) -> str:
    try:
        return _normalize_note(note).split(".", 1)[0]
    except Exception:
        return ""


def _relation(midi_note: str, matched_note: str) -> str:
    midi_note = _normalize_note(midi_note)
    matched_note = _normalize_note(matched_note)
    if not midi_note or not matched_note:
        return "UNRESOLVED"
    if midi_note == matched_note:
        return "EXACT"
    if _pitch_class(midi_note) == _pitch_class(matched_note) and _octave(midi_note) != _octave(matched_note):
        return "OCTAVE_SUBSTITUTION"
    if _octave(midi_note) == _octave(matched_note) and _pitch_class(midi_note) != _pitch_class(matched_note):
        return "STEP_SUBSTITUTION"
    return "FOREIGN_SUBSTITUTION"


def _time_band(delta: int) -> str:
    if delta <= 2:
        return "VERY_CLOSE_0_2"
    if delta <= 5:
        return "CLOSE_3_5"
    if delta <= 8:
        return "MID_6_8"
    return "FAR_9_PLUS"


def _source_mode(row: dict[str, Any]) -> str:
    event_kind = str(row.get("matched_event_kind", "")).strip()
    support = str(row.get("matched_field_support_kind", "")).strip()
    if event_kind == "event_field_only":
        return "FIELD_BRANCH_CAPTURE"
    if event_kind == "notechain_backbone" and support == "exact_onset_support":
        return "NOTECHAIN_EXACT_SUPPORT_CAPTURE"
    if event_kind == "notechain_backbone" and support == "foreign_field_support":
        return "NOTECHAIN_FOREIGN_FIELD_CAPTURE"
    if event_kind == "notechain_backbone" and support == "pitchclass_onset_support":
        return "NOTECHAIN_PITCHCLASS_SUPPORT_CAPTURE"
    if event_kind == "notechain_backbone":
        return "NOTECHAIN_OTHER_CAPTURE"
    return "UNRESOLVED_SOURCE"


def _proto_visibility(row: dict[str, Any]) -> str:
    exact_proto = _safe_int(row.get("exact_proto_nearby"), 0)
    pitch_proto = _safe_int(row.get("pitch_proto_nearby"), 0)
    if exact_proto > 0:
        return "EXACT_PROTO_WAS_NEARBY"
    if pitch_proto > 0:
        return "PITCHCLASS_PROTO_WAS_NEARBY"
    return "NO_PROTO_TRACE"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Analyze cases where the algorithm lands in the correct musical breathing window but chooses the wrong note."
    )
    ap.add_argument("--recognized-audit-csv", required=True)
    ap.add_argument("--out-audit-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    args = ap.parse_args()

    rows = _load_csv(Path(args.recognized_audit_csv))
    target_rows = [
        row for row in rows
        if str(row.get("breathing_status", "")).strip() == "WRONG_NOTE_IN_BREATH_WINDOW"
    ]

    relation_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    time_counter: Counter[str] = Counter()
    proto_counter: Counter[str] = Counter()
    polyphony_counter: dict[int, Counter[str]] = defaultdict(Counter)
    onset_group_counter: Counter[str] = Counter()

    out_rows: list[dict[str, Any]] = []
    for row in target_rows:
        midi_note = _normalize_note(row.get("midi_note_token", ""))
        matched_note = _normalize_note(row.get("matched_event_note", ""))
        rel = _relation(midi_note, matched_note)
        src = _source_mode(row)
        delta = _safe_int(row.get("start_frame_delta"), 999999)
        band = _time_band(delta)
        proto = _proto_visibility(row)
        poly = _safe_int(row.get("onset_polyphony"), 0)
        onset_group_id = str(row.get("matched_onset_group_id", "")).strip()

        relation_counter[rel] += 1
        source_counter[src] += 1
        time_counter[band] += 1
        proto_counter[proto] += 1
        polyphony_counter[poly][rel] += 1
        if onset_group_id:
            onset_group_counter[onset_group_id] += 1

        out_rows.append(
            {
                "event_index": row.get("event_index", ""),
                "midi_note_token": midi_note,
                "matched_event_note": matched_note,
                "relation_type": rel,
                "source_mode": src,
                "time_band": band,
                "proto_visibility": proto,
                "start_frame60": row.get("start_frame60", ""),
                "start_frame_delta": row.get("start_frame_delta", ""),
                "onset_polyphony": row.get("onset_polyphony", ""),
                "matched_onset_group_id": onset_group_id,
                "matched_event_kind": row.get("matched_event_kind", ""),
                "matched_field_support_kind": row.get("matched_field_support_kind", ""),
                "nearby_event_notes_json": row.get("nearby_event_notes_json", ""),
            }
        )

    repeated_groups = sum(1 for _, count in onset_group_counter.items() if count > 1)

    out_csv = Path(args.out_audit_csv)
    out_summary = Path(args.out_summary_txt)
    out_meta = Path(args.out_meta_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if out_rows:
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)

    lines = [
        "WRONG NOTE IN BREATH WINDOW AUDIT",
        "=" * 72,
        f"case_count                    : {len(out_rows)}",
        f"reused_matched_onset_groups   : {repeated_groups}",
        "",
        "RELATION TYPES",
        "-" * 72,
    ]
    for key in sorted(relation_counter):
        lines.append(f"{key:28s}: {relation_counter[key]}")
    lines.extend(["", "SOURCE MODES", "-" * 72])
    for key in sorted(source_counter):
        lines.append(f"{key:28s}: {source_counter[key]}")
    lines.extend(["", "TIME BANDS", "-" * 72])
    for key in sorted(time_counter):
        lines.append(f"{key:28s}: {time_counter[key]}")
    lines.extend(["", "PROTO VISIBILITY", "-" * 72])
    for key in sorted(proto_counter):
        lines.append(f"{key:28s}: {proto_counter[key]}")
    lines.extend(["", "BY ONSET POLYPHONY", "-" * 72])
    for poly in sorted(polyphony_counter):
        lines.append(f"polyphony={poly}")
        for key in sorted(polyphony_counter[poly]):
            lines.append(f"  {key:26s}: {polyphony_counter[poly][key]}")
    out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    meta = {
        "stage": "wrong_note_in_breath_audit",
        "inputs": {
            "recognized_audit_csv": args.recognized_audit_csv,
        },
        "result": {
            "case_count": len(out_rows),
            "reused_matched_onset_groups": repeated_groups,
            "relation_counter": dict(relation_counter),
            "source_counter": dict(source_counter),
            "time_counter": dict(time_counter),
            "proto_counter": dict(proto_counter),
            "polyphony_counter": {str(k): dict(v) for k, v in polyphony_counter.items()},
        },
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
