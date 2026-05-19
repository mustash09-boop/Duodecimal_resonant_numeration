from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Promote compact attack-like event-field groups into notechain rescue audition candidates."
    )
    ap.add_argument("--event-field-groups-csv", required=True)
    ap.add_argument("--event-field-proto-exciters-csv", required=True)
    ap.add_argument("--notechain-proto-exciters-csv", required=True)
    ap.add_argument("--out-rescue-proto-exciters-csv", required=True)
    ap.add_argument("--out-combined-notechain-proto-exciters-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--max-onset-span-frames", type=int, default=4)
    ap.add_argument("--min-onset-compactness", type=float, default=0.80)
    ap.add_argument("--min-attack-ratio", type=float, default=0.05)
    ap.add_argument("--min-attack-compactness", type=float, default=0.70)
    ap.add_argument("--min-exciter-confidence", type=float, default=0.72)
    args = ap.parse_args()

    group_rows = _load_csv(Path(args.event_field_groups_csv))
    event_proto_rows = _load_csv(Path(args.event_field_proto_exciters_csv))
    notechain_proto_rows = _load_csv(Path(args.notechain_proto_exciters_csv))

    event_proto_by_id = {str(row.get("proto_exciter_id", "")).strip(): row for row in event_proto_rows}
    existing_notechain_ids = {str(row.get("proto_exciter_id", "")).strip() for row in notechain_proto_rows}

    rescue_rows: list[dict[str, Any]] = []
    rescue_ids: set[str] = set()
    selected_group_count = 0

    for group in group_rows:
        if _safe_int(group.get("onset_span_frames"), 999999) > int(args.max_onset_span_frames):
            continue
        if _safe_float(group.get("onset_compactness"), 0.0) < float(args.min_onset_compactness):
            continue
        if _safe_float(group.get("attack_ratio"), 0.0) < float(args.min_attack_ratio):
            continue
        if _safe_float(group.get("attack_compactness"), 0.0) < float(args.min_attack_compactness):
            continue

        dominant_note = _normalize_note(group.get("dominant_note_token", ""))
        dominant_pc = _pitch_class(dominant_note)
        try:
            member_proto_ids = json.loads(str(group.get("member_proto_ids_json", "[]")) or "[]")
        except Exception:
            member_proto_ids = []

        candidates: list[dict[str, Any]] = []
        for proto_id in member_proto_ids:
            proto = event_proto_by_id.get(str(proto_id).strip())
            if proto is None:
                continue
            if str(proto.get("proto_exciter_id", "")).strip() in existing_notechain_ids:
                continue
            if _safe_float(proto.get("exciter_confidence"), 0.0) < float(args.min_exciter_confidence):
                continue
            proto_note = _normalize_note(proto.get("coarse_note", ""))
            if dominant_pc and _pitch_class(proto_note) != dominant_pc:
                continue
            candidates.append(proto)

        if not candidates:
            continue

        candidates.sort(
            key=lambda row: (
                _safe_float(row.get("exciter_confidence"), 0.0),
                -_safe_int(row.get("duration_frames"), 0),
                -_safe_float(row.get("peak_seed_score"), 0.0),
            ),
            reverse=True,
        )
        selected = dict(candidates[0])
        selected["rescue_source"] = "event_field_group"
        selected["rescue_group_id"] = str(group.get("event_group_id", "")).strip()
        selected["rescue_group_dominant_note"] = dominant_note
        rescue_id = str(selected.get("proto_exciter_id", "")).strip()
        if rescue_id and rescue_id not in rescue_ids:
            rescue_rows.append(selected)
            rescue_ids.add(rescue_id)
            selected_group_count += 1

    combined_rows = list(notechain_proto_rows) + rescue_rows
    combined_rows.sort(key=lambda row: (_safe_int(row.get("start_frame"), 0), _safe_int(row.get("end_frame"), 0), str(row.get("coarse_note", ""))))

    out_rescue = Path(args.out_rescue_proto_exciters_csv)
    out_combined = Path(args.out_combined_notechain_proto_exciters_csv)
    out_summary = Path(args.out_summary_txt)
    out_meta = Path(args.out_meta_json)
    out_rescue.parent.mkdir(parents=True, exist_ok=True)

    fieldnames_set: set[str] = set()
    for row in combined_rows:
        fieldnames_set.update(row.keys())
    if not fieldnames_set and notechain_proto_rows:
        fieldnames_set.update(notechain_proto_rows[0].keys())
    fieldnames = sorted(fieldnames_set)
    with out_rescue.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rescue_rows)

    with out_combined.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(combined_rows)

    summary_lines = [
        "EVENT FIELD NOTECHAIN RESCUE SELECTOR",
        "=" * 72,
        f"notechain_proto_input     : {len(notechain_proto_rows)}",
        f"event_field_group_input   : {len(group_rows)}",
        f"event_field_proto_input   : {len(event_proto_rows)}",
        f"rescue_proto_selected     : {len(rescue_rows)}",
        f"selected_group_count      : {selected_group_count}",
        f"combined_notechain_proto  : {len(combined_rows)}",
    ]
    out_summary.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    meta = {
        "stage": "event_field_notechain_rescue_selector",
        "inputs": {
            "event_field_groups_csv": args.event_field_groups_csv,
            "event_field_proto_exciters_csv": args.event_field_proto_exciters_csv,
            "notechain_proto_exciters_csv": args.notechain_proto_exciters_csv,
        },
        "parameters": {
            "max_onset_span_frames": int(args.max_onset_span_frames),
            "min_onset_compactness": float(args.min_onset_compactness),
            "min_attack_ratio": float(args.min_attack_ratio),
            "min_attack_compactness": float(args.min_attack_compactness),
            "min_exciter_confidence": float(args.min_exciter_confidence),
        },
        "result": {
            "rescue_proto_selected": len(rescue_rows),
            "selected_group_count": selected_group_count,
            "combined_notechain_proto": len(combined_rows),
        },
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
