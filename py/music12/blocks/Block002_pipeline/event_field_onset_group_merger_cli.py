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


def _mergeable(
    group: dict[str, Any],
    row: dict[str, Any],
    max_gap_frames: int,
    onset_cluster_window_frames: int,
    max_total_onset_span_frames: int,
) -> bool:
    row_start = _safe_int(row.get("start_frame"), 0)
    if row_start > _safe_int(group.get("end_frame"), 0) + max_gap_frames:
        return False
    if row_start > _safe_int(group.get("onset_start_max_frame"), 0) + onset_cluster_window_frames:
        return False
    if row_start > _safe_int(group.get("onset_start_min_frame"), 0) + max_total_onset_span_frames:
        return False
    row_dom = _normalize_note(row.get("dominant_note_token", ""))
    row_pc = _pitch_class(row_dom)
    if row_dom and row_dom == str(group.get("dominant_note_token", "")):
        return True
    if row_pc and row_pc in set(group.get("pitch_classes", set())):
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge overlapping event-field entities into onset groups so the event branch can be compared to MIDI-scale event counts."
    )
    ap.add_argument("--event-field-entities-csv", required=True)
    ap.add_argument("--out-merged-entities-csv", required=True)
    ap.add_argument("--out-merged-frames-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--max-gap-frames", type=int, default=2)
    ap.add_argument("--onset-cluster-window-frames", type=int, default=5)
    ap.add_argument("--max-total-onset-span-frames", type=int, default=10)
    args = ap.parse_args()

    entity_rows = _load_csv(Path(args.event_field_entities_csv))
    entity_rows.sort(key=lambda r: (_safe_int(r.get("start_frame"), 0), _safe_int(r.get("end_frame"), 0), _normalize_note(r.get("dominant_note_token", ""))))

    groups: list[dict[str, Any]] = []
    next_group_id = 1

    for row in entity_rows:
        row_start = _safe_int(row.get("start_frame"), 0)
        row_end = _safe_int(row.get("end_frame"), row_start)
        row_dom = _normalize_note(row.get("dominant_note_token", ""))
        row_pc = _pitch_class(row_dom)

        target_group = None
        for group in reversed(groups):
            if _mergeable(
                group,
                row,
                int(args.max_gap_frames),
                int(args.onset_cluster_window_frames),
                int(args.max_total_onset_span_frames),
            ):
                target_group = group
                break
            if row_start > _safe_int(group.get("end_frame"), 0) + int(args.max_gap_frames):
                break

        if target_group is None:
            target_group = {
                "event_group_id": next_group_id,
                "start_frame": row_start,
                "end_frame": row_end,
                "onset_start_min_frame": row_start,
                "onset_start_max_frame": row_start,
                "dominant_note_token": row_dom,
                "pitch_classes": set([row_pc]) if row_pc else set(),
                "member_proto_ids": [],
                "dominant_note_counter": Counter(),
                "phase_counter": Counter(),
                "entity_count": 0,
                "confidence_sum": 0.0,
                "field_strength_sum": 0.0,
            }
            groups.append(target_group)
            next_group_id += 1

        target_group["start_frame"] = min(_safe_int(target_group.get("start_frame"), row_start), row_start)
        target_group["end_frame"] = max(_safe_int(target_group.get("end_frame"), row_end), row_end)
        target_group["onset_start_min_frame"] = min(_safe_int(target_group.get("onset_start_min_frame"), row_start), row_start)
        target_group["onset_start_max_frame"] = max(_safe_int(target_group.get("onset_start_max_frame"), row_start), row_start)
        target_group["entity_count"] = _safe_int(target_group.get("entity_count"), 0) + 1
        target_group["confidence_sum"] = _safe_float(target_group.get("confidence_sum"), 0.0) + _safe_float(row.get("exciter_confidence"), 0.0)
        target_group["field_strength_sum"] = _safe_float(target_group.get("field_strength_sum"), 0.0) + _safe_float(row.get("mean_field_strength"), 0.0)
        target_group["member_proto_ids"].append(str(row.get("proto_exciter_id", "")))
        if row_dom:
            target_group["dominant_note_counter"][row_dom] += 1
        if row_pc:
            target_group["pitch_classes"].add(row_pc)
        phase_counts = json.loads(str(row.get("phase_counts_json", "{}")) or "{}")
        for key, value in phase_counts.items():
            target_group["phase_counter"][str(key)] += _safe_int(value, 0)

    merged_entities: list[dict[str, Any]] = []
    merged_frames: list[dict[str, Any]] = []

    for group in groups:
        dominant_note = group["dominant_note_counter"].most_common(1)[0][0] if group["dominant_note_counter"] else str(group.get("dominant_note_token", ""))
        start_frame = _safe_int(group.get("start_frame"), 0)
        end_frame = _safe_int(group.get("end_frame"), start_frame)
        onset_start_min_frame = _safe_int(group.get("onset_start_min_frame"), start_frame)
        onset_start_max_frame = _safe_int(group.get("onset_start_max_frame"), start_frame)
        onset_span_frames = onset_start_max_frame - onset_start_min_frame
        entity_count = _safe_int(group.get("entity_count"), 0)
        mean_conf = _safe_float(group.get("confidence_sum"), 0.0) / max(entity_count, 1)
        mean_strength = _safe_float(group.get("field_strength_sum"), 0.0) / max(entity_count, 1)
        onset_compactness = entity_count / max(onset_span_frames + 1, 1)
        attack_frame_count = _safe_int(group["phase_counter"].get("ATTACK", 0), 0)
        total_phase_frame_count = sum(_safe_int(v, 0) for v in group["phase_counter"].values())
        attack_ratio = attack_frame_count / max(total_phase_frame_count, 1)
        attack_compactness = attack_frame_count / max(onset_span_frames + 1, 1)
        merged_entities.append(
            {
                "event_group_id": _safe_int(group.get("event_group_id"), 0),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "onset_start_min_frame": onset_start_min_frame,
                "onset_start_max_frame": onset_start_max_frame,
                "onset_span_frames": onset_span_frames,
                "onset_compactness": f"{onset_compactness:.9f}",
                "attack_frame_count": attack_frame_count,
                "total_phase_frame_count": total_phase_frame_count,
                "attack_ratio": f"{attack_ratio:.9f}",
                "attack_compactness": f"{attack_compactness:.9f}",
                "frame_count": end_frame - start_frame + 1,
                "entity_count": entity_count,
                "dominant_note_token": dominant_note,
                "pitch_classes_json": json.dumps(sorted(group["pitch_classes"]), ensure_ascii=False),
                "mean_exciter_confidence": f"{mean_conf:.9f}",
                "mean_field_strength": f"{mean_strength:.9f}",
                "phase_counts_json": json.dumps(dict(group["phase_counter"]), ensure_ascii=False, sort_keys=True),
                "member_proto_ids_json": json.dumps(group["member_proto_ids"], ensure_ascii=False),
            }
        )
        for frame_index in range(start_frame, end_frame + 1):
            merged_frames.append(
                {
                    "event_group_id": _safe_int(group.get("event_group_id"), 0),
                    "frame_index": frame_index,
                    "dominant_note_token": dominant_note,
                }
            )

    merged_entities.sort(key=lambda r: (_safe_int(r.get("start_frame"), 0), _safe_int(r.get("event_group_id"), 0)))
    merged_frames.sort(key=lambda r: (_safe_int(r.get("frame_index"), 0), _safe_int(r.get("event_group_id"), 0)))

    out_entities = Path(args.out_merged_entities_csv)
    out_frames = Path(args.out_merged_frames_csv)
    out_summary = Path(args.out_summary_txt)
    out_meta = Path(args.out_meta_json)
    out_entities.parent.mkdir(parents=True, exist_ok=True)

    entity_fields = [
        "event_group_id",
        "start_frame",
        "end_frame",
        "onset_start_min_frame",
        "onset_start_max_frame",
        "onset_span_frames",
        "onset_compactness",
        "attack_frame_count",
        "total_phase_frame_count",
        "attack_ratio",
        "attack_compactness",
        "frame_count",
        "entity_count",
        "dominant_note_token",
        "pitch_classes_json",
        "mean_exciter_confidence",
        "mean_field_strength",
        "phase_counts_json",
        "member_proto_ids_json",
    ]
    with out_entities.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=entity_fields)
        w.writeheader()
        w.writerows(merged_entities)

    frame_fields = ["event_group_id", "frame_index", "dominant_note_token"]
    with out_frames.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=frame_fields)
        w.writeheader()
        w.writerows(merged_frames)

    summary_lines = [
        "EVENT FIELD ONSET GROUP MERGER",
        "=" * 72,
        f"input_entities        : {len(entity_rows)}",
        f"merged_groups         : {len(merged_entities)}",
        f"merged_frames         : {len(merged_frames)}",
        f"mean_entities_per_grp : {sum(_safe_int(r.get('entity_count'), 0) for r in merged_entities) / max(len(merged_entities), 1):.6f}",
        f"mean_onset_span       : {sum(_safe_int(r.get('onset_span_frames'), 0) for r in merged_entities) / max(len(merged_entities), 1):.6f}",
        f"mean_attack_ratio     : {sum(_safe_float(r.get('attack_ratio'), 0.0) for r in merged_entities) / max(len(merged_entities), 1):.6f}",
        f"mean_attack_compactness: {sum(_safe_float(r.get('attack_compactness'), 0.0) for r in merged_entities) / max(len(merged_entities), 1):.6f}",
    ]
    out_summary.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    meta = {
        "stage": "event_field_onset_group_merger",
        "inputs": {
            "event_field_entities_csv": args.event_field_entities_csv,
        },
        "parameters": {
            "max_gap_frames": int(args.max_gap_frames),
            "onset_cluster_window_frames": int(args.onset_cluster_window_frames),
            "max_total_onset_span_frames": int(args.max_total_onset_span_frames),
        },
        "result": {
            "input_entities": len(entity_rows),
            "merged_groups": len(merged_entities),
            "merged_frames": len(merged_frames),
        },
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
