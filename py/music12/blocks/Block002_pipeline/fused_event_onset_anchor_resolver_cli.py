from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Assign onset-anchor groups to fused musical events without merging the events themselves."
    )
    ap.add_argument("--fused-events-csv", required=True)
    ap.add_argument("--out-anchored-events-csv", required=True)
    ap.add_argument("--out-onset-groups-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--onset-anchor-window-frames", type=int, default=3)
    args = ap.parse_args()

    rows = _load_csv(Path(args.fused_events_csv))
    musical_kinds = {"notechain_backbone", "event_field_only"}
    musical_rows = [row for row in rows if str(row.get("event_kind", "")).strip() in musical_kinds]
    residue_rows = [row for row in rows if str(row.get("event_kind", "")).strip() not in musical_kinds]
    musical_rows.sort(key=lambda r: (_safe_int(r.get("start_frame"), 0), _safe_int(r.get("end_frame"), 0), _safe_int(r.get("fused_event_id"), 0)))

    groups: list[dict[str, Any]] = []
    anchored_rows: list[dict[str, Any]] = []
    current_group = None
    next_group_id = 1

    for row in musical_rows:
        start_frame = _safe_int(row.get("start_frame"), 0)
        if current_group is None or start_frame - _safe_int(current_group.get("anchor_frame"), 0) > int(args.onset_anchor_window_frames):
            current_group = {
                "onset_group_id": next_group_id,
                "anchor_frame": start_frame,
                "start_min_frame": start_frame,
                "start_max_frame": start_frame,
                "member_count": 0,
                "kind_counter": Counter(),
                "support_counter": Counter(),
                "main_note_tokens": [],
            }
            groups.append(current_group)
            next_group_id += 1

        current_group["start_min_frame"] = min(_safe_int(current_group.get("start_min_frame"), start_frame), start_frame)
        current_group["start_max_frame"] = max(_safe_int(current_group.get("start_max_frame"), start_frame), start_frame)
        current_group["member_count"] = _safe_int(current_group.get("member_count"), 0) + 1
        current_group["kind_counter"][str(row.get("event_kind", "")).strip()] += 1
        current_group["support_counter"][str(row.get("field_support_kind", "")).strip()] += 1
        current_group["main_note_tokens"].append(str(row.get("main_note_token", "")).strip())

        anchored = dict(row)
        anchored["onset_group_id"] = current_group["onset_group_id"]
        anchored["onset_anchor_frame"] = current_group["anchor_frame"]
        anchored_rows.append(anchored)

    for row in residue_rows:
        anchored = dict(row)
        anchored["onset_group_id"] = ""
        anchored["onset_anchor_frame"] = ""
        anchored_rows.append(anchored)

    anchored_rows.sort(key=lambda r: (_safe_int(r.get("start_frame"), 0), _safe_int(r.get("end_frame"), 0), _safe_int(r.get("fused_event_id"), 0)))

    onset_groups: list[dict[str, Any]] = []
    for group in groups:
        onset_groups.append(
            {
                "onset_group_id": _safe_int(group.get("onset_group_id"), 0),
                "anchor_frame": _safe_int(group.get("anchor_frame"), 0),
                "start_min_frame": _safe_int(group.get("start_min_frame"), 0),
                "start_max_frame": _safe_int(group.get("start_max_frame"), 0),
                "start_span_frames": _safe_int(group.get("start_max_frame"), 0) - _safe_int(group.get("start_min_frame"), 0),
                "member_count": _safe_int(group.get("member_count"), 0),
                "kind_counts_json": json.dumps(dict(group["kind_counter"]), ensure_ascii=False, sort_keys=True),
                "support_counts_json": json.dumps(dict(group["support_counter"]), ensure_ascii=False, sort_keys=True),
                "main_note_tokens_json": json.dumps(group["main_note_tokens"], ensure_ascii=False),
            }
        )

    out_events = Path(args.out_anchored_events_csv)
    out_groups = Path(args.out_onset_groups_csv)
    out_summary = Path(args.out_summary_txt)
    out_meta = Path(args.out_meta_json)
    out_events.parent.mkdir(parents=True, exist_ok=True)

    event_fields = list(anchored_rows[0].keys()) if anchored_rows else []
    with out_events.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=event_fields)
        w.writeheader()
        w.writerows(anchored_rows)

    group_fields = [
        "onset_group_id",
        "anchor_frame",
        "start_min_frame",
        "start_max_frame",
        "start_span_frames",
        "member_count",
        "kind_counts_json",
        "support_counts_json",
        "main_note_tokens_json",
    ]
    with out_groups.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=group_fields)
        w.writeheader()
        w.writerows(onset_groups)

    summary_lines = [
        "FUSED EVENT ONSET ANCHOR RESOLVER",
        "=" * 72,
        f"musical_events          : {len(musical_rows)}",
        f"ambient_residue_events  : {len(residue_rows)}",
        f"resolved_onset_groups   : {len(onset_groups)}",
        f"onset_anchor_window     : {int(args.onset_anchor_window_frames)}",
        f"mean_members_per_group  : {sum(_safe_int(row.get('member_count'), 0) for row in onset_groups) / max(len(onset_groups), 1):.6f}",
        f"mean_start_span_frames  : {sum(_safe_int(row.get('start_span_frames'), 0) for row in onset_groups) / max(len(onset_groups), 1):.6f}",
    ]
    out_summary.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    meta = {
        "stage": "fused_event_onset_anchor_resolver",
        "inputs": {
            "fused_events_csv": args.fused_events_csv,
        },
        "parameters": {
            "onset_anchor_window_frames": int(args.onset_anchor_window_frames),
        },
        "result": {
            "musical_events": len(musical_rows),
            "ambient_residue_events": len(residue_rows),
            "resolved_onset_groups": len(onset_groups),
        },
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
