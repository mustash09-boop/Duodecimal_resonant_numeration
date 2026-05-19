# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        w.writerows(rows)


def _build_event_groups(rows: List[Dict[str, Any]], window: int) -> List[Dict[str, Any]]:
    ordered = sorted(rows, key=lambda r: (_safe_int(r.get("birth_frame"), 0), str(r.get("candidate_note", ""))))
    groups: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    anchor = None
    for r in ordered:
        birth = _safe_int(r.get("birth_frame"), 0)
        if anchor is None or birth - anchor > window:
            if current:
                groups.append(current)
            current = [r]
            anchor = birth
        else:
            current.append(r)
    if current:
        groups.append(current)

    out: List[Dict[str, Any]] = []
    for idx, bucket in enumerate(groups, start=1):
        roles = Counter(str(r.get("acoustic_event_role", "")).strip() for r in bucket)
        notes = sorted({str(r.get("candidate_note", "")).strip() for r in bucket if str(r.get("candidate_note", "")).strip()})
        out.append(
            {
                "event_group_id": idx,
                "anchor_frame": min(_safe_int(r.get("birth_frame"), 0) for r in bucket),
                "end_anchor_frame": max(_safe_int(r.get("birth_frame"), 0) for r in bucket),
                "member_count": len(bucket),
                "notes": notes,
                "role_counts": roles,
            }
        )
    return out


def _build_midi_groups(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[str(r.get("onset_group", ""))].append(r)
    out: List[Dict[str, Any]] = []
    for gid, bucket in sorted(by.items(), key=lambda kv: _safe_int(kv[0], 0)):
        out.append(
            {
                "midi_onset_group": gid,
                "anchor_frame": min(_safe_int(r.get("start_frame60"), 0) for r in bucket),
                "member_count": len(bucket),
                "notes": sorted({str(r.get("note_token", "")).strip() for r in bucket if str(r.get("note_token", "")).strip()}),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Interval-based onset surplus audit.")
    ap.add_argument("--events-csv", required=True)
    ap.add_argument("--midi-events-csv", required=True)
    ap.add_argument("--out-audit-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--onset-window", type=int, default=3)
    args = ap.parse_args()

    event_rows = _load_csv(Path(args.events_csv))
    midi_rows = _load_csv(Path(args.midi_events_csv))
    event_groups = _build_event_groups(event_rows, args.onset_window)
    midi_groups = _build_midi_groups(midi_rows)

    midi_anchors = [int(m["anchor_frame"]) for m in midi_groups]
    intervals = []
    for i, mg in enumerate(midi_groups):
        left = -10**9 if i == 0 else (midi_anchors[i - 1] + midi_anchors[i]) / 2.0
        right = 10**9 if i == len(midi_groups) - 1 else (midi_anchors[i] + midi_anchors[i + 1]) / 2.0
        intervals.append((left, right))

    by_midi: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    audit_rows: List[Dict[str, Any]] = []
    for eg in event_groups:
        ef = int(eg["anchor_frame"])
        chosen = None
        for mg, (left, right) in zip(midi_groups, intervals):
            if ef >= left and ef < right:
                chosen = mg
                break
        if chosen is None:
            chosen = midi_groups[-1]
        row = {
            "event_group_id": eg["event_group_id"],
            "event_anchor_frame": eg["anchor_frame"],
            "event_member_count": eg["member_count"],
            "event_notes": " ".join(eg["notes"]),
            "event_role_counts": " | ".join(f"{k}:{v}" for k, v in eg["role_counts"].most_common()),
            "midi_onset_group": chosen["midi_onset_group"],
            "midi_anchor_frame": chosen["anchor_frame"],
            "midi_notes": " ".join(chosen["notes"]),
            "time_distance": abs(ef - int(chosen["anchor_frame"])),
        }
        by_midi[str(chosen["midi_onset_group"])].append(row)
        audit_rows.append(row)

    extra_counts = Counter()
    extra_total = 0
    for gid, bucket in by_midi.items():
        bucket.sort(key=lambda r: (int(r["time_distance"]), abs(int(r["event_member_count"]) - 1)))
        for idx, r in enumerate(bucket):
            if idx == 0:
                r["interval_role"] = "PRIMARY_INTERVAL_MATCH"
            else:
                extra_total += 1
                roles = str(r["event_role_counts"])
                if "BODY_RETURN_CLUSTER_REPRESENTATIVE" in roles:
                    cls = "EXTRA_BODY_RETURN_INTERVAL"
                elif "INTERNAL_WAVE_EVENT" in roles:
                    cls = "EXTRA_INTERNAL_WAVE_INTERVAL"
                elif "SHORT_RESONANCE_TRACE" in roles or "HALL_OR_FIELD_TRACE" in roles:
                    cls = "EXTRA_TRACE_INTERVAL"
                else:
                    cls = "EXTRA_MIXED_INTERVAL"
                r["interval_role"] = cls
                extra_counts[cls] += 1

    audit_rows.sort(key=lambda r: (_safe_int(r["event_anchor_frame"]), _safe_int(r["event_group_id"])))
    _write_csv(Path(args.out_audit_csv), audit_rows, audit_rows[0].keys())

    summary = [
        "LEGACY ONSET INTERVAL AUDIT",
        "=" * 72,
        f"event_groups             : {len(event_groups)}",
        f"midi_onset_groups        : {len(midi_groups)}",
        f"onset_group_gap          : {len(event_groups) - len(midi_groups)}",
        f"extra_groups_by_interval : {extra_total}",
        "",
        "extra_interval_class_counts:",
    ]
    for k, v in sorted(extra_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        summary.append(f"  {k}: {v}")

    Path(args.out_summary_txt).write_text("\n".join(summary) + "\n", encoding="utf-8")
    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "result": {
                    "event_groups": len(event_groups),
                    "midi_onset_groups": len(midi_groups),
                    "onset_group_gap": len(event_groups) - len(midi_groups),
                    "extra_groups_by_interval": extra_total,
                    "extra_interval_class_counts": dict(sorted(extra_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
                }
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
