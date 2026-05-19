# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
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


def _cluster_by_birth(rows: List[Dict[str, Any]], window: int) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    ordered = sorted(rows, key=lambda r: (_safe_int(r.get("birth_frame"), 0), str(r.get("candidate_note", ""))))
    group_sizes: Dict[str, int] = {}
    current: List[Dict[str, Any]] = []
    anchor = None
    gid = 0

    def flush(bucket: List[Dict[str, Any]], group_id: int) -> None:
        size = len(bucket)
        for r in bucket:
            group_sizes[str(r.get("merged_event_id", ""))] = size
            r["_birth_group_id"] = group_id

    for r in ordered:
        birth = _safe_int(r.get("birth_frame"), 0)
        if anchor is None:
            gid += 1
            anchor = birth
            current = [r]
            continue
        if birth - anchor <= window:
            current.append(r)
        else:
            flush(current, gid)
            gid += 1
            anchor = birth
            current = [r]
    if current:
        flush(current, gid)
    return ordered, group_sizes


def _same_note_neighbors(rows: List[Dict[str, Any]], near_gap: int) -> Dict[str, Dict[str, Any]]:
    by_note: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        by_note[str(r.get("candidate_note", ""))].append(r)
    for note, items in by_note.items():
        items = sorted(items, key=lambda r: (_safe_int(r.get("birth_frame"), 0), _safe_int(r.get("end_frame"), 0)))
        for i, r in enumerate(items):
            rid = str(r.get("merged_event_id", ""))
            rb = _safe_int(r.get("birth_frame"), 0)
            re = _safe_int(r.get("end_frame"), 0)
            min_gap = 999999
            overlap = 0
            prev_id = ""
            next_id = ""
            for j in range(i - 1, -1, -1):
                other = items[j]
                ob = _safe_int(other.get("birth_frame"), 0)
                oe = _safe_int(other.get("end_frame"), 0)
                gap = max(ob - re, rb - oe, 0)
                inter = max(0, min(re, oe) - max(rb, ob) + 1)
                if inter > 0 or gap <= near_gap:
                    prev_id = str(other.get("merged_event_id", ""))
                    min_gap = min(min_gap, gap)
                    overlap = max(overlap, inter)
                    break
                if rb - oe > near_gap:
                    break
            for j in range(i + 1, len(items)):
                other = items[j]
                ob = _safe_int(other.get("birth_frame"), 0)
                oe = _safe_int(other.get("end_frame"), 0)
                gap = max(ob - re, rb - oe, 0)
                inter = max(0, min(re, oe) - max(rb, ob) + 1)
                if inter > 0 or gap <= near_gap:
                    next_id = str(other.get("merged_event_id", ""))
                    min_gap = min(min_gap, gap)
                    overlap = max(overlap, inter)
                    break
                if ob - re > near_gap:
                    break
            out[rid] = {
                "same_note_min_gap": -1 if min_gap == 999999 else min_gap,
                "same_note_overlap_frames": overlap,
                "same_note_prev_id": prev_id,
                "same_note_next_id": next_id,
            }
    return out


def _classify_event(r: Dict[str, Any], group_size: int, same_note_info: Dict[str, Any]) -> str:
    refined = str(r.get("refined_lifecycle_kind", ""))
    duration = _safe_int(r.get("duration_frames"), 0)
    frame_count = _safe_int(r.get("frame_count"), 0)
    re_count = _safe_int(r.get("re_excitation_count"), 0)
    overlap = _safe_int(same_note_info.get("same_note_overlap_frames"), 0)
    near_gap = _safe_int(same_note_info.get("same_note_min_gap"), -1)
    energy_span = _safe_float(r.get("relative_energy_span"), 0.0)

    if refined in {"resonance_trace_lifecycle", "fragmented_lifecycle"}:
        return "TRACE_OR_FRAGMENT"
    if overlap > 0 or (0 <= near_gap <= 2):
        return "SAME_NOTE_NEAR_REBIRTH"
    if refined == "weak_or_short_lifecycle" and (duration <= 4 or frame_count <= 4):
        return "VERY_SHORT_EVENT"
    if refined == "weak_or_short_lifecycle" and group_size >= 2:
        return "WEAK_CLUSTER_MEMBER"
    if refined == "coherent_sustained_lifecycle_with_internal_waves" and re_count >= 2 and energy_span >= 0.25:
        return "INTERNAL_WAVE_HEAVY"
    if group_size >= 3:
        return "DENSE_ONSET_CLUSTER"
    return "STABLE_BACKBONE"


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit residual fragmentation types in legacy-cleaned events.")
    ap.add_argument("--events-csv", required=True)
    ap.add_argument("--midi-meta-json", required=True)
    ap.add_argument("--out-audit-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--onset-window", type=int, default=3)
    ap.add_argument("--same-note-near-gap", type=int, default=6)
    args = ap.parse_args()

    rows = _load_csv(Path(args.events_csv))
    midi_meta = json.loads(Path(args.midi_meta_json).read_text(encoding="utf-8"))

    ordered, group_sizes = _cluster_by_birth(rows, args.onset_window)
    same_note = _same_note_neighbors(ordered, args.same_note_near_gap)

    out_rows: List[Dict[str, Any]] = []
    class_counts: Dict[str, int] = defaultdict(int)
    group_hist: Dict[int, int] = defaultdict(int)

    for r in ordered:
        rid = str(r.get("merged_event_id", ""))
        size = int(group_sizes.get(rid, 1))
        info = same_note.get(rid, {"same_note_min_gap": -1, "same_note_overlap_frames": 0, "same_note_prev_id": "", "same_note_next_id": ""})
        cls = _classify_event(r, size, info)
        class_counts[cls] += 1
        group_hist[size] += 1
        rr = dict(r)
        rr["birth_group_size"] = size
        rr["same_note_min_gap"] = info["same_note_min_gap"]
        rr["same_note_overlap_frames"] = info["same_note_overlap_frames"]
        rr["same_note_prev_id"] = info["same_note_prev_id"]
        rr["same_note_next_id"] = info["same_note_next_id"]
        rr["residual_fragmentation_class"] = cls
        out_rows.append(rr)

    out_rows.sort(
        key=lambda r: (
            _safe_int(r.get("birth_frame"), 0),
            {"TRACE_OR_FRAGMENT": 0, "SAME_NOTE_NEAR_REBIRTH": 1, "VERY_SHORT_EVENT": 2, "WEAK_CLUSTER_MEMBER": 3, "INTERNAL_WAVE_HEAVY": 4, "DENSE_ONSET_CLUSTER": 5, "STABLE_BACKBONE": 6}.get(str(r.get("residual_fragmentation_class", "")), 9),
            str(r.get("candidate_note", "")),
        )
    )

    _write_csv(Path(args.out_audit_csv), out_rows, out_rows[0].keys())

    summary_lines = [
        "LEGACY EVENT RESIDUAL FRAGMENTATION AUDIT",
        "=" * 72,
        f"events_csv             : {args.events_csv}",
        f"input_events           : {len(rows)}",
        f"target_event_count     : {midi_meta.get('event_count', 0)}",
        f"target_onset_groups    : {midi_meta.get('unique_onset_groups', 0)}",
        f"event_gap_to_target    : {len(rows) - int(midi_meta.get('event_count', 0))}",
        "",
        "residual_fragmentation_class_counts:",
    ]
    for k, v in sorted(class_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        summary_lines.append(f"  {k}: {v}")
    summary_lines.append("")
    summary_lines.append("birth_group_size_histogram:")
    for k, v in sorted(group_hist.items()):
        summary_lines.append(f"  {k}: {v}")

    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "inputs": {
                    "events_csv": args.events_csv,
                    "midi_meta_json": args.midi_meta_json,
                },
                "parameters": {
                    "onset_window": args.onset_window,
                    "same_note_near_gap": args.same_note_near_gap,
                },
                "result": {
                    "input_events": len(rows),
                    "target_event_count": midi_meta.get("event_count", 0),
                    "target_onset_groups": midi_meta.get("unique_onset_groups", 0),
                    "event_gap_to_target": len(rows) - int(midi_meta.get("event_count", 0)),
                    "residual_fragmentation_class_counts": dict(sorted(class_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
                    "birth_group_size_histogram": dict(sorted(group_hist.items())),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
