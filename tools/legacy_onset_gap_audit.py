# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


ALPHABET12 = "123456789ABC"


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


def _note_pc(note: str) -> str:
    s = str(note or "").strip()
    if "." not in s:
        return ""
    return s.split(".", 1)[1].split("'", 1)[0]


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
        notes = [str(r.get("candidate_note", "")).strip() for r in bucket if str(r.get("candidate_note", "")).strip()]
        pcs = [_note_pc(n) for n in notes if _note_pc(n)]
        roles = Counter(str(r.get("acoustic_event_role", "")).strip() for r in bucket)
        frames = [_safe_int(r.get("birth_frame"), 0) for r in bucket]
        out.append(
            {
                "event_group_id": idx,
                "anchor_frame": min(frames),
                "end_anchor_frame": max(frames),
                "member_count": len(bucket),
                "notes": sorted(set(notes)),
                "pitchclasses": sorted(set(pcs)),
                "role_counts": roles,
                "members": bucket,
            }
        )
    return out


def _build_midi_groups(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_group[str(r.get("onset_group", ""))].append(r)
    out: List[Dict[str, Any]] = []
    for gid, bucket in sorted(by_group.items(), key=lambda kv: _safe_int(kv[0], 0)):
        notes = [str(r.get("note_token", "")).strip() for r in bucket if str(r.get("note_token", "")).strip()]
        pcs = [_note_pc(n) for n in notes if _note_pc(n)]
        out.append(
            {
                "midi_onset_group": gid,
                "anchor_frame": min(_safe_int(r.get("start_frame60"), 0) for r in bucket),
                "member_count": len(bucket),
                "notes": sorted(set(notes)),
                "pitchclasses": sorted(set(pcs)),
            }
        )
    return out


def _best_midi_match(event_group: Dict[str, Any], midi_groups: List[Dict[str, Any]], match_window: int) -> Dict[str, Any]:
    enotes: Set[str] = set(event_group["notes"])
    epcs: Set[str] = set(event_group["pitchclasses"])
    ef = int(event_group["anchor_frame"])
    local = [mg for mg in midi_groups if abs(ef - int(mg["anchor_frame"])) <= match_window]
    candidates = local if local else midi_groups
    best = None
    best_key = None
    for mg in candidates:
        mf = int(mg["anchor_frame"])
        mnotes: Set[str] = set(mg["notes"])
        mpcs: Set[str] = set(mg["pitchclasses"])
        exact = len(enotes & mnotes)
        pc = len(epcs & mpcs)
        time_dist = abs(ef - mf)
        # prefer local time first, then note overlap
        key = (-time_dist, exact, pc, -abs(event_group["member_count"] - mg["member_count"]))
        if best is None or key > best_key:
            best = mg
            best_key = key
    assert best is not None
    return {
        "midi_onset_group": best["midi_onset_group"],
        "midi_anchor_frame": best["anchor_frame"],
        "midi_member_count": best["member_count"],
        "exact_note_overlap": best_key[0],
        "pitchclass_overlap": best_key[1],
        "time_distance": abs(ef - int(best["anchor_frame"])),
        "midi_notes": best["notes"],
        "midi_pitchclasses": best["pitchclasses"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit remaining onset-group surplus after legacy live-piano cleanup.")
    ap.add_argument("--events-csv", required=True)
    ap.add_argument("--midi-events-csv", required=True)
    ap.add_argument("--out-audit-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--onset-window", type=int, default=3)
    ap.add_argument("--match-window", type=int, default=18)
    args = ap.parse_args()

    event_rows = _load_csv(Path(args.events_csv))
    midi_rows = _load_csv(Path(args.midi_events_csv))
    event_groups = _build_event_groups(event_rows, args.onset_window)
    midi_groups = _build_midi_groups(midi_rows)

    assignments: List[Dict[str, Any]] = []
    by_midi: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for eg in event_groups:
        match = _best_midi_match(eg, midi_groups, args.match_window)
        row = {
            "event_group_id": eg["event_group_id"],
            "event_anchor_frame": eg["anchor_frame"],
            "event_end_anchor_frame": eg["end_anchor_frame"],
            "event_member_count": eg["member_count"],
            "event_notes": " ".join(eg["notes"]),
            "event_pitchclasses": " ".join(eg["pitchclasses"]),
            "event_role_counts": " | ".join(f"{k}:{v}" for k, v in eg["role_counts"].most_common()),
            **match,
        }
        assignments.append(row)
        by_midi[str(match["midi_onset_group"])].append(row)

    # Mark primary and extra groups for MIDI groups with multiple event-group assignments
    extra_count = 0
    class_counts = Counter()
    for gid, bucket in by_midi.items():
        bucket.sort(key=lambda r: (int(r["time_distance"]), -(int(r["exact_note_overlap"])), -(int(r["pitchclass_overlap"])), abs(int(r["event_member_count"]) - int(r["midi_member_count"]))))
        for idx, r in enumerate(bucket):
            if idx == 0:
                r["onset_gap_role"] = "PRIMARY_MATCH"
            else:
                extra_count += 1
                role_text = str(r.get("event_role_counts", ""))
                if "BODY_RETURN_CLUSTER_REPRESENTATIVE" in role_text:
                    cls = "EXTRA_BODY_RETURN_GROUP"
                elif "INTERNAL_WAVE_EVENT" in role_text:
                    cls = "EXTRA_INTERNAL_WAVE_GROUP"
                elif "SHORT_RESONANCE_TRACE" in role_text or "HALL_OR_FIELD_TRACE" in role_text:
                    cls = "EXTRA_TRACE_GROUP"
                else:
                    cls = "EXTRA_MIXED_GROUP"
                r["onset_gap_role"] = cls
                class_counts[cls] += 1

    assignments.sort(key=lambda r: (_safe_int(r["event_anchor_frame"]), _safe_int(r["event_group_id"])))
    _write_csv(Path(args.out_audit_csv), assignments, assignments[0].keys())

    summary_lines = [
        "LEGACY ONSET GAP AUDIT",
        "=" * 72,
        f"events_csv              : {args.events_csv}",
        f"midi_events_csv         : {args.midi_events_csv}",
        f"event_groups            : {len(event_groups)}",
        f"midi_onset_groups       : {len(midi_groups)}",
        f"onset_group_gap         : {len(event_groups) - len(midi_groups)}",
        f"match_window            : {args.match_window}",
        f"extra_groups_after_match: {extra_count}",
        "",
        "extra_group_class_counts:",
    ]
    for k, v in sorted(class_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        summary_lines.append(f"  {k}: {v}")

    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "inputs": {
                    "events_csv": args.events_csv,
                    "midi_events_csv": args.midi_events_csv,
                },
                "result": {
                    "event_groups": len(event_groups),
                    "midi_onset_groups": len(midi_groups),
                    "onset_group_gap": len(event_groups) - len(midi_groups),
                    "match_window": args.match_window,
                    "extra_groups_after_match": extra_count,
                    "extra_group_class_counts": dict(sorted(class_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
