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


def _json_list(value: Any) -> list[str]:
    try:
        raw = json.loads(str(value or "[]"))
        if isinstance(raw, list):
            return [str(x).strip() for x in raw if str(x).strip()]
    except Exception:
        return []
    return []


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


def _group_truth(midi_rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for row in midi_rows:
        gid = str(row.get("onset_group", "")).strip()
        note = _normalize_note(row.get("expected_note_token", row.get("note_token", "")))
        if gid and note:
            out[gid].append(note)
    return out


def _event_level_counts(
    truth_by_group: dict[str, list[str]],
    predicted_by_group: dict[str, list[str]],
    topk: int,
) -> Counter[str]:
    counter: Counter[str] = Counter()
    for gid, truth in truth_by_group.items():
        pred = predicted_by_group.get(gid, [])[:topk]
        pred_set = set(pred)
        pred_pc = {_pitch_class(x) for x in pred}
        for note in truth:
            if note in pred_set:
                counter["EXACT"] += 1
            elif _pitch_class(note) in pred_pc:
                counter["PITCHCLASS"] += 1
            elif pred:
                counter["WRONG"] += 1
            else:
                counter["EMPTY"] += 1
    return counter


def _group_level_counts(
    truth_by_group: dict[str, list[str]],
    predicted_by_group: dict[str, list[str]],
    topk: int,
) -> Counter[str]:
    counter: Counter[str] = Counter()
    for gid, truth in truth_by_group.items():
        pred = predicted_by_group.get(gid, [])[:topk]
        truth_set = set(truth)
        pred_set = set(pred)
        truth_pc = {_pitch_class(x) for x in truth}
        pred_pc = {_pitch_class(x) for x in pred}
        if truth_set == pred_set and truth_set:
            counter["EXACT_GROUP"] += 1
        elif truth_pc == pred_pc and truth_pc:
            counter["PITCHCLASS_GROUP"] += 1
        elif pred:
            counter["WRONG_GROUP"] += 1
        else:
            counter["EMPTY_GROUP"] += 1
    return counter


def _load_main_top1(path: Path) -> dict[str, str]:
    rows = _load_csv(path)
    out: dict[str, str] = {}
    for row in rows:
        gid = str(row.get("onset_group", "")).strip()
        rank = _safe_int(row.get("new_rank"), 0)
        note = _normalize_note(row.get("note_token", ""))
        if gid and rank == 1 and note:
            out[gid] = note
    return out


def _load_cloud(path: Path) -> dict[str, list[str]]:
    rows = _load_csv(path)
    if not rows:
        return {}

    sample = rows[0]
    out: dict[str, list[str]] = {}

    if "reranked_notes_json" in sample:
        for row in rows:
            gid = str(row.get("onset_group", "")).strip()
            notes = [_normalize_note(x) for x in _json_list(row.get("reranked_notes_json", ""))]
            out[gid] = [x for x in notes if x]
        return out

    if "new_rank" in sample and "note_token" in sample:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row.get("onset_group", "")).strip()].append(row)
        for gid, grows in grouped.items():
            grows.sort(key=lambda r: _safe_int(r.get("new_rank"), 0))
            out[gid] = [_normalize_note(r.get("note_token", "")) for r in grows if _normalize_note(r.get("note_token", ""))]
        return out

    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build a two-stage external result: the main note comes from the precise main-note layer, "
            "while the rest of the candidate cloud comes from a wider cloud source."
        )
    )
    ap.add_argument("--main-top1-csv", required=True)
    ap.add_argument("--cloud-source-csv", required=True)
    ap.add_argument("--midi-events-csv", required=True)
    ap.add_argument("--out-audit-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--cloud-keep", type=int, default=8)
    args = ap.parse_args()

    main_top1 = _load_main_top1(Path(args.main_top1_csv))
    cloud = _load_cloud(Path(args.cloud_source_csv))
    truth_by_group = _group_truth(_load_csv(Path(args.midi_events_csv)))

    audit_rows: list[dict[str, Any]] = []
    predicted_by_group: dict[str, list[str]] = {}
    top1_changed = 0

    for gid, cloud_notes in cloud.items():
        main_note = _normalize_note(main_top1.get(gid, ""))
        merged: list[str] = []
        if main_note:
            merged.append(main_note)
        for note in cloud_notes[: int(args.cloud_keep)]:
            n = _normalize_note(note)
            if n and n not in merged:
                merged.append(n)
        predicted_by_group[gid] = merged

        cloud_top = _normalize_note(cloud_notes[0]) if cloud_notes else ""
        if main_note and cloud_top and main_note != cloud_top:
            top1_changed += 1

        audit_rows.append(
            {
                "onset_group": gid,
                "main_note": main_note,
                "cloud_top_note": cloud_top,
                "top1_changed": int(main_note and cloud_top and main_note != cloud_top),
                "merged_notes_json": json.dumps(merged, ensure_ascii=False),
                "cloud_notes_json": json.dumps([_normalize_note(x) for x in cloud_notes[: int(args.cloud_keep)]], ensure_ascii=False),
            }
        )

    event_top1 = _event_level_counts(truth_by_group, predicted_by_group, 1)
    event_top3 = _event_level_counts(truth_by_group, predicted_by_group, 3)
    event_top5 = _event_level_counts(truth_by_group, predicted_by_group, 5)
    group_top3 = _group_level_counts(truth_by_group, predicted_by_group, 3)
    group_top5 = _group_level_counts(truth_by_group, predicted_by_group, 5)

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
        "TWO STAGE MAINNOTE CLOUD MERGER",
        "=" * 72,
        f"onset_groups                 : {len(predicted_by_group)}",
        f"cloud_keep                   : {args.cloud_keep}",
        f"groups_with_top1_change      : {top1_changed}",
        "",
        "EVENT TOP-1",
        "-" * 72,
    ]
    for key in sorted(event_top1):
        lines.append(f"{key:28s}: {event_top1[key]}")
    lines.extend(["", "EVENT TOP-3", "-" * 72])
    for key in sorted(event_top3):
        lines.append(f"{key:28s}: {event_top3[key]}")
    lines.extend(["", "EVENT TOP-5", "-" * 72])
    for key in sorted(event_top5):
        lines.append(f"{key:28s}: {event_top5[key]}")
    lines.extend(["", "GROUP TOP-3", "-" * 72])
    for key in sorted(group_top3):
        lines.append(f"{key:28s}: {group_top3[key]}")
    lines.extend(["", "GROUP TOP-5", "-" * 72])
    for key in sorted(group_top5):
        lines.append(f"{key:28s}: {group_top5[key]}")
    out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    meta = {
        "stage": "two_stage_mainnote_cloud_merger",
        "inputs": {
            "main_top1_csv": args.main_top1_csv,
            "cloud_source_csv": args.cloud_source_csv,
            "midi_events_csv": args.midi_events_csv,
        },
        "parameters": {
            "cloud_keep": int(args.cloud_keep),
        },
        "result": {
            "top1_changed": top1_changed,
            "event_top1": dict(event_top1),
            "event_top3": dict(event_top3),
            "event_top5": dict(event_top5),
            "group_top3": dict(group_top3),
            "group_top5": dict(group_top5),
        },
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
