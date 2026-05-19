# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _detect_schema(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "empty"

    keys = set(rows[0].keys())

    if {"note_token", "attentional_salience"}.issubset(keys):
        return "lineage_attention"

    if "confirmed_notes" in keys:
        return "confirmed_notes"

    return "unknown"


def _parse_lineage_attention_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []

    for r in rows:
        note = str(r.get("note_token", "")).strip()
        if not note:
            continue

        frame_index = _safe_int(r.get("frame_index"), 0)
        time_sec = _safe_float(r.get("time_sec"), 0.0)
        score = _safe_float(r.get("attentional_salience"), 0.0)
        entity_id = str(r.get("entity_id", "")).strip()
        focus_rank = _safe_int(r.get("focus_rank"), 0)

        out.append({
            "frame_index": frame_index,
            "time_sec": time_sec,
            "note_token": note,
            "score": score,
            "entity_id": entity_id,
            "focus_rank": focus_rank,
            "identity_key": f"{entity_id}::{note}" if entity_id else note,
            "source_schema": "lineage_attention",
        })

    return out


def _parse_confirmed_notes_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []

    for r in rows:
        frame_index = _safe_int(r.get("frame_index"), 0)
        time_sec = _safe_float(r.get("time_sec"), 0.0)
        raw = str(r.get("confirmed_notes", "")).strip()

        if not raw:
            continue

        for part in raw.split("|"):
            part = part.strip()
            if not part or ":" not in part:
                continue

            note, score = part.rsplit(":", 1)
            note = note.strip()
            score_f = _safe_float(score, 0.0)

            out.append({
                "frame_index": frame_index,
                "time_sec": time_sec,
                "note_token": note,
                "score": score_f,
                "entity_id": "",
                "focus_rank": 0,
                "identity_key": note,
                "source_schema": "confirmed_notes",
            })

    return out


def _build_events(
    frame_notes: List[Dict[str, Any]],
    *,
    max_gap_sec: float,
) -> List[Dict[str, Any]]:
    frame_notes.sort(
        key=lambda x: (
            str(x.get("identity_key", "")),
            _safe_float(x.get("time_sec"), 0.0),
        )
    )

    events = []
    active: Dict[str, Dict[str, Any]] = {}

    for fn in frame_notes:
        key = str(fn["identity_key"])
        t = _safe_float(fn["time_sec"], 0.0)
        score = _safe_float(fn["score"], 0.0)

        if key not in active:
            active[key] = {
                "identity_key": key,
                "note_token": fn["note_token"],
                "entity_id": fn.get("entity_id", ""),
                "start_sec": t,
                "end_sec": t,
                "scores": [score],
                "frame_indices": [fn["frame_index"]],
                "focus_ranks": [fn.get("focus_rank", 0)],
            }
            continue

        ev = active[key]
        gap = t - ev["end_sec"]

        if gap <= max_gap_sec:
            ev["end_sec"] = t
            ev["scores"].append(score)
            ev["frame_indices"].append(fn["frame_index"])
            ev["focus_ranks"].append(fn.get("focus_rank", 0))
        else:
            events.append(ev)
            active[key] = {
                "identity_key": key,
                "note_token": fn["note_token"],
                "entity_id": fn.get("entity_id", ""),
                "start_sec": t,
                "end_sec": t,
                "scores": [score],
                "frame_indices": [fn["frame_index"]],
                "focus_ranks": [fn.get("focus_rank", 0)],
            }

    events.extend(active.values())
    return events


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Track persistent note identities from framewise note streams."
    )

    ap.add_argument("--frame_summary_csv", required=True)
    ap.add_argument("--out_note_events_csv", required=True)
    ap.add_argument("--out_frame_notes_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--min_duration_sec", type=float, default=0.08)
    ap.add_argument("--max_gap_sec", type=float, default=0.05)
    ap.add_argument("--min_mean_score", type=float, default=0.24)

    args = ap.parse_args()

    in_csv = Path(args.frame_summary_csv)
    out_events = Path(args.out_note_events_csv)
    out_frames = Path(args.out_frame_notes_csv)
    out_meta = Path(args.out_meta_json)
    out_txt = Path(args.out_summary_txt)

    rows = _load_rows(in_csv)
    schema = _detect_schema(rows)

    if schema == "lineage_attention":
        frame_notes = _parse_lineage_attention_rows(rows)
    elif schema == "confirmed_notes":
        frame_notes = _parse_confirmed_notes_rows(rows)
    elif schema == "empty":
        frame_notes = []
    else:
        raise RuntimeError(
            "Unsupported frame_summary_csv schema. Expected either "
            "'note_token + attentional_salience' or 'confirmed_notes'."
        )

    events = _build_events(
        frame_notes,
        max_gap_sec=args.max_gap_sec,
    )

    final_events = []

    for ev in events:
        duration = ev["end_sec"] - ev["start_sec"]
        mean_score = sum(ev["scores"]) / max(len(ev["scores"]), 1)
        max_score = max(ev["scores"]) if ev["scores"] else 0.0
        mean_focus_rank = (
            sum(_safe_float(x, 0.0) for x in ev["focus_ranks"])
            / max(len(ev["focus_ranks"]), 1)
        )

        if duration < args.min_duration_sec:
            continue

        if mean_score < args.min_mean_score:
            continue

        final_events.append({
            "event_id": len(final_events) + 1,
            "identity_key": ev["identity_key"],
            "entity_id": ev.get("entity_id", ""),
            "note_token": ev["note_token"],
            "time_start_sec": f"{ev['start_sec']:.9f}",
            "time_end_sec": f"{ev['end_sec']:.9f}",
            "duration_sec": f"{duration:.9f}",
            "frame_count": len(ev["frame_indices"]),
            "mean_score": f"{mean_score:.9f}",
            "max_score": f"{max_score:.9f}",
            "mean_focus_rank": f"{mean_focus_rank:.9f}",
            "start_frame": min(ev["frame_indices"]),
            "end_frame": max(ev["frame_indices"]),
        })

    final_events.sort(
        key=lambda r: (
            _safe_float(r["time_start_sec"]),
            r["note_token"],
            r["identity_key"],
        )
    )

    out_events.parent.mkdir(parents=True, exist_ok=True)

    event_fields = [
        "event_id",
        "identity_key",
        "entity_id",
        "note_token",
        "time_start_sec",
        "time_end_sec",
        "duration_sec",
        "frame_count",
        "mean_score",
        "max_score",
        "mean_focus_rank",
        "start_frame",
        "end_frame",
    ]

    with out_events.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=event_fields)
        w.writeheader()
        w.writerows(final_events)

    frame_fields = [
        "frame_index",
        "time_sec",
        "identity_key",
        "entity_id",
        "note_token",
        "score",
        "focus_rank",
        "source_schema",
    ]

    with out_frames.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=frame_fields)
        w.writeheader()
        w.writerows(frame_notes)

    meta = {
        "stage": "persistent_note_identity_tracker",
        "input": str(in_csv),
        "input_schema": schema,
        "outputs": {
            "note_events_csv": str(out_events),
            "frame_notes_csv": str(out_frames),
            "meta_json": str(out_meta),
            "summary_txt": str(out_txt),
        },
        "parameters": {
            "min_duration_sec": args.min_duration_sec,
            "max_gap_sec": args.max_gap_sec,
            "min_mean_score": args.min_mean_score,
        },
        "result": {
            "raw_frame_note_rows": len(frame_notes),
            "raw_events_before_filter": len(events),
            "final_note_events": len(final_events),
        },
    }

    out_meta.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    txt = [
        "PERSISTENT NOTE IDENTITY TRACKER",
        "=" * 72,
        f"input                    : {in_csv}",
        f"input_schema             : {schema}",
        f"note_events_csv          : {out_events}",
        f"frame_notes_csv          : {out_frames}",
        f"raw_frame_note_rows      : {len(frame_notes)}",
        f"raw_events_before_filter : {len(events)}",
        f"final_note_events        : {len(final_events)}",
        "",
        "Principle:",
        "  This stage converts framewise note identities into",
        "  persistent harmonic identity events over time.",
        "  Lineage-attention streams are tracked by entity_id + note_token.",
        "  Single-frame apparitions are suppressed.",
        "",
    ]

    out_txt.write_text("\n".join(txt), encoding="utf-8")

    print("persistent note identity tracker complete")
    print(json.dumps(meta["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()