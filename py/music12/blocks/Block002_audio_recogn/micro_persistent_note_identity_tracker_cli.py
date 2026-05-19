# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List


ITEM_RE = re.compile(r"(?P<note>[^:|]+):(?P<score>[0-9.]+)")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _extract_notes(raw: str) -> List[Dict[str, Any]]:
    out = []

    for part in str(raw or "").split("|"):
        part = part.strip()
        if not part:
            continue

        m = ITEM_RE.search(part)
        if not m:
            continue

        note = m.group("note").strip()
        score = _safe_float(m.group("score"), 0.0)

        if note:
            out.append({
                "note_token": note,
                "score": score,
            })

    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Track persistent notes from micro-region frame summaries."
    )

    ap.add_argument("--frame_summary_csv", required=True)
    ap.add_argument("--out_note_events_csv", required=True)
    ap.add_argument("--out_frame_notes_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--source_column", default="top_region_families")
    ap.add_argument("--min_duration_sec", type=float, default=0.08)
    ap.add_argument("--max_gap_sec", type=float, default=0.05)
    ap.add_argument("--min_mean_score", type=float, default=0.72)

    args = ap.parse_args()

    in_csv = Path(args.frame_summary_csv)
    out_events = Path(args.out_note_events_csv)
    out_frames = Path(args.out_frame_notes_csv)
    out_meta = Path(args.out_meta_json)
    out_txt = Path(args.out_summary_txt)

    rows = _load_csv(in_csv)

    frame_notes = []

    for r in rows:
        frame_index = _safe_int(r.get("frame_index"), 0)

        # Если time_sec нет, восстанавливаем по 60 fps.
        time_sec = _safe_float(
            r.get("time_sec"),
            frame_index / 60.0,
        )

        raw = str(r.get(args.source_column, "")).strip()

        for item in _extract_notes(raw):
            frame_notes.append({
                "frame_index": frame_index,
                "time_sec": time_sec,
                "note_token": item["note_token"],
                "score": item["score"],
            })

    frame_notes.sort(key=lambda x: (x["note_token"], x["time_sec"]))

    active: Dict[str, Dict[str, Any]] = {}
    events = []

    for fn in frame_notes:
        note = fn["note_token"]
        t = fn["time_sec"]
        score = fn["score"]

        if note not in active:
            active[note] = {
                "note_token": note,
                "start_sec": t,
                "end_sec": t,
                "scores": [score],
                "frame_indices": [fn["frame_index"]],
            }
            continue

        ev = active[note]
        gap = t - ev["end_sec"]

        if gap <= args.max_gap_sec:
            ev["end_sec"] = t
            ev["scores"].append(score)
            ev["frame_indices"].append(fn["frame_index"])
        else:
            events.append(ev)
            active[note] = {
                "note_token": note,
                "start_sec": t,
                "end_sec": t,
                "scores": [score],
                "frame_indices": [fn["frame_index"]],
            }

    events.extend(active.values())

    final_events = []

    for ev in events:
        duration = ev["end_sec"] - ev["start_sec"]
        mean_score = sum(ev["scores"]) / max(len(ev["scores"]), 1)
        max_score = max(ev["scores"]) if ev["scores"] else 0.0

        if duration < args.min_duration_sec:
            continue

        if mean_score < args.min_mean_score:
            continue

        final_events.append({
            "event_id": len(final_events) + 1,
            "note_token": ev["note_token"],
            "time_start_sec": f"{ev['start_sec']:.9f}",
            "time_end_sec": f"{ev['end_sec']:.9f}",
            "duration_sec": f"{duration:.9f}",
            "frame_count": len(ev["frame_indices"]),
            "mean_score": f"{mean_score:.9f}",
            "max_score": f"{max_score:.9f}",
            "start_frame": min(ev["frame_indices"]),
            "end_frame": max(ev["frame_indices"]),
        })

    final_events.sort(
        key=lambda r: (
            _safe_float(r["time_start_sec"]),
            r["note_token"],
        )
    )

    out_events.parent.mkdir(parents=True, exist_ok=True)

    event_fields = [
        "event_id",
        "note_token",
        "time_start_sec",
        "time_end_sec",
        "duration_sec",
        "frame_count",
        "mean_score",
        "max_score",
        "start_frame",
        "end_frame",
    ]

    with out_events.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=event_fields)
        w.writeheader()
        w.writerows(final_events)

    frame_fields = ["frame_index", "time_sec", "note_token", "score"]

    with out_frames.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=frame_fields)
        w.writeheader()
        w.writerows(frame_notes)

    meta = {
        "stage": "micro_persistent_note_identity_tracker",
        "input": str(in_csv),
        "source_column": args.source_column,
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
        "MICRO PERSISTENT NOTE IDENTITY TRACKER",
        "=" * 72,
        f"input                    : {in_csv}",
        f"source_column            : {args.source_column}",
        f"note_events_csv          : {out_events}",
        f"frame_notes_csv          : {out_frames}",
        "",
        f"raw_frame_note_rows      : {len(frame_notes)}",
        f"raw_events_before_filter : {len(events)}",
        f"final_note_events        : {len(final_events)}",
        "",
        "Principle:",
        "  Convert micro-region frame resonance identities",
        "  into persistent note events over time.",
        "",
    ]

    out_txt.write_text("\n".join(txt), encoding="utf-8")

    print("micro persistent note tracker complete")
    print(json.dumps(meta["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()