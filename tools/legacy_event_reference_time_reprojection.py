# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import bisect
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


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


def _interp_time(real_times: List[float], ref_times: List[float], t: float) -> float:
    i = bisect.bisect_left(real_times, t)
    if i <= 0:
        return ref_times[0]
    if i >= len(real_times):
        return ref_times[-1]
    x0, x1 = real_times[i - 1], real_times[i]
    y0, y1 = ref_times[i - 1], ref_times[i]
    if x1 == x0:
        return y0
    a = (t - x0) / (x1 - x0)
    return y0 + a * (y1 - y0)


def _soft_group_count(frames: List[int], window: int) -> int:
    if not frames:
        return 0
    frames = sorted(frames)
    groups = 1
    anchor = frames[0]
    for f in frames[1:]:
        if f - anchor > window:
            groups += 1
            anchor = f
    return groups


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproject legacy live-piano events from performance time into MIDI reference time.")
    ap.add_argument("--events-csv", required=True)
    ap.add_argument("--tempo-aligned-csv", required=True)
    ap.add_argument("--midi-meta-json", required=True)
    ap.add_argument("--out-events-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--soft-window", type=int, default=3)
    args = ap.parse_args()

    events = _load_csv(Path(args.events_csv))
    align = _load_csv(Path(args.tempo_aligned_csv))
    midi_meta = json.loads(Path(args.midi_meta_json).read_text(encoding="utf-8"))

    real_times = [_safe_float(r.get("time_real_sec"), 0.0) for r in align]
    ref_times = [_safe_float(r.get("time_reference_sec"), 0.0) for r in align]

    out_rows: List[Dict[str, Any]] = []
    ref_birth_frames: List[int] = []
    ref_end_frames: List[int] = []
    for r in events:
        birth_frame = _safe_int(r.get("birth_frame"), 0)
        end_frame = _safe_int(r.get("end_frame"), birth_frame)
        birth_real_sec = birth_frame / max(args.fps, 1e-9)
        end_real_sec = end_frame / max(args.fps, 1e-9)
        birth_ref_sec = _interp_time(real_times, ref_times, birth_real_sec)
        end_ref_sec = _interp_time(real_times, ref_times, end_real_sec)
        birth_ref_frame = round(birth_ref_sec * args.fps)
        end_ref_frame = round(end_ref_sec * args.fps)
        rr = dict(r)
        rr["birth_frame_reference60"] = birth_ref_frame
        rr["end_frame_reference60"] = end_ref_frame
        rr["birth_sec_reference"] = f"{birth_ref_sec:.9f}"
        rr["end_sec_reference"] = f"{end_ref_sec:.9f}"
        rr["birth_sec_real"] = f"{birth_real_sec:.9f}"
        rr["end_sec_real"] = f"{end_real_sec:.9f}"
        out_rows.append(rr)
        ref_birth_frames.append(birth_ref_frame)
        ref_end_frames.append(end_ref_frame)

    out_rows.sort(key=lambda r: (_safe_int(r.get("birth_frame_reference60"), 0), str(r.get("candidate_note", ""))))
    _write_csv(Path(args.out_events_csv), out_rows, out_rows[0].keys())

    raw_groups = _soft_group_count([_safe_int(r.get("birth_frame"), 0) for r in events], args.soft_window)
    ref_groups = _soft_group_count(ref_birth_frames, args.soft_window)

    summary_lines = [
        "LEGACY EVENT REFERENCE TIME REPROJECTION",
        "=" * 72,
        f"events_csv                 : {args.events_csv}",
        f"tempo_aligned_csv          : {args.tempo_aligned_csv}",
        f"input_events               : {len(events)}",
        f"raw_soft_onset_groups      : {raw_groups}",
        f"reference_soft_onset_groups: {ref_groups}",
        f"target_event_count         : {midi_meta.get('event_count', 0)}",
        f"target_onset_groups        : {midi_meta.get('unique_onset_groups', 0)}",
        f"event_gap_to_target        : {len(events) - int(midi_meta.get('event_count', 0))}",
        f"reference_soft_gap_to_target: {ref_groups - int(midi_meta.get('unique_onset_groups', 0))}",
    ]
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "inputs": {
                    "events_csv": args.events_csv,
                    "tempo_aligned_csv": args.tempo_aligned_csv,
                    "midi_meta_json": args.midi_meta_json,
                },
                "parameters": {
                    "fps": args.fps,
                    "soft_window": args.soft_window,
                },
                "result": {
                    "input_events": len(events),
                    "raw_soft_onset_groups": raw_groups,
                    "reference_soft_onset_groups": ref_groups,
                    "target_event_count": midi_meta.get("event_count", 0),
                    "target_onset_groups": midi_meta.get("unique_onset_groups", 0),
                    "event_gap_to_target": len(events) - int(midi_meta.get("event_count", 0)),
                    "reference_soft_gap_to_target": ref_groups - int(midi_meta.get("unique_onset_groups", 0)),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
