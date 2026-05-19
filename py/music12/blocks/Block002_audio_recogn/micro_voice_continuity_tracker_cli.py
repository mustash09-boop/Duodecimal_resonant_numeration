# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


ALPHABET12 = "123456789ABC"


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


def _token_to_abs_degree(token: str) -> Optional[int]:
    try:
        token = str(token).strip().upper()
        octave_raw, rest = token.split(".", 1)
        degree_raw = rest.split("'", 1)[0]

        octave = 0
        for ch in octave_raw:
            if ch not in ALPHABET12:
                return None
            octave = octave * 12 + (ALPHABET12.index(ch) + 1)

        if degree_raw not in ALPHABET12:
            return None

        return octave * 12 + ALPHABET12.index(degree_raw)
    except Exception:
        return None


def _pitch_distance(a: str, b: str) -> float:
    aa = _token_to_abs_degree(a)
    bb = _token_to_abs_degree(b)

    if aa is None or bb is None:
        return 9999.0

    return abs(float(aa - bb))


def _group_by_frame(rows: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}

    for r in rows:
        frame = _safe_int(r.get("frame_index"), 0)
        out.setdefault(frame, []).append(r)

    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Track continuous voice lines from simultaneous micro causal notes."
    )

    ap.add_argument("--frame_notes_csv", required=True)

    ap.add_argument("--out_voice_events_csv", required=True)
    ap.add_argument("--out_voice_summary_csv", required=True)
    ap.add_argument("--out_frame_voice_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--max_pitch_jump", type=float, default=5.0)
    ap.add_argument("--max_gap_frames", type=int, default=3)
    ap.add_argument("--min_voice_len_frames", type=int, default=6)

    args = ap.parse_args()

    rows = _load_csv(Path(args.frame_notes_csv))
    by_frame = _group_by_frame(rows)

    active_voices: Dict[int, Dict[str, Any]] = {}
    finished_voices: List[Dict[str, Any]] = []
    frame_voice_rows: List[Dict[str, Any]] = []

    next_voice_id = 1

    for frame in sorted(by_frame):
        notes = sorted(
            by_frame[frame],
            key=lambda r: _safe_int(r.get("rank"), 999),
        )

        used_voice_ids = set()

        for note_row in notes:
            note = str(note_row.get("note_token", "")).strip()
            score = _safe_float(note_row.get("score"), 0.0)

            best_voice_id = None
            best_cost = 999999.0

            for vid, voice in active_voices.items():
                if vid in used_voice_ids:
                    continue

                gap = frame - voice["last_frame"]

                if gap < 0 or gap > args.max_gap_frames:
                    continue

                dist = _pitch_distance(note, voice["last_note"])

                if dist > args.max_pitch_jump:
                    continue

                # Prefer small pitch movement, small gap, stronger score.
                cost = dist + gap * 0.75 - score * 0.05

                if cost < best_cost:
                    best_cost = cost
                    best_voice_id = vid

            if best_voice_id is None:
                vid = next_voice_id
                next_voice_id += 1

                active_voices[vid] = {
                    "voice_id": vid,
                    "start_frame": frame,
                    "last_frame": frame,
                    "last_note": note,
                    "notes": [note],
                    "scores": [score],
                    "frame_indices": [frame],
                }
            else:
                vid = best_voice_id
                voice = active_voices[vid]
                voice["last_frame"] = frame
                voice["last_note"] = note
                voice["notes"].append(note)
                voice["scores"].append(score)
                voice["frame_indices"].append(frame)

            used_voice_ids.add(vid)

            frame_voice_rows.append({
                "frame_index": frame,
                "voice_id": vid,
                "note_token": note,
                "score": f"{score:.9f}",
                "rank": note_row.get("rank", ""),
            })

        # Close voices that have not been seen for too long.
        to_close = []

        for vid, voice in active_voices.items():
            if frame - voice["last_frame"] > args.max_gap_frames:
                to_close.append(vid)

        for vid in to_close:
            finished_voices.append(active_voices.pop(vid))

    finished_voices.extend(active_voices.values())

    kept_voices = []

    for voice in finished_voices:
        length = len(voice["frame_indices"])

        if length < args.min_voice_len_frames:
            continue

        mean_score = sum(voice["scores"]) / max(length, 1)

        kept_voices.append({
            "voice_id": voice["voice_id"],
            "start_frame": min(voice["frame_indices"]),
            "end_frame": max(voice["frame_indices"]),
            "length_frames": length,
            "mean_score": f"{mean_score:.9f}",
            "start_note": voice["notes"][0],
            "end_note": voice["notes"][-1],
            "unique_note_count": len(set(voice["notes"])),
            "note_path": " ".join(voice["notes"]),
        })

    kept_voices.sort(
        key=lambda r: (
            _safe_int(r["start_frame"]),
            _safe_int(r["voice_id"]),
        )
    )

    out_voice_events = Path(args.out_voice_events_csv)
    out_voice_summary = Path(args.out_voice_summary_csv)
    out_frame_voice = Path(args.out_frame_voice_csv)
    out_meta = Path(args.out_meta_json)
    out_txt = Path(args.out_summary_txt)

    out_voice_events.parent.mkdir(parents=True, exist_ok=True)

    voice_fields = [
        "voice_id",
        "start_frame",
        "end_frame",
        "length_frames",
        "mean_score",
        "start_note",
        "end_note",
        "unique_note_count",
        "note_path",
    ]

    with out_voice_events.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=voice_fields)
        w.writeheader()
        w.writerows(kept_voices)

    mean_voice_len = (
        sum(_safe_int(v["length_frames"]) for v in kept_voices)
        / max(len(kept_voices), 1)
    )

    summary_rows = [{
        "input_frame_note_rows": len(rows),
        "finished_voices_total": len(finished_voices),
        "kept_voices": len(kept_voices),
        "max_voice_length_frames": max(
            (_safe_int(v["length_frames"]) for v in kept_voices),
            default=0,
        ),
        "mean_voice_length_frames": f"{mean_voice_len:.6f}",
    }]

    with out_voice_summary.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "input_frame_note_rows",
                "finished_voices_total",
                "kept_voices",
                "max_voice_length_frames",
                "mean_voice_length_frames",
            ],
        )
        w.writeheader()
        w.writerows(summary_rows)

    with out_frame_voice.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "frame_index",
                "voice_id",
                "note_token",
                "score",
                "rank",
            ],
        )
        w.writeheader()
        w.writerows(frame_voice_rows)

    meta = {
        "stage": "micro_voice_continuity_tracker",
        "inputs": {
            "frame_notes_csv": args.frame_notes_csv,
        },
        "outputs": {
            "voice_events_csv": args.out_voice_events_csv,
            "voice_summary_csv": args.out_voice_summary_csv,
            "frame_voice_csv": args.out_frame_voice_csv,
            "meta_json": args.out_meta_json,
            "summary_txt": args.out_summary_txt,
        },
        "parameters": {
            "max_pitch_jump": args.max_pitch_jump,
            "max_gap_frames": args.max_gap_frames,
            "min_voice_len_frames": args.min_voice_len_frames,
        },
        "result": summary_rows[0],
    }

    out_meta.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    txt = [
        "MICRO VOICE CONTINUITY TRACKER",
        "=" * 72,
        f"frame_notes_csv       : {args.frame_notes_csv}",
        "",
        f"input_frame_note_rows : {len(rows)}",
        f"finished_voices_total : {len(finished_voices)}",
        f"kept_voices           : {len(kept_voices)}",
        f"max_voice_length      : {summary_rows[0]['max_voice_length_frames']}",
        f"mean_voice_length     : {summary_rows[0]['mean_voice_length_frames']}",
        "",
        "Principle:",
        "  Convert simultaneous causal note frames",
        "  into continuous voice trajectories.",
        "",
    ]

    out_txt.write_text("\n".join(txt), encoding="utf-8")

    print("micro voice continuity tracker complete")
    print(json.dumps(meta["result"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()