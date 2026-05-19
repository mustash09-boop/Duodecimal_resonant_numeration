from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import mido

from music12.core.notation12 import abs_semitone_to_token, parse_token


MIDI_TO_SYSTEM_OCTAVE_SHIFT = 36  # 3 октавы вверх

def midi_note_to_token(midi_note: int) -> str:
    """
    MIDI 0 -> system token with fixed +3 octave shift.
    3 octaves = 36 semitone steps.
    """
    if midi_note < 0:
        raise ValueError("midi_note must be >= 0")
    return abs_semitone_to_token(int(midi_note) + MIDI_TO_SYSTEM_OCTAVE_SHIFT)


def round_frame60(seconds: float) -> int:
    return int(round(seconds * 60.0))


def load_midi_note_events(midi_path: Path) -> Tuple[List[dict], Dict[str, float]]:
    mid = mido.MidiFile(str(midi_path))

    tempo = 500000  # default 120 BPM until first set_tempo
    ticks_per_beat = mid.ticks_per_beat

    active: Dict[Tuple[int, int], List[Tuple[float, int]]] = defaultdict(list)
    events: List[dict] = []

    for track_index, track in enumerate(mid.tracks):
        abs_ticks = 0
        abs_seconds = 0.0
        current_tempo = tempo

        for msg in track:
            delta_ticks = msg.time
            abs_ticks += delta_ticks
            abs_seconds += mido.tick2second(delta_ticks, ticks_per_beat, current_tempo)

            if msg.type == "set_tempo":
                current_tempo = msg.tempo
                continue

            if msg.type == "note_on" and msg.velocity > 0:
                active[(track_index, msg.note)].append((abs_seconds, msg.velocity))
                continue

            if msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (track_index, msg.note)
                if not active[key]:
                    continue

                start_sec, velocity = active[key].pop()
                end_sec = abs_seconds
                if end_sec < start_sec:
                    continue

                token = midi_note_to_token(msg.note)
                parsed = parse_token(token)

                events.append(
                    {
                        "track_index": track_index,
                        "midi_note": int(msg.note),
                        "velocity": int(velocity),
                        "time_start_sec": float(start_sec),
                        "time_end_sec": float(end_sec),
                        "duration_sec": float(end_sec - start_sec),
                        "note_token": token,
                        "octave_token": parsed.oct,
                        "step_token": parsed.step,
                    }
                )

    events.sort(key=lambda x: (x["time_start_sec"], x["midi_note"], x["time_end_sec"]))

    meta = {
        "ticks_per_beat": ticks_per_beat,
        "track_count": len(mid.tracks),
        "type": mid.type,
        "length_seconds": float(mid.length),
    }
    return events, meta


def build_onset_groups(events: List[dict], time_quant_sec: float = 1.0 / 120.0) -> Tuple[List[dict], List[dict]]:
    """
    Group starts that are effectively simultaneous.
    Default quantization is 1/120 sec to avoid floating-point dust.
    """
    onset_buckets: Dict[int, List[int]] = defaultdict(list)

    for i, ev in enumerate(events):
        bucket = int(round(ev["time_start_sec"] / time_quant_sec))
        onset_buckets[bucket].append(i)

    onset_rows: List[dict] = []
    enriched_events: List[dict] = []

    bucket_items = sorted(onset_buckets.items(), key=lambda x: x[0])

    event_to_group: Dict[int, int] = {}
    group_id = 0

    for bucket, event_indices in bucket_items:
        t_sec = min(events[i]["time_start_sec"] for i in event_indices)
        notes = [events[i]["note_token"] for i in event_indices]
        notes_sorted = sorted(notes)

        onset_rows.append(
            {
                "onset_group": group_id,
                "time_start_sec": float(t_sec),
                "start_frame60": round_frame60(t_sec),
                "polyphony": len(event_indices),
                "notes": " | ".join(notes_sorted),
            }
        )

        for i in event_indices:
            event_to_group[i] = group_id

        group_id += 1

    for i, ev in enumerate(events):
        onset_group = event_to_group[i]
        onset_info = onset_rows[onset_group]

        row = dict(ev)
        row["event_index"] = i
        row["start_frame60"] = round_frame60(ev["time_start_sec"])
        row["end_frame60"] = round_frame60(ev["time_end_sec"])
        row["duration_frames60"] = row["end_frame60"] - row["start_frame60"]
        row["onset_group"] = onset_group
        row["onset_polyphony"] = onset_info["polyphony"]
        row["onset_notes"] = onset_info["notes"]
        enriched_events.append(row)

    return enriched_events, onset_rows


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Read MIDI and export note events in 12-radix notation with 1/60s grid."
    )
    ap.add_argument("--midi", required=True, help="Input MIDI file")
    ap.add_argument("--out_events_csv", required=True, help="Output CSV with note events")
    ap.add_argument("--out_onsets_csv", required=True, help="Output CSV with onset groups")
    ap.add_argument("--out_meta_json", required=True, help="Output JSON metadata")
    args = ap.parse_args()

    midi_path = Path(args.midi).resolve()
    out_events_csv = Path(args.out_events_csv).resolve()
    out_onsets_csv = Path(args.out_onsets_csv).resolve()
    out_meta_json = Path(args.out_meta_json).resolve()

    events, midi_meta = load_midi_note_events(midi_path)
    enriched_events, onset_rows = build_onset_groups(events)

    event_fields = [
        "event_index",
        "track_index",
        "midi_note",
        "velocity",
        "time_start_sec",
        "time_end_sec",
        "duration_sec",
        "start_frame60",
        "end_frame60",
        "duration_frames60",
        "note_token",
        "octave_token",
        "step_token",
        "onset_group",
        "onset_polyphony",
        "onset_notes",
    ]

    onset_fields = [
        "onset_group",
        "time_start_sec",
        "start_frame60",
        "polyphony",
        "notes",
    ]

    write_csv(out_events_csv, enriched_events, event_fields)
    write_csv(out_onsets_csv, onset_rows, onset_fields)

    meta = {
        "midi": str(midi_path),
        "midi_meta": midi_meta,
        "derived": {
            "event_count": len(enriched_events),
            "onset_group_count": len(onset_rows),
            "max_polyphony": max((r["polyphony"] for r in onset_rows), default=0),
        },
        "grid": {
            "frame_rate_hz": 60.0,
            "frame_duration_sec": 1.0 / 60.0,
        },
        "notation": {
            "anchor_note": "9.A",
            "anchor_hz": 440.0,
            "digit_alphabet": "123456789ABC",
            "zero_allowed": False,
            "midi_octave_shift": 36
        },
        "outputs": {
            "events_csv": str(out_events_csv),
            "onsets_csv": str(out_onsets_csv),
            "meta_json": str(out_meta_json),
        },
    }

    out_meta_json.parent.mkdir(parents=True, exist_ok=True)
    out_meta_json.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("midi to duodecimal events complete")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()