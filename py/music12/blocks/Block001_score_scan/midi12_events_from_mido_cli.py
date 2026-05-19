# -*- coding: utf-8 -*-
"""
MIDI12 EVENTS FROM MIDO

Current-compatible MIDI -> midi_events_12.csv converter.

Purpose:
- create MIDI reference events for Bach_Invention_1 and similar score/audio comparisons
- preserve polyphonic reference information
- generate the CSV schema expected by midi_reference_structure_cli.py

Does NOT infer f0.
Does NOT generate harmonic hypotheses.

Anchor convention:
MIDI 69 = 9.A'- = A4 = 440 Hz
"""

import argparse
import csv
import json
import os

import mido


DEGREES = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C"]


def to_base12_no_zero(n: int) -> str:
    """
    Project-style positive 12-radix octave representation using:
    1 2 3 4 5 6 7 8 9 A B C

    For 1..12:
        1 -> 1
        ...
        9 -> 9
        10 -> A
        11 -> B
        12 -> C

    For values above 12:
        13 -> 11
        14 -> 12
        ...
    """
    if n <= 0:
        raise ValueError(f"Invalid octave for no-zero notation: {n}")

    if 1 <= n <= 12:
        return DEGREES[n - 1]

    digits = []
    x = n

    while x > 0:
        r = x % 12

        if r == 0:
            digits.append("C")
            x = x // 12 - 1
        else:
            digits.append(DEGREES[r - 1])
            x = x // 12

    return "".join(reversed(digits))


def semitone_to_token_from_a4(midi_note: int) -> str:
    """
    Map MIDI notes into project 12-radix token space.

    MIDI 69 -> 9.A'-.
    Each semitone increments degree.
    Every 12 degrees roll into the next octave.
    """
    delta = int(midi_note) - 69

    anchor_oct = 9
    anchor_degree_index = 9  # A is degree index 9 in 1..9,A,B,C

    absolute_degree = anchor_oct * 12 + anchor_degree_index + delta

    octave = absolute_degree // 12
    degree = absolute_degree % 12

    return f"{to_base12_no_zero(octave)}.{DEGREES[degree]}'-"


def midi_note_to_hz(midi_note: int) -> float:
    return 440.0 * (2.0 ** ((int(midi_note) - 69) / 12.0))


def read_midi_events(path: str):
    """
    Read MIDI file and return note events with absolute time in seconds.

    Handles:
    - note_on velocity > 0 as note start
    - note_off or note_on velocity == 0 as note end

    Important:
    This keeps track/channel separation.
    """
    mid = mido.MidiFile(path)

    ticks_per_beat = mid.ticks_per_beat
    tempo = 500000  # default 120 bpm

    events = []
    active = {}

    for track_index, track in enumerate(mid.tracks):
        abs_tick = 0
        current_tempo = tempo

        for msg in track:
            abs_tick += msg.time

            if msg.type == "set_tempo":
                current_tempo = msg.tempo
                continue

            sec = mido.tick2second(abs_tick, ticks_per_beat, current_tempo)

            channel = msg.channel if hasattr(msg, "channel") else 0

            if msg.type == "note_on" and msg.velocity > 0:
                key = (track_index, channel, msg.note)

                active[key] = {
                    "track": track_index,
                    "channel": channel,
                    "midi_note": msg.note,
                    "velocity": msg.velocity,
                    "start_tick": abs_tick,
                    "start_sec": sec,
                }

            elif msg.type == "note_off" or (msg.type == "note_on" and getattr(msg, "velocity", 0) == 0):
                note = getattr(msg, "note", None)

                if note is None:
                    continue

                key = (track_index, channel, note)

                if key not in active:
                    continue

                ev = active.pop(key)

                ev["end_tick"] = abs_tick
                ev["end_sec"] = sec
                ev["duration_sec"] = max(0.0, ev["end_sec"] - ev["start_sec"])

                events.append(ev)

    events.sort(key=lambda r: (r["start_sec"], r["midi_note"], r["track"]))
    return events


def assign_onset_groups(rows, tolerance_sec: float = 1.0 / 60.0):
    """
    Assign onset groups using a 60-based temporal tolerance.

    Events that start within tolerance_sec are treated as belonging
    to the same onset group.

    Adds:
    - onset_group
    - onset_notes
    - onset_polyphony
    """
    if not rows:
        return rows

    sorted_rows = sorted(rows, key=lambda r: (float(r["time_start_sec"]), int(r["midi_note"])))

    groups = []
    current = []
    current_start = None

    for row in sorted_rows:
        t = float(row["time_start_sec"])

        if current_start is None:
            current_start = t
            current = [row]
            continue

        if abs(t - current_start) <= tolerance_sec:
            current.append(row)
        else:
            groups.append(current)
            current_start = t
            current = [row]

    if current:
        groups.append(current)

    for group_index, group in enumerate(groups, start=1):
        notes = sorted(set(r["note12"] for r in group))
        onset_notes = " ".join(notes)
        onset_polyphony = len(notes)

        for r in group:
            r["onset_group"] = group_index
            r["onset_notes"] = onset_notes
            r["onset_polyphony"] = onset_polyphony

    return sorted(rows, key=lambda r: int(r["event_index"]))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--midi", required=True)
    ap.add_argument("--out_events_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument(
        "--onset_tolerance_sec",
        type=float,
        default=1.0 / 60.0,
        help="Events starting within this tolerance are grouped as one onset group.",
    )

    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_events_csv), exist_ok=True)

    events = read_midi_events(args.midi)

    rows = []

    for i, ev in enumerate(events, start=1):
        token = semitone_to_token_from_a4(ev["midi_note"])
        hz = midi_note_to_hz(ev["midi_note"])

        octave_token = token.split(".")[0]
        step_token = token.split(".")[1].split("'")[0]

        start_frame60 = int(round(ev["start_sec"] * 60.0))
        end_frame60 = int(round(ev["end_sec"] * 60.0))
        duration_frames60 = max(0, end_frame60 - start_frame60)

        rows.append({
            "event_index": i,
            "event_id": i,

            "track_index": ev["track"],
            "track": ev["track"],

            "channel": ev["channel"],
            "midi_note": ev["midi_note"],
            "velocity": ev["velocity"],

            "time_start_sec": f"{ev['start_sec']:.9f}",
            "time_end_sec": f"{ev['end_sec']:.9f}",

            "start_sec": f"{ev['start_sec']:.9f}",
            "end_sec": f"{ev['end_sec']:.9f}",
            "duration_sec": f"{ev['duration_sec']:.9f}",

            "start_frame60": start_frame60,
            "end_frame60": end_frame60,
            "duration_frames60": duration_frames60,

            "note12": token,
            "expected_note": token,
            "expected_note_token": token,
            "note_token": token,
            "token": token,

            "octave_token": octave_token,
            "step_token": step_token,

            "onset_group": 0,
            "onset_notes": token,
            "onset_polyphony": 1,

            "freq_hz": f"{hz:.6f}",
            "semantic_layer": "midi_reference",
        })

    rows = assign_onset_groups(rows, tolerance_sec=args.onset_tolerance_sec)

    fieldnames = [
        "event_index",
        "event_id",

        "track_index",
        "track",

        "channel",
        "midi_note",
        "velocity",

        "time_start_sec",
        "time_end_sec",

        "start_sec",
        "end_sec",
        "duration_sec",

        "start_frame60",
        "end_frame60",
        "duration_frames60",

        "note12",
        "expected_note",
        "expected_note_token",
        "note_token",
        "token",

        "octave_token",
        "step_token",

        "onset_group",
        "onset_notes",
        "onset_polyphony",

        "freq_hz",
        "semantic_layer",
    ]

    with open(args.out_events_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    max_polyphony = max((int(r["onset_polyphony"]) for r in rows), default=0)
    unique_onsets = len(set(r["onset_group"] for r in rows))

    meta = {
        "midi": args.midi,
        "out_events_csv": args.out_events_csv,
        "event_count": len(rows),
        "unique_onset_groups": unique_onsets,
        "max_onset_polyphony": max_polyphony,
        "anchor": "MIDI 69 = 9.A'- = A4 = 440 Hz",
        "time_grid": "frames60 = round(seconds * 60)",
        "onset_tolerance_sec": args.onset_tolerance_sec,
        "purpose": "MIDI reference only; no f0 inference; no harmonic hypotheses",
    }

    with open(args.out_meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("MIDI12 EVENTS FROM MIDO DONE")
    print(f"midi                 : {args.midi}")
    print(f"events_csv           : {args.out_events_csv}")
    print(f"meta_json            : {args.out_meta_json}")
    print(f"events               : {len(rows)}")
    print(f"unique_onset_groups  : {unique_onsets}")
    print(f"max_onset_polyphony  : {max_polyphony}")


if __name__ == "__main__":
    main()