# -*- coding: utf-8 -*-
"""
PERCUSSION MANIFEST BUILDER

Для ударных файлов вида:
  agogo-bells__025_mezzo-forte_struck-singly.wav
  triangle__phrase_mezzo-piano_rhythm.wav
  suspended-cymbal__very-long_cresc-decresc_roll.wav

Создаёт event-manifest без expected_note.
"""

import os
import csv
import argparse


DYNAMICS = {
    "pianissimo",
    "piano",
    "mezzo-piano",
    "mezzo-forte",
    "forte",
    "fortissimo",
    "crescendo",
    "decrescendo",
    "cresc-decresc",
}


def detect_gesture_type(event_id, articulation):
    event_id = (event_id or "").lower()
    articulation = (articulation or "").lower()

    if event_id in {"phrase"}:
        return "phrase"
    if event_id in {"long", "very-long"}:
        return "sustain"
    if articulation in {"roll", "rhythm", "glissando"}:
        return articulation
    return "single"


def parse_percussion_filename(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]

    if "__" not in stem:
        return None

    instrument_name, rest = stem.split("__", 1)
    parts = rest.split("_")

    if len(parts) < 3:
        return None

    event_id = parts[0]
    dynamic = parts[1]
    articulation = "_".join(parts[2:])

    return {
        "instrument_name": instrument_name,
        "event_id": event_id,
        "dynamic": dynamic,
        "articulation": articulation,
        "gesture_type": detect_gesture_type(event_id, articulation),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_core_list", required=True)
    parser.add_argument("--instrument_family", default="percussion")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_core_list), exist_ok=True)

    rows = []
    bad = []

    for root, _, files in os.walk(args.input_dir):
        for f in files:
            if not f.lower().endswith(".wav"):
                continue

            wav_path = os.path.join(root, f)
            parsed = parse_percussion_filename(f)

            if parsed is None:
                bad.append(wav_path)
                continue

            row = {
                "original_filename": f,
                "wav_path": wav_path,
                "instrument_family": args.instrument_family,
                "instrument_name": parsed["instrument_name"],
                "event_id": parsed["event_id"],
                "dynamic": parsed["dynamic"],
                "articulation": parsed["articulation"],
                "gesture_type": parsed["gesture_type"],
                "semantic_layer": "percussion_events",
                "expected_note": "",
                "expected_note_token": "",
                "source_type": "percussion_event",
            }

            rows.append(row)

    fieldnames = [
        "original_filename",
        "wav_path",
        "instrument_family",
        "instrument_name",
        "event_id",
        "dynamic",
        "articulation",
        "gesture_type",
        "semantic_layer",
        "expected_note",
        "expected_note_token",
        "source_type",
    ]

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(args.out_core_list, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(r["wav_path"] + "\n")

    bad_txt = os.path.splitext(args.out_csv)[0] + "_bad_files.txt"
    with open(bad_txt, "w", encoding="utf-8") as f:
        for p in bad:
            f.write(p + "\n")

    print("PERCUSSION MANIFEST DONE")
    print(f"input_dir     : {args.input_dir}")
    print(f"out_csv       : {args.out_csv}")
    print(f"files         : {len(rows)}")
    print(f"bad_files     : {len(bad)}")


if __name__ == "__main__":
    main()