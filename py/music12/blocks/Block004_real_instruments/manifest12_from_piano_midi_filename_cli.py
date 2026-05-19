# -*- coding: utf-8 -*-
"""
MANIFEST12 FROM KEYED / PIANO-LIKE FILENAME

Под общий формат:
  001_piano_midi_5.A-.wav
  001_piano_real_5.A-.wav
  001_celesta_5.A-.wav
  001_harpsichord_5.A-.wav

То есть:
  NNN_SOURCE_TAG_NOTE12.wav

Создаёт manifest, совместимый с instrument_pipeline_runner_cli.

Критичные поля:
  original_filename
  note12
  semantic_layer
  parse_status
"""

import os
import re
import csv
import argparse


RE_NAME = re.compile(
    r"^(?P<idx>\d{3})_(?P<source_tag>[A-Za-z0-9_]+)_(?P<token>[1-9ABC]+[.][1-9ABC][ia0-9ABC'\-]*)[.]wav$",
    re.IGNORECASE,
)


def parse_name(filename):
    m = RE_NAME.match(filename)
    if not m:
        return None

    return {
        "index": m.group("idx"),
        "source_tag": m.group("source_tag"),
        "note12": m.group("token"),
    }


def make_row_ok(filename, wav_path, instrument_name, parsed):
    token = parsed["note12"]

    return {
        "original_filename": filename,
        "filename": filename,
        "wav_path": wav_path,
        "audio_path": wav_path,
        "path": wav_path,

        "index": parsed["index"],
        "instrument_name": instrument_name,
        "source_tag": parsed["source_tag"],

        "note12": token,
        "expected_note": token,
        "expected_note_token": token,
        "note_token": token,
        "token": token,

        "layer": "all",
        "semantic_layer": "all",
        "source_type": "keyed_note",

        "parse_status": "OK",
        "reason": "",
    }


def make_row_fail(filename, wav_path, instrument_name):
    return {
        "original_filename": filename,
        "filename": filename,
        "wav_path": wav_path,
        "audio_path": wav_path,
        "path": wav_path,

        "index": "",
        "instrument_name": instrument_name,
        "source_tag": "",

        "note12": "",
        "expected_note": "",
        "expected_note_token": "",
        "note_token": "",
        "token": "",

        "layer": "all",
        "semantic_layer": "all",
        "source_type": "keyed_note",

        "parse_status": "FAIL",
        "reason": "filename does not match NNN_SOURCE_TAG_NOTE12.wav",
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--instrument_name", default="piano_like")

    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    rows = []

    for filename in sorted(os.listdir(args.audio_dir)):
        if not filename.lower().endswith(".wav"):
            continue

        wav_path = os.path.join(args.audio_dir, filename)
        parsed = parse_name(filename)

        if parsed is None:
            rows.append(make_row_fail(filename, wav_path, args.instrument_name))
        else:
            rows.append(make_row_ok(filename, wav_path, args.instrument_name, parsed))

    fieldnames = [
        "original_filename",
        "filename",
        "wav_path",
        "audio_path",
        "path",

        "index",
        "instrument_name",
        "source_tag",

        "note12",
        "expected_note",
        "expected_note_token",
        "note_token",
        "token",

        "layer",
        "semantic_layer",
        "source_type",

        "parse_status",
        "reason",
    ]

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(1 for r in rows if r["parse_status"] == "OK")
    fail = sum(1 for r in rows if r["parse_status"] != "OK")

    print("KEYED / PIANO-LIKE MANIFEST DONE")
    print(f"audio_dir : {args.audio_dir}")
    print(f"out_csv   : {args.out_csv}")
    print(f"rows      : {len(rows)}")
    print(f"ok        : {ok}")
    print(f"fail      : {fail}")


if __name__ == "__main__":
    main()