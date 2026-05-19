# -*- coding: utf-8 -*-
"""
INSTRUMENT NOTE FILE INDEX

Создаёт таблицу соответствия:
инструмент / файл / нота

Не анализирует звук.
Не сравнивает инструменты.
Только строит честный индекс соответствий.
"""

import os
import re
import argparse
import pandas as pd


DIGITS12 = "123456789ABC"

NOTE12_RE = re.compile(r"(?P<note>[1-9ABC]+[.][1-9ABC](?:'[-ia0-9ABC]*)?-?)", re.IGNORECASE)
WESTERN_RE = re.compile(r"(?P<note>[A-G][#b]?\d)")


WESTERN_TO_DEGREE = {
    "C": "1",
    "C#": "2",
    "Db": "2",
    "D": "3",
    "D#": "4",
    "Eb": "4",
    "E": "5",
    "F": "6",
    "F#": "7",
    "Gb": "7",
    "G": "8",
    "G#": "9",
    "Ab": "9",
    "A": "A",
    "A#": "B",
    "Bb": "B",
    "B": "C",
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def western_to_note12(western):
    """
    A4 -> 9.A-
    C4 -> 9.1-
    Это согласовано с якорем A4 = 9.A-
    """
    m = re.match(r"^([A-G][#b]?)(\d)$", western)
    if not m:
        return ""

    name = m.group(1)
    octave = int(m.group(2))

    degree = WESTERN_TO_DEGREE.get(name)
    if not degree:
        return ""

    # В этой системе A4 = 9.A-, значит C4 тоже в 9-й 12-ричной октаве.
    note12_octave = octave + 5

    return f"{note12_octave}.{degree}-"


def extract_note_from_name(name):
    """
    Ищет сначала 12-ричную ноту:
      001_piano_midi_9.A-
      049_9.A-
      001_6.5'-_double-bass2_4string

    Потом западную:
      violin_A4_1_forte_arco-normal
    """
    s = str(name)

    m = NOTE12_RE.search(s)
    if m:
        return m.group("note").replace("'", "")

    m = WESTERN_RE.search(s)
    if m:
        return western_to_note12(m.group("note"))

    return ""


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--block004_root", required=True)
    ap.add_argument("--out_csv", required=True)

    args = ap.parse_args()

    ensure_dir(os.path.dirname(args.out_csv))

    rows = []

    for instrument in sorted(os.listdir(args.block004_root)):
        if instrument.lower() == "percussion":
            continue
        if instrument.startswith("_"):
            continue

        instrument_root = os.path.join(args.block004_root, instrument)
        spiral_dir = os.path.join(instrument_root, "50_spiral3d")

        if not os.path.isdir(spiral_dir):
            continue

        for f in sorted(os.listdir(spiral_dir)):
            if not f.endswith("__spiral3d_points.csv"):
                continue

            stem = f.replace("__spiral3d_points.csv", "")
            note12 = extract_note_from_name(stem)

            csv_path = os.path.join(spiral_dir, f)
            png_path = os.path.join(spiral_dir, f.replace("__spiral3d_points.csv", "__spiral3d.png"))
            html_path = os.path.join(spiral_dir, f.replace("__spiral3d_points.csv", "__spiral3d.html"))

            rows.append({
                "instrument": instrument,
                "source_note_name": stem,
                "canonical_note12": note12,
                "spiral3d_csv": csv_path,
                "spiral3d_png": png_path if os.path.exists(png_path) else "",
                "spiral3d_html": html_path if os.path.exists(html_path) else "",
                "parse_status": "OK" if note12 else "FAIL",
            })

    df = pd.DataFrame(rows)

    df.to_csv(args.out_csv, index=False)

    print("INSTRUMENT NOTE FILE INDEX DONE")
    print(f"rows : {len(df)}")
    print(f"ok   : {int((df['parse_status'] == 'OK').sum()) if len(df) else 0}")
    print(f"fail : {int((df['parse_status'] == 'FAIL').sum()) if len(df) else 0}")
    print(f"out  : {args.out_csv}")


if __name__ == "__main__":
    main()