from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


STEP_INDEX = {
    "1": 0,
    "2": 1,
    "3": 2,
    "4": 3,
    "5": 4,
    "6": 5,
    "7": 6,
    "8": 7,
    "9": 8,
    "A": 9,
    "B": 10,
    "C": 11,
}


def parse_note_coord(note: str) -> tuple[int, str]:
    """
    Поддерживает строки вида:
        9.1'-
        8.B'-
        7.5'-{1}
        9.1'-{1.2}

    Нам для спирали сейчас нужны только:
        octave
        base step
    """
    note = note.strip()

    if "." not in note:
        raise ValueError(f"Invalid note format (no '.'): {note}")

    octave_str, rest = note.split(".", 1)
    octave = int(octave_str)

    # step token идёт в начале rest
    # может быть:
    #   1'-
    #   A'-
    #   B'-{1}
    #   10'-
    if rest.startswith("10"):
        step = "10"
    else:
        step = rest[0]

    if step not in STEP_INDEX:
        raise ValueError(f"Unsupported step token in note: {note}")

    return octave, step


def note_to_xy(note: str) -> tuple[float, float]:
    octave, step = parse_note_coord(note)

    idx = STEP_INDEX[step]
    angle = 2.0 * math.pi * idx / 12.0
    radius = float(octave)

    x = radius * math.cos(angle)
    y = radius * math.sin(angle)

    return x, y


def load_events(path: Path):
    events = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        required = {"time_start", "time_end", "note"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

        for r in reader:
            events.append(
                {
                    "time": float(r["time_start"]),
                    "note": r["note"],
                }
            )

    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice_events_csv", required=True)
    parser.add_argument("--out_png", required=True)
    args = parser.parse_args()

    events = load_events(Path(args.voice_events_csv))

    xs = []
    ys = []

    for e in events:
        x, y = note_to_xy(e["note"])
        xs.append(x)
        ys.append(y)

    plt.figure(figsize=(8, 8))
    plt.plot(xs, ys, "-o", markersize=3)
    plt.title("Spiral Melody Trajectory")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)

    out_path = Path(args.out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()