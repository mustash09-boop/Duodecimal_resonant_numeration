from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


DIGITS12 = "123456789ABC"
ANCHOR_TOKEN = "9.A-"
ANCHOR_HZ = 440.0


NOTE_INDEX_FROM_A4 = {
    "C": -9,
    "Cs": -8,
    "D": -7,
    "Ds": -6,
    "E": -5,
    "F": -4,
    "Fs": -3,
    "G": -2,
    "Gs": -1,
    "A": 0,
    "As": 1,
    "B": 2,
}


def int_to_bij12(n: int) -> str:
    if n <= 0:
        raise ValueError(f"12-radix octave must be positive, got {n}")

    out = []
    while n > 0:
        n, r = divmod(n - 1, 12)
        out.append(DIGITS12[r])
    return "".join(reversed(out))


def western_to_hz(note: str) -> float:
    """
    Examples:
      A4  -> 440.0
      As4 -> A#4 / Bb4
      Cs6 -> C#6 / Db6
    """
    m = re.fullmatch(r"([A-G]s?)(\d+)", note)
    if not m:
        return 0.0

    pitch, octave_s = m.groups()
    octave = int(octave_s)

    if pitch not in NOTE_INDEX_FROM_A4:
        return 0.0

    semitone_from_a4 = NOTE_INDEX_FROM_A4[pitch] + (octave - 4) * 12
    return ANCHOR_HZ * (2.0 ** (semitone_from_a4 / 12.0))


def hz_to_12(freq_hz: float) -> str:
    """
    Anchor:
      A4 = 440 Hz = 9.A-

    Uses frequency as truth, not Western octave arithmetic.
    """
    if freq_hz <= 0:
        return ""

    semitone_offset = 12.0 * math.log2(freq_hz / ANCHOR_HZ)

    # nearest 12-TET step around anchor
    nearest = int(round(semitone_offset))

    anchor_octave = 9
    anchor_degree0 = 9  # A is the 10th symbol, zero-based index 9

    total_degree0 = anchor_degree0 + nearest

    octave_shift, degree0 = divmod(total_degree0, 12)
    octave12 = anchor_octave + octave_shift

    return f"{int_to_bij12(octave12)}.{DIGITS12[degree0]}-"


def western_to_12(note: str) -> str:
    hz = western_to_hz(note)
    return hz_to_12(hz)


def parse_filename(name: str) -> dict:
    stem = Path(name).stem
    ext = Path(name).suffix.lower()

    parts = stem.split("_")
    if len(parts) < 5:
        return {
            "original_filename": name,
            "new_filename": "",
            "instrument": "",
            "source_note": "",
            "note12": "",
            "source_hz": "",
            "duration_or_type": "",
            "dynamic": "",
            "technique": "",
            "semantic_layer": "",
            "is_phrase": 0,
            "is_special_technique": 0,
            "is_core_single": 0,
            "parse_status": "BAD_FORMAT",
        }

    instrument = parts[0]
    source_note = parts[1]
    duration_or_type = parts[2]
    dynamic = parts[3]
    technique = "_".join(parts[4:])

    source_hz = western_to_hz(source_note)
    note12 = western_to_12(source_note)

    is_phrase = duration_or_type in {"phrase", "long", "very-long"}

    is_special = any(x in technique for x in [
        "harmonic",
        "glissando",
        "tremolo",
        "pizz",
        "col-legno",
        "sul-ponticello",
        "sul-tasto",
        "con-sord",
        "non-vibrato",
        "molto-vibrato",
        "punta-darco",
        "punta-d'arco",
        "martele",
        "spiccato",
        "staccato",
        "detache",
        "tenuto",
        "au-talon",
    ])

    is_core_single = (
        not is_phrase
        and technique == "arco-normal"
        and dynamic in {"mezzo-forte", "forte"}
        and duration_or_type in {"05", "1", "15"}
    )

    semantic_layer = "02_variants"

    if is_core_single:
        semantic_layer = "01_core_notes"
    elif is_phrase:
        semantic_layer = "05_phrases"
    elif "harmonic" in technique:
        semantic_layer = "04_harmonics"
    elif is_special:
        semantic_layer = "06_special_effects"

    safe_tech = technique.replace("'", "")

    new_filename = (
        f"{instrument}__{note12}__src-{source_note}"
        f"__hz-{source_hz:.6f}"
        f"__dur-{duration_or_type}"
        f"__dyn-{dynamic}"
        f"__tech-{safe_tech}{ext}"
    )

    return {
        "original_filename": name,
        "new_filename": new_filename,
        "instrument": instrument,
        "source_note": source_note,
        "note12": note12,
        "source_hz": f"{source_hz:.6f}" if source_hz > 0 else "",
        "duration_or_type": duration_or_type,
        "dynamic": dynamic,
        "technique": technique,
        "semantic_layer": semantic_layer,
        "is_phrase": int(is_phrase),
        "is_special_technique": int(is_special),
        "is_core_single": int(is_core_single),
        "parse_status": "OK" if note12 else "BAD_NOTE",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build 12-radix manifest for violin sample filenames.")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_core_list", required=True)
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_core_list = Path(args.out_core_list).resolve()

    files = sorted([
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".wav", ".mp3"}
    ])

    rows = [parse_filename(p.name) for p in files]

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "original_filename",
        "new_filename",
        "instrument",
        "source_note",
        "note12",
        "source_hz",
        "duration_or_type",
        "dynamic",
        "technique",
        "semantic_layer",
        "is_phrase",
        "is_special_technique",
        "is_core_single",
        "parse_status",
    ]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    core = [
        r for r in rows
        if r.get("is_core_single") == 1 and r.get("parse_status") == "OK"
    ]

    out_core_list.parent.mkdir(parents=True, exist_ok=True)
    out_core_list.write_text(
        "\n".join(r["original_filename"] for r in core),
        encoding="utf-8",
    )

    print(f"Files total : {len(rows)}")
    print(f"Core singles: {len(core)}")
    print(f"Manifest    : {out_csv}")
    print(f"Core list   : {out_core_list}")


if __name__ == "__main__":
    main()