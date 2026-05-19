from __future__ import annotations

import argparse
import csv
from collections import defaultdict, Counter
from pathlib import Path

from music12.core.harmonic_alphabet12 import harmonic_token_from_root


# ============================================================
# HELPERS
# ============================================================

def parse_harmonics(value: str) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in value.split() if x.strip()]


def extract_root_from_folder(folder_name: str) -> str:
    """
    001__RealPiano_1__5.A-  →  5.A-
    """
    parts = folder_name.split("__")
    if len(parts) >= 3:
        return parts[2]
    return folder_name


def build_ideal_chain(root: str, max_h: int = 8) -> set[str]:
    chain = set()
    for h in range(1, max_h + 1):
        try:
            chain.add(harmonic_token_from_root(root, h))
        except Exception:
            continue
    return chain


# ============================================================
# CORE
# ============================================================

def build_response(input_dir: Path):
    per_note_residuals = defaultdict(Counter)
    global_residuals = Counter()

    folders = [p for p in input_dir.iterdir() if p.is_dir()]

    for folder in folders:
        root = extract_root_from_folder(folder.name)

        csv_files = list(folder.glob("*__stabilized__with_phase.csv"))
        if not csv_files:
            continue

        ideal_chain = build_ideal_chain(root)

        for f in csv_files:
            with f.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)

                for row in reader:
                    harmonics = []

                    if "dominant_harmonics" in row:
                        harmonics += parse_harmonics(row["dominant_harmonics"])

                    if "matched_harmonics_window" in row:
                        harmonics += parse_harmonics(row["matched_harmonics_window"])

                    for h in harmonics:
                        if h not in ideal_chain:
                            per_note_residuals[root][h] += 1
                            global_residuals[h] += 1

    return per_note_residuals, global_residuals


# ============================================================
# WRITE
# ============================================================

def write_per_note(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["root", "residual_harmonic", "count"])

        for root in sorted(data.keys()):
            for h, count in data[root].most_common():
                writer.writerow([root, h, count])


def write_global(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["harmonic", "count"])

        for h, count in data.most_common():
            writer.writerow([h, count])


def write_txt(path: Path, per_note, global_data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        f.write("INSTRUMENT RESONANCE RESPONSE\n")
        f.write("=" * 60 + "\n\n")

        f.write("GLOBAL RESONANCE COMPONENTS\n")
        for h, count in global_data.most_common(20):
            f.write(f"{h}: {count}\n")

        f.write("\nPER NOTE RESPONSE\n")
        for root in sorted(per_note.keys()):
            f.write(f"\n[{root}]\n")
            for h, count in per_note[root].most_common(10):
                f.write(f"  {h}: {count}\n")


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_per_note", required=True)
    ap.add_argument("--out_global", required=True)
    ap.add_argument("--out_txt", required=True)
    args = ap.parse_args()

    per_note, global_data = build_response(Path(args.input_dir))

    write_per_note(Path(args.out_per_note), per_note)
    write_global(Path(args.out_global), global_data)
    write_txt(Path(args.out_txt), per_note, global_data)

    print("instrument resonance response built")


if __name__ == "__main__":
    main()