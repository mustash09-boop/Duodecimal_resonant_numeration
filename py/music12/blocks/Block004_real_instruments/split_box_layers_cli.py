from __future__ import annotations

import argparse
import csv
from pathlib import Path


def sf(v, d=0.0):
    try:
        return float(v)
    except:
        return d


def split_layers(rows):
    breath = []
    resonance = []

    for r in rows:
        hz = sf(r.get("cluster_center_hz"))

        if hz < 80:
            breath.append(r)
        else:
            resonance.append(r)

    return breath, resonance


def write_csv(path: Path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--box_csv", required=True)
    ap.add_argument("--out_breath_csv", required=True)
    ap.add_argument("--out_resonance_csv", required=True)

    args = ap.parse_args()

    box_csv = Path(args.box_csv)

    with box_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
        fieldnames = r.fieldnames

    breath, resonance = split_layers(rows)

    write_csv(Path(args.out_breath_csv), breath, fieldnames)
    write_csv(Path(args.out_resonance_csv), resonance, fieldnames)

    print(f"breath: {len(breath)}")
    print(f"resonance: {len(resonance)}")


if __name__ == "__main__":
    main()