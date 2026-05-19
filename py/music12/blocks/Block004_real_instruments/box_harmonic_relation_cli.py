from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def sf(v, d=0.0):
    try:
        return float(v)
    except:
        return d


def cents_delta(a, b):
    if a <= 0 or b <= 0:
        return 0
    return 1200 * math.log2(a / b)


def classify(cluster_hz, root_hz, max_harmonic=12):
    best_h = 0
    best_delta = 1e9

    for h in range(1, max_harmonic + 1):
        target = root_hz * h
        d = abs(cents_delta(cluster_hz, target))

        if d < best_delta:
            best_delta = d
            best_h = h

    if best_delta < 30:
        return "HARMONIC", best_h, best_delta

    if best_delta < 80:
        return "NEAR_HARMONIC", best_h, best_delta

    return "NON_HARMONIC", 0, best_delta


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--box_csv", required=True)
    ap.add_argument("--notes_csv", required=True)
    ap.add_argument("--out_csv", required=True)

    args = ap.parse_args()

    # === средний root по инструменту ===
    roots = []
    with open(args.notes_csv, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            hz = sf(row.get("root_hz"))
            if hz > 0:
                roots.append(hz)

    avg_root = sum(roots) / len(roots) if roots else 0

    rows = []

    with open(args.box_csv, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            hz = sf(row.get("cluster_center_hz"))

            cls, h, delta = classify(hz, avg_root)

            rows.append({
                "cluster_hz": hz,
                "token": row.get("dominant_token"),
                "percent_notes": row.get("percent_notes"),
                "mean_amp": row.get("mean_sum_amplitude"),
                "class": cls,
                "harmonic_index": h,
                "delta_cents": round(delta, 2),
            })

    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print("Done")


if __name__ == "__main__":
    main()