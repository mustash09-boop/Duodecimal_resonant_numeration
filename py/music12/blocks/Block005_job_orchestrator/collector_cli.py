from __future__ import annotations

import argparse

from .collector_core import collect_rows
from .manifest_io import read_manifest_csv, write_manifest_csv


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect Block005 job states into manifest.")
    ap.add_argument("--jobs_root", required=True)
    ap.add_argument("--manifest_csv", required=True)
    args = ap.parse_args()

    rows = read_manifest_csv(args.manifest_csv)
    rows = collect_rows(args.jobs_root, rows)
    write_manifest_csv(args.manifest_csv, rows)

    counts = {}
    for row in rows:
        counts[row.state] = counts.get(row.state, 0) + 1

    for k in sorted(counts):
        print(f"{k}={counts[k]}")


if __name__ == "__main__":
    main()