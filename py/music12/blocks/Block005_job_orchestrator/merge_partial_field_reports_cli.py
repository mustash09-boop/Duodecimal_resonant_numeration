from __future__ import annotations

import argparse
import csv
import heapq
from pathlib import Path


def open_readers(paths: list[Path]):
    readers = []
    for p in paths:
        f = p.open("r", encoding="utf-8", newline="")
        readers.append((csv.DictReader(f), f))
    return readers


def merge_events_stream(segment_dirs: list[Path], out_csv: Path) -> int:
    paths = [d / "field_events.csv" for d in segment_dirs if (d / "field_events.csv").exists()]
    readers = open_readers(paths)

    heap = []
    for i, (reader, _) in enumerate(readers):
        row = next(reader, None)
        if row:
            heapq.heappush(
                heap,
                (
                    float(row.get("time_sec", 0.0)),
                    int(row.get("frame_index", 0)),
                    int(row.get("probe_index", 0)),
                    i,
                    row,
                ),
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = None

        while heap:
            _, _, _, i, row = heapq.heappop(heap)

            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()

            writer.writerow(row)
            count += 1

            reader, _ = readers[i]
            next_row = next(reader, None)
            if next_row:
                heapq.heappush(
                    heap,
                    (
                        float(next_row.get("time_sec", 0.0)),
                        int(next_row.get("frame_index", 0)),
                        int(next_row.get("probe_index", 0)),
                        i,
                        next_row,
                    ),
                )

    for _, f in readers:
        f.close()

    return count


def merge_trajectories_stream(segment_dirs: list[Path], out_csv: Path) -> int:
    paths = [d / "field_trajectories.csv" for d in segment_dirs if (d / "field_trajectories.csv").exists()]
    readers = open_readers(paths)

    heap = []
    for i, (reader, _) in enumerate(readers):
        row = next(reader, None)
        if row:
            heapq.heappush(
                heap,
                (
                    float(row.get("time_start_sec", 0.0)),
                    float(row.get("time_end_sec", 0.0)),
                    int(row.get("trajectory_id", 0)),
                    i,
                    row,
                ),
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    new_id = 1

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = None

        while heap:
            _, _, _, i, row = heapq.heappop(heap)

            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()

            row["trajectory_id"] = str(new_id)
            new_id += 1

            writer.writerow(row)
            count += 1

            reader, _ = readers[i]
            next_row = next(reader, None)
            if next_row:
                heapq.heappush(
                    heap,
                    (
                        float(next_row.get("time_start_sec", 0.0)),
                        float(next_row.get("time_end_sec", 0.0)),
                        int(next_row.get("trajectory_id", 0)),
                        i,
                        next_row,
                    ),
                )

    for _, f in readers:
        f.close()

    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conclusion_root", required=True)
    ap.add_argument("--segment_names", nargs="+", required=True)
    ap.add_argument("--out_events_csv", required=True)
    ap.add_argument("--out_trajectories_csv", required=True)
    args = ap.parse_args()

    root = Path(args.conclusion_root).resolve()
    seg_dirs = [root / s for s in args.segment_names if (root / s).exists()]

    print(f"segments={len(seg_dirs)}")

    events_count = merge_events_stream(seg_dirs, Path(args.out_events_csv))
    print(f"events_rows={events_count}")

    traj_count = merge_trajectories_stream(seg_dirs, Path(args.out_trajectories_csv))
    print(f"trajectories_rows={traj_count}")


if __name__ == "__main__":
    main()