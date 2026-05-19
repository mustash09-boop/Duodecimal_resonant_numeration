from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_times_csv(path: Path) -> list[float]:
    out: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if "time_seconds" in row:
                out.append(float(row["time_seconds"]))
            elif "time_sec" in row:
                out.append(float(row["time_sec"]))
            else:
                raise ValueError(f"Unsupported times CSV columns in {path}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plan frame segments for distributed resonance field processing"
    )
    ap.add_argument("--times_csv", required=True)
    ap.add_argument("--segments_dir", required=True)
    ap.add_argument("--out_manifest_csv", required=True)
    ap.add_argument("--segment_seconds", type=float, default=5.0)
    ap.add_argument("--overlap_seconds", type=float, default=0.08)
    args = ap.parse_args()

    times_csv = Path(args.times_csv).resolve()
    segments_dir = Path(args.segments_dir).resolve()
    out_manifest_csv = Path(args.out_manifest_csv).resolve()

    times = load_times_csv(times_csv)
    if not times:
        raise SystemExit(f"No times loaded from {times_csv}")

    segments_dir.mkdir(parents=True, exist_ok=True)
    out_manifest_csv.parent.mkdir(parents=True, exist_ok=True)

    step_seconds = args.segment_seconds - args.overlap_seconds
    if step_seconds <= 0:
        raise SystemExit("segment_seconds must be > overlap_seconds")

    last_time = times[-1]
    segment_starts: list[float] = []
    t = 0.0
    while t <= last_time + 1e-9:
        segment_starts.append(round(t, 6))
        t += step_seconds

    rows: list[dict[str, object]] = []

    for seg_idx, start_sec in enumerate(segment_starts):
        end_sec = start_sec + args.segment_seconds

        frame_indices = [
            i for i, ts in enumerate(times)
            if start_sec <= ts < end_sec
        ]
        if not frame_indices:
            continue

        frame_start = min(frame_indices)
        frame_end_exclusive = max(frame_indices) + 1

        seg_name = f"seg_{seg_idx:04d}"
        seg_dir = segments_dir / seg_name
        seg_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "segment_name": seg_name,
            "segment_index": seg_idx,
            "start_seconds": start_sec,
            "end_seconds": round(end_sec, 6),
            "frame_start": frame_start,
            "frame_end_exclusive": frame_end_exclusive,
            "frame_count": frame_end_exclusive - frame_start,
        }
        (seg_dir / "segment_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        rows.append(
            {
                "segment_name": seg_name,
                "segment_index": seg_idx,
                "start_seconds": start_sec,
                "end_seconds": round(end_sec, 6),
                "frame_start": frame_start,
                "frame_end_exclusive": frame_end_exclusive,
                "frame_count": frame_end_exclusive - frame_start,
                "segment_dir": str(seg_dir),
                "meta_json": str(seg_dir / "segment_meta.json"),
                "events_csv": str(seg_dir / "events.csv"),
                "trajectories_csv": str(seg_dir / "trajectories.csv"),
            }
        )

    with out_manifest_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(json.dumps(
        {
            "times_csv": str(times_csv),
            "segment_seconds": args.segment_seconds,
            "overlap_seconds": args.overlap_seconds,
            "segment_count": len(rows),
            "out_manifest_csv": str(out_manifest_csv),
            "segments_dir": str(segments_dir),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()