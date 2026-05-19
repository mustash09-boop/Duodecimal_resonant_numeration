from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def load_times_csv(path: Path) -> list[float]:
    out: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        fieldnames = set(r.fieldnames or [])
        if "time_seconds" not in fieldnames and "time_sec" not in fieldnames:
            raise ValueError(
                f"Unsupported times CSV format in {path}. "
                f"Expected 'time_seconds' or 'time_sec', got {sorted(fieldnames)}"
            )

        for row in r:
            if "time_seconds" in row and str(row.get("time_seconds", "")).strip():
                out.append(float(row["time_seconds"]))
            elif "time_sec" in row and str(row.get("time_sec", "")).strip():
                out.append(float(row["time_sec"]))

    return out


def write_manifest_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plan distributed resonance-field jobs from merged probe times"
    )
    ap.add_argument("--times_csv", required=True)
    ap.add_argument("--jobs_root", required=True)
    ap.add_argument("--job_prefix", required=True)
    ap.add_argument("--num_jobs", type=int, required=True)

    ap.add_argument("--matrix_csv", required=True)
    ap.add_argument("--coords_delta_csv", required=True)
    ap.add_argument("--times_cloud_csv", required=True)
    ap.add_argument("--results_root", required=True)

    ap.add_argument("--energy_threshold", type=float, default=0.0)
    ap.add_argument("--top_k_per_frame", type=int, default=0)
    ap.add_argument("--max_time_gap_sec", type=float, default=0.05)
    ap.add_argument("--max_phase_gap_deg", type=float, default=3.0)
    ap.add_argument("--max_radial_gap", type=float, default=0.35)

    args = ap.parse_args()

    times_csv = Path(args.times_csv).resolve()
    jobs_root = Path(args.jobs_root).resolve()

    if args.num_jobs <= 0:
        raise SystemExit("--num_jobs must be > 0")

    times = load_times_csv(times_csv)
    if not times:
        raise SystemExit(f"No frames loaded from {times_csv}")

    n_frames = len(times)
    chunk = math.ceil(n_frames / args.num_jobs)

    jobs_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []

    for i in range(args.num_jobs):
        frame_start = i * chunk
        frame_end = min((i + 1) * chunk, n_frames)

        if frame_start >= frame_end:
            continue

        job_id = f"{args.job_prefix}_{i:04d}"
        job_dir = jobs_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        start_seconds = times[frame_start]
        end_seconds = times[frame_end - 1]

        out_dir_cloud = f"{args.results_root.rstrip('/')}/{job_id}"

        job = {
            "job_id": job_id,
            "task_type": "resonance_field_segment",
            "matrix_csv": args.matrix_csv,
            "coords_delta_csv": args.coords_delta_csv,
            "times_csv": args.times_cloud_csv,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "energy_threshold": args.energy_threshold,
            "top_k_per_frame": args.top_k_per_frame,
            "max_time_gap_sec": args.max_time_gap_sec,
            "max_phase_gap_deg": args.max_phase_gap_deg,
            "max_radial_gap": args.max_radial_gap,
            "out_dir": out_dir_cloud,
        }

        job_json = job_dir / "job.json"
        job_json.write_text(
            json.dumps(job, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        manifest_rows.append(
            {
                "job_id": job_id,
                "job_dir": str(job_dir),
                "job_json": str(job_json),
                "frame_start": frame_start,
                "frame_end": frame_end,
                "frame_count": frame_end - frame_start,
                "start_seconds": round(start_seconds, 6),
                "end_seconds": round(end_seconds, 6),
                "out_dir": out_dir_cloud,
            }
        )

    manifest_csv = jobs_root / "manifest_field_jobs.csv"
    write_manifest_csv(manifest_csv, manifest_rows)

    summary = {
        "times_csv": str(times_csv),
        "jobs_root": str(jobs_root),
        "job_prefix": args.job_prefix,
        "num_jobs_requested": args.num_jobs,
        "num_jobs_created": len(manifest_rows),
        "n_frames": n_frames,
        "chunk_frames": chunk,
        "manifest_csv": str(manifest_csv),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()