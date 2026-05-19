from __future__ import annotations

import argparse
from pathlib import Path

from .manifest_io import write_json, write_manifest_csv, job_paths
from .planner_core import build_segment_jobs


def parse_harmonic_weights(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plan Block005 jobs for segmented resonance probe runs."
    )
    ap.add_argument("--wav", required=True)
    ap.add_argument("--jobs_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--manifest_csv", required=True)

    ap.add_argument("--total_duration_sec", required=True, type=float)
    ap.add_argument("--segment_seconds", required=True, type=float)
    ap.add_argument("--overlap_seconds", default=0.08, type=float)

    ap.add_argument("--octave_min", default="5")
    ap.add_argument("--octave_max", default="C")
    ap.add_argument("--detail_depth", default=2, type=int)
    ap.add_argument("--projection_depth", default=2, type=int)
    ap.add_argument("--time_step_seconds", default=1.0 / 60.0, type=float)
    ap.add_argument("--window_seconds", default=0.08, type=float)
    ap.add_argument(
        "--harmonic_weights",
        default="1.0,0.5,0.3,0.2,0.12,0.08,0.05,0.03",
    )
    ap.add_argument("--job_prefix", default="probe_seg")

    args = ap.parse_args()

    weights = parse_harmonic_weights(args.harmonic_weights)

    jobs, manifest = build_segment_jobs(
        wav_path=args.wav,
        jobs_root=args.jobs_root,
        out_root=args.out_root,
        total_duration_sec=args.total_duration_sec,
        segment_seconds=args.segment_seconds,
        overlap_seconds=args.overlap_seconds,
        octave_min=args.octave_min,
        octave_max=args.octave_max,
        detail_depth=args.detail_depth,
        projection_depth=args.projection_depth,
        time_step_seconds=args.time_step_seconds,
        window_seconds=args.window_seconds,
        harmonic_weights=weights,
        job_prefix=args.job_prefix,
    )

    Path(args.jobs_root).mkdir(parents=True, exist_ok=True)

    for job in jobs:
        paths = job_paths(args.jobs_root, job.job_id)
        paths["root"].mkdir(parents=True, exist_ok=True)
        write_json(paths["spec"], job)

    write_manifest_csv(args.manifest_csv, manifest)

    print(f"planned_jobs={len(jobs)}")
    print(f"jobs_root={args.jobs_root}")
    print(f"manifest_csv={args.manifest_csv}")


if __name__ == "__main__":
    main()