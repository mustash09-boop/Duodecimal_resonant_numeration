from __future__ import annotations

from pathlib import Path
from typing import List

from .models import JobSpec, JobManifestRow


def build_segment_jobs(
    *,
    wav_path: str,
    jobs_root: str,
    out_root: str,
    total_duration_sec: float,
    segment_seconds: float,
    overlap_seconds: float,
    octave_min: str,
    octave_max: str,
    detail_depth: int,
    projection_depth: int,
    time_step_seconds: float,
    window_seconds: float,
    harmonic_weights: list[float],
    task_kind: str = "resonance_probe_segment",
    job_prefix: str = "job",
) -> tuple[list[JobSpec], list[JobManifestRow]]:
    if segment_seconds <= 0:
        raise ValueError("segment_seconds must be > 0")
    if overlap_seconds < 0:
        raise ValueError("overlap_seconds must be >= 0")
    if total_duration_sec <= 0:
        raise ValueError("total_duration_sec must be > 0")

    jobs: List[JobSpec] = []
    manifest: List[JobManifestRow] = []

    step = max(segment_seconds - overlap_seconds, 0.001)
    idx = 0
    t0 = 0.0

    while t0 < total_duration_sec:
        t1 = min(t0 + segment_seconds, total_duration_sec)

        job_id = f"{job_prefix}_{idx:04d}"
        out_dir = str(Path(out_root) / job_id)

        spec = JobSpec(
            job_id=job_id,
            task_kind=task_kind,
            wav_path=wav_path,
            out_dir=out_dir,
            time_start=round(t0, 6),
            time_end=round(t1, 6),
            octave_min=octave_min,
            octave_max=octave_max,
            detail_depth=detail_depth,
            projection_depth=projection_depth,
            time_step_seconds=time_step_seconds,
            window_seconds=window_seconds,
            harmonic_weights=harmonic_weights,
            maxwell_tag=job_id,
        )
        jobs.append(spec)

        manifest.append(
            JobManifestRow(
                job_id=job_id,
                task_kind=task_kind,
                wav_path=wav_path,
                out_dir=out_dir,
                state="queued",
                provider="local",
                notes=f"segment {t0:.3f}-{t1:.3f}",
            )
        )

        idx += 1
        t0 += step

    return jobs, manifest