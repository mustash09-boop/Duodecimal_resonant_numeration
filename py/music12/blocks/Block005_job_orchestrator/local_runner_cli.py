from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

from .manifest_io import read_json, write_json, job_paths
from .models import JobSpec, JobStatus, JobResult


def utc_now() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def build_probe_command(project_root: str, spec: JobSpec) -> list[str]:
    out_dir = Path(spec.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "music12.demons.demon_maxwell_cli",
        "-m",
        "music12.blocks.Block002_audio_recogn.resonance_probe12_scan_cli",
        "--task-class",
        "module_run",
        "--project-root",
        project_root,
        "--logdir",
        spec.logdir,
        "--tag",
        spec.maxwell_tag or spec.job_id,
        "--",
        "--wav",
        spec.wav_path,
        "--out_matrix_csv",
        str(out_dir / "probe_matrix.csv"),
        "--out_meta_json",
        str(out_dir / "probe_meta.json"),
        "--out_times_csv",
        str(out_dir / "probe_times.csv"),
        "--out_coords_csv",
        str(out_dir / "probe_coords.csv"),
        "--octave_min",
        spec.octave_min,
        "--octave_max",
        spec.octave_max,
        "--detail_depth",
        str(spec.detail_depth),
        "--projection_depth",
        str(spec.projection_depth),
        "--time_step_seconds",
        str(spec.time_step_seconds),
        "--window_seconds",
        str(spec.window_seconds),
        "--harmonic_weights",
        ",".join(str(x) for x in spec.harmonic_weights),
    ]

    if spec.time_start is not None:
        cmd.extend(["--time_start", str(spec.time_start)])
    if spec.time_end is not None:
        cmd.extend(["--time_end", str(spec.time_end)])

    for k, v in spec.extra_args.items():
        cmd.extend([f"--{k}", str(v)])

    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one Block005 job locally via Maxwell.")
    ap.add_argument("--project_root", required=True)
    ap.add_argument("--jobs_root", required=True)
    ap.add_argument("--job_id", required=True)
    ap.add_argument("--timeout_sec", type=int, default=7200)
    args = ap.parse_args()

    paths = job_paths(args.jobs_root, args.job_id)
    spec = read_json(paths["spec"], JobSpec)

    status = JobStatus(
        job_id=spec.job_id,
        state="running",
        provider="local",
        created_at=utc_now(),
        started_at=utc_now(),
        message="started local Maxwell run",
    )
    write_json(paths["status"], status)

    cmd = build_probe_command(args.project_root, spec)

    with open(paths["stdout"], "w", encoding="utf-8") as fout, open(
        paths["stderr"], "w", encoding="utf-8"
    ) as ferr:
        try:
            proc = subprocess.run(
                cmd,
                cwd=args.project_root,
                stdout=fout,
                stderr=ferr,
                timeout=args.timeout_sec,
                check=False,
            )
            rc = proc.returncode
            state = "done" if rc == 0 else "failed"
            msg = f"finished with return_code={rc}"
        except subprocess.TimeoutExpired:
            rc = None
            state = "timeout"
            msg = f"timeout after {args.timeout_sec} sec"

    status.state = state
    status.finished_at = utc_now()
    status.return_code = rc
    status.message = msg
    write_json(paths["status"], status)

    out_dir = Path(spec.out_dir)
    result = JobResult(
        job_id=spec.job_id,
        state=state,
        probe_matrix_csv=str(out_dir / "probe_matrix.csv") if (out_dir / "probe_matrix.csv").exists() else None,
        probe_meta_json=str(out_dir / "probe_meta.json") if (out_dir / "probe_meta.json").exists() else None,
        probe_times_csv=str(out_dir / "probe_times.csv") if (out_dir / "probe_times.csv").exists() else None,
        probe_coords_csv=str(out_dir / "probe_coords.csv") if (out_dir / "probe_coords.csv").exists() else None,
        maxwell_report_json=None,
        maxwell_report_txt=None,
        notes=msg,
    )
    write_json(paths["result"], result)

    print(f"job_id={spec.job_id}")
    print(f"state={state}")
    print(msg)


if __name__ == "__main__":
    main()