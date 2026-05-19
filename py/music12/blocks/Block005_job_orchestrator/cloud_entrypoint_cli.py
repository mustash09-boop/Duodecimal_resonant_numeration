from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_probe_command(project_root: Path, spec: dict[str, Any], results_dir: Path) -> list[str]:
    results_dir.mkdir(parents=True, exist_ok=True)

    harmonic_weights = spec.get(
        "harmonic_weights",
        [1.0, 0.5, 0.3, 0.2, 0.12, 0.08, 0.05, 0.03],
    )
    harmonic_weights_str = ",".join(str(x) for x in harmonic_weights)

    cmd = [
        sys.executable,
        "-m",
        "music12.demons.demon_maxwell_cli",
        "-m",
        "music12.blocks.Block002_audio_recogn.resonance_probe12_scan_cli",
        "--task-class",
        "module_run",
        "--project-root",
        str(project_root),
        "--logdir",
        str(spec.get("logdir", "_demon_logs")),
        "--tag",
        str(spec.get("maxwell_tag") or spec["job_id"]),
        "--",
        "--wav",
        str(spec["wav_path"]),
        "--out_matrix_csv",
        str(results_dir / "probe_matrix.csv"),
        "--out_meta_json",
        str(results_dir / "probe_meta.json"),
        "--out_times_csv",
        str(results_dir / "probe_times.csv"),
        "--out_coords_csv",
        str(results_dir / "probe_coords.csv"),
        "--octave_min",
        str(spec.get("octave_min", "5")),
        "--octave_max",
        str(spec.get("octave_max", "C")),
        "--detail_depth",
        str(spec.get("detail_depth", 2)),
        "--projection_depth",
        str(spec.get("projection_depth", 2)),
        "--time_step_seconds",
        str(spec.get("time_step_seconds", 1.0 / 60.0)),
        "--window_seconds",
        str(spec.get("window_seconds", 0.08)),
        "--harmonic_weights",
        harmonic_weights_str,
    ]

    attack_portion = spec.get("attack_portion")
    if attack_portion is not None:
        cmd.extend(["--attack_portion", str(attack_portion)])

    decay_portion = spec.get("decay_portion")
    if decay_portion is not None:
        cmd.extend(["--decay_portion", str(decay_portion)])

    time_start = spec.get("time_start")
    if time_start is not None:
        cmd.extend(["--time_start", str(time_start)])

    time_end = spec.get("time_end")
    if time_end is not None:
        cmd.extend(["--time_end", str(time_end)])

    extra_args = spec.get("extra_args", {})
    for k, v in extra_args.items():
        cmd.extend([f"--{k}", str(v)])

    return cmd


def build_field_command(spec_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "music12.blocks.Block005_job_orchestrator.field_from_job_cli",
        "--job_json",
        str(spec_path),
    ]


def find_latest_maxwell_reports(logdir: Path, tag: str) -> tuple[str | None, str | None]:
    if not logdir.exists():
        return None, None

    json_candidates = sorted(
        logdir.glob(f"{tag}*maxwell_report.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    txt_candidates = sorted(
        logdir.glob(f"{tag}*maxwell_report.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    json_path = str(json_candidates[0]) if json_candidates else None
    txt_path = str(txt_candidates[0]) if txt_candidates else None
    return json_path, txt_path


def build_command(project_root: Path, spec: dict[str, Any], job_spec_path: Path, results_dir: Path) -> list[str]:
    task_kind = str(spec.get("task_kind", "")).strip()

    if task_kind == "resonance_probe_segment":
        return build_probe_command(project_root=project_root, spec=spec, results_dir=results_dir)

    if task_kind == "resonance_field_segment":
        return build_field_command(job_spec_path)

    raise ValueError(f"Unknown task_kind: {task_kind!r}")


def build_result_payload(project_root: Path, spec: dict[str, Any], results_dir: Path, state: str, message: str) -> dict[str, Any]:
    task_kind = str(spec.get("task_kind", "")).strip()
    tag = str(spec.get("maxwell_tag") or spec["job_id"])
    logdir = project_root / str(spec.get("logdir", "_demon_logs"))
    maxwell_json, maxwell_txt = find_latest_maxwell_reports(logdir=logdir, tag=tag)

    if task_kind == "resonance_probe_segment":
        probe_matrix = results_dir / "probe_matrix.csv"
        probe_meta = results_dir / "probe_meta.json"
        probe_times = results_dir / "probe_times.csv"
        probe_coords = results_dir / "probe_coords.csv"

        return {
            "job_id": spec["job_id"],
            "task_kind": task_kind,
            "state": state,
            "probe_matrix_csv": str(probe_matrix) if probe_matrix.exists() else None,
            "probe_meta_json": str(probe_meta) if probe_meta.exists() else None,
            "probe_times_csv": str(probe_times) if probe_times.exists() else None,
            "probe_coords_csv": str(probe_coords) if probe_coords.exists() else None,
            "maxwell_report_json": maxwell_json,
            "maxwell_report_txt": maxwell_txt,
            "notes": message,
        }

    if task_kind == "resonance_field_segment":
        field_events = results_dir / "field_events.csv"
        field_trajectories = results_dir / "field_trajectories.csv"
        field_meta = results_dir / "field_meta.json"

        return {
            "job_id": spec["job_id"],
            "task_kind": task_kind,
            "state": state,
            "field_events_csv": str(field_events) if field_events.exists() else None,
            "field_trajectories_csv": str(field_trajectories) if field_trajectories.exists() else None,
            "field_meta_json": str(field_meta) if field_meta.exists() else None,
            "maxwell_report_json": maxwell_json,
            "maxwell_report_txt": maxwell_txt,
            "notes": message,
        }

    return {
        "job_id": spec["job_id"],
        "task_kind": task_kind,
        "state": state,
        "notes": message,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Cloud entrypoint for Block005 jobs")
    ap.add_argument("--job_spec", required=True, help="Path to mounted job.json")
    ap.add_argument("--results_dir", required=True, help="Directory for outputs/result/status/stdout/stderr")
    ap.add_argument(
        "--project_root",
        default=os.environ.get("MUSIC12_PROJECT_ROOT", "/workspace"),
        help="Project root inside container",
    )
    ap.add_argument("--timeout_sec", type=int, default=7200)
    args = ap.parse_args()

    job_spec_path = Path(args.job_spec).resolve()
    results_dir = Path(args.results_dir).resolve()
    project_root = Path(args.project_root).resolve()

    results_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = results_dir / "stdout.txt"
    stderr_path = results_dir / "stderr.txt"
    status_path = results_dir / "status.json"
    result_path = results_dir / "result.json"

    if not job_spec_path.exists():
        raise FileNotFoundError(f"job_spec not found: {job_spec_path}")

    spec = read_json(job_spec_path)
    job_id = str(spec["job_id"])
    task_kind = str(spec.get("task_kind", "")).strip()

    running_status = {
        "job_id": job_id,
        "task_kind": task_kind,
        "state": "running",
        "provider": "gcp_batch",
        "provider_job_id": os.environ.get("BATCH_JOB_ID", ""),
        "created_at": utc_now(),
        "started_at": utc_now(),
        "finished_at": None,
        "message": f"started cloud run for task_kind={task_kind}",
        "return_code": None,
    }
    write_json(status_path, running_status)

    cmd = build_command(
        project_root=project_root,
        spec=spec,
        job_spec_path=job_spec_path,
        results_dir=results_dir,
    )

    state = "failed"
    return_code: int | None = None
    message = "unknown failure"

    with stdout_path.open("w", encoding="utf-8") as fout, stderr_path.open("w", encoding="utf-8") as ferr:
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(project_root),
                stdout=fout,
                stderr=ferr,
                timeout=args.timeout_sec,
                check=False,
            )
            return_code = proc.returncode
            state = "done" if return_code == 0 else "failed"
            message = f"finished with return_code={return_code}"
        except subprocess.TimeoutExpired:
            state = "timeout"
            message = f"timeout after {args.timeout_sec} sec"
        except Exception as exc:
            state = "failed"
            message = f"{type(exc).__name__}: {exc}"

    finished_status = {
        "job_id": job_id,
        "task_kind": task_kind,
        "state": state,
        "provider": "gcp_batch",
        "provider_job_id": os.environ.get("BATCH_JOB_ID", ""),
        "created_at": running_status["created_at"],
        "started_at": running_status["started_at"],
        "finished_at": utc_now(),
        "message": message,
        "return_code": return_code,
    }
    write_json(status_path, finished_status)

    result = build_result_payload(
        project_root=project_root,
        spec=spec,
        results_dir=results_dir,
        state=state,
        message=message,
    )
    write_json(result_path, result)

    print(f"job_id={job_id}")
    print(f"task_kind={task_kind}")
    print(f"state={state}")
    print(message)

    if state == "done":
        sys.exit(0)
    if state == "timeout":
        sys.exit(124)
    sys.exit(return_code if return_code not in (None, 0) else 1)


if __name__ == "__main__":
    main()