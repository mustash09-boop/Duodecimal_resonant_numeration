# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Sequence


DEFAULT_PYTHON_EXE = Path(
    r"C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.10_3.10.3056.0_x64__qbz5n2kfra8p0\python3.10.exe"
)


def _run_step(
    *,
    title: str,
    module: str,
    tag: str,
    module_args: Sequence[str],
    project_root: Path,
    python_exe: Path,
) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    cmd = [
        str(python_exe),
        "-m",
        "music12.demons.demon_maxwell_cli",
        "-m",
        module,
        "--task-class",
        "module_run",
        "--project-root",
        str(project_root),
        "--logdir",
        "_demon_logs",
        "--tag",
        tag,
        "--",
        *module_args,
    ]

    print(" ".join(cmd))

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "py")

    result = subprocess.run(
        cmd,
        cwd=project_root,
        env=env,
        check=False,
    )

    if result.returncode != 0:
        raise SystemExit(f"Step failed: {title}")


def _run_tool_script(
    *,
    title: str,
    script_path: Path,
    script_args: Sequence[str],
    project_root: Path,
    python_exe: Path,
) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    cmd = [str(python_exe), str(script_path), *script_args]
    print(" ".join(cmd))

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "py")

    result = subprocess.run(
        cmd,
        cwd=project_root,
        env=env,
        check=False,
    )

    if result.returncode != 0:
        raise SystemExit(f"Script failed: {title}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run the full Block002 research stack starting from existing probe_* artifacts: "
            "candidate inference, micro clustering, harmonic families, classic tail, and optional excitation branch."
        )
    )
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--report-dir", required=True)
    ap.add_argument("--reference-events-csv", required=True)
    ap.add_argument("--probe-meta-json", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--midi-meta-json", default="")
    ap.add_argument("--python-exe", default=str(DEFAULT_PYTHON_EXE))
    ap.add_argument("--skip-excitation-branch", action="store_true")
    ap.add_argument("--matrix-cache-dir", default="")
    ap.add_argument("--resume-candidate-inference", action="store_true")
    ap.add_argument("--candidate-flush-every", type=int, default=50)
    ap.add_argument("--candidate-start-frame", type=int, default=0)
    ap.add_argument("--candidate-stop-frame", type=int, default=-1)
    ap.add_argument(
        "--tail-mode",
        choices=["full", "baseline-only", "causal-only"],
        default="full",
    )
    args = ap.parse_args()

    project_root = Path(args.project_root)
    report_dir = Path(args.report_dir)
    reference_events_csv = Path(args.reference_events_csv)
    probe_meta_json = Path(args.probe_meta_json)
    prefix = str(args.prefix)
    python_exe = Path(args.python_exe)
    midi_meta_json = str(args.midi_meta_json).strip()
    matrix_cache_dir = str(args.matrix_cache_dir).strip()

    matrix = report_dir / f"{prefix}_probe_matrix_micro_full.csv"
    times = report_dir / f"{prefix}_probe_times_micro_full.csv"
    coords = report_dir / f"{prefix}_probe_coords_micro_full.csv"

    framewise = report_dir / f"{prefix}_framewise_candidates_micro_v1.csv"
    framewise_readable = report_dir / f"{prefix}_framewise_candidates_micro_v1_readable.csv"
    clusters = report_dir / f"{prefix}_micro_clusters_v1.csv"
    clusters_readable = report_dir / f"{prefix}_micro_clusters_v1_readable.csv"
    families = report_dir / f"{prefix}_micro_families_v1.csv"
    family_frame = report_dir / f"{prefix}_micro_family_frame_summary_v1.csv"

    _run_step(
        title="1. Candidate inference (Block002_pipeline)",
        module="music12.blocks.Block002_pipeline.resonance_candidate_inference_cli",
        tag=f"{prefix}_candidate_inference_pipeline_v1",
        module_args=[
            "--matrix_csv", str(matrix),
            "--times_csv", str(times),
            "--coords_csv", str(coords),
            "--out_framewise_csv", str(framewise),
            "--out_framewise_readable_csv", str(framewise_readable),
            "--out_meta_json", str(report_dir / f"{prefix}_framewise_candidates_micro_v1_meta.json"),
            "--energy_threshold", "0.003",
            "--top_n_candidates", "96",
            "--max_polyphonic_candidates", "48",
            "--analysis_min_hz", "30",
            "--analysis_max_hz", "12000",
            "--flush_every", str(args.candidate_flush_every),
            "--start_frame", str(args.candidate_start_frame),
            "--stop_frame", str(args.candidate_stop_frame),
            "--progress_json", str(report_dir / f"{prefix}_framewise_candidates_micro_v1_progress.json"),
            *(
                ["--matrix_cache_dir", matrix_cache_dir]
                if matrix_cache_dir
                else []
            ),
            *(
                ["--resume_if_possible"]
                if args.resume_candidate_inference
                else []
            ),
        ],
        project_root=project_root,
        python_exe=python_exe,
    )

    _run_step(
        title="2. Micro candidate clustering (Block002_pipeline)",
        module="music12.blocks.Block002_pipeline.micro_candidate_cluster_cli",
        tag=f"{prefix}_micro_clusters_pipeline_v1",
        module_args=[
            "--framewise_csv", str(framewise),
            "--out_cluster_csv", str(clusters),
            "--out_cluster_readable_csv", str(clusters_readable),
            "--out_meta_json", str(report_dir / f"{prefix}_micro_clusters_meta_v1.json"),
            "--out_summary_txt", str(report_dir / f"{prefix}_micro_clusters_summary_v1.txt"),
        ],
        project_root=project_root,
        python_exe=python_exe,
    )

    _run_step(
        title="3. Micro harmonic families (Block002_pipeline)",
        module="music12.blocks.Block002_pipeline.micro_harmonic_family_builder_cli",
        tag=f"{prefix}_micro_families_pipeline_v1",
        module_args=[
            "--micro_clusters_csv", str(clusters),
            "--out_family_csv", str(families),
            "--out_frame_summary_csv", str(family_frame),
            "--out_meta_json", str(report_dir / f"{prefix}_micro_family_meta_v1.json"),
            "--out_summary_txt", str(report_dir / f"{prefix}_micro_family_summary_v1.txt"),
            "--anchor_token", "9.A'-",
            "--anchor_hz", "440",
            "--max_harmonic", "8",
            "--tolerance_cents", "35",
            "--max_families_per_frame", "12",
        ],
        project_root=project_root,
        python_exe=python_exe,
    )

    _run_tool_script(
        title="4. Classic tail from micro_families (Block002_pipeline)",
        script_path=project_root / "tools" / "block002_pipeline_runner.py",
        script_args=[
            "--project-root", str(project_root),
            "--report-dir", str(report_dir),
            "--reference-events-csv", str(reference_events_csv),
            "--probe-meta-json", str(probe_meta_json),
            "--prefix", prefix,
            "--python-exe", str(python_exe),
            "--mode", str(args.tail_mode),
        ],
        project_root=project_root,
        python_exe=python_exe,
    )

    if not args.skip_excitation_branch:
        script_args = [
            "--project-root", str(project_root),
            "--report-dir", str(report_dir),
            "--prefix", prefix,
            "--python-exe", str(python_exe),
        ]
        if midi_meta_json:
            script_args.extend(["--midi-meta-json", midi_meta_json])

        _run_tool_script(
            title="5. Excitation-first branch (Block002_pipeline)",
            script_path=project_root / "tools" / "block002_excitation_runner.py",
            script_args=script_args,
            project_root=project_root,
            python_exe=python_exe,
        )

    print()
    print("BLOCK002 FULL FROM PROBE COMPLETE")
    print(f"report_dir = {report_dir}")


if __name__ == "__main__":
    main()
