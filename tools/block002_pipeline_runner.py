# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
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


def _load_probe_duration_sec(probe_meta_json: Path) -> float:
    data = json.loads(probe_meta_json.read_text(encoding="utf-8"))
    return float(data["time_slice"]["effective_duration_seconds"])


def _load_reference_duration_sec(reference_events_csv: Path) -> float:
    with reference_events_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return float(rows[-1]["end_sec"])


def run_pipeline(
    *,
    project_root: Path,
    python_exe: Path,
    report_dir: Path,
    reference_events_csv: Path,
    probe_meta_json: Path,
    prefix: str,
    mode: str,
) -> None:
    detected_duration_sec = _load_probe_duration_sec(probe_meta_json)
    reference_duration_sec = _load_reference_duration_sec(reference_events_csv)

    families = report_dir / f"{prefix}_micro_families_v1.csv"
    directed_edges = report_dir / f"{prefix}_micro_directed_edges_v1.csv"
    directed_nodes = report_dir / f"{prefix}_micro_directed_nodes_v1.csv"
    causal_roles = report_dir / f"{prefix}_micro_causal_roles_v1.csv"
    causal_centers = report_dir / f"{prefix}_micro_causal_note_centers_v1.csv"
    simul_frames = report_dir / f"{prefix}_micro_simul_frame_notes_v1.csv"
    simul_readable = report_dir / f"{prefix}_micro_simul_readable_v1.csv"
    voice_events = report_dir / f"{prefix}_micro_voice_events_v1.csv"
    voice_summary = report_dir / f"{prefix}_micro_voice_summary_v1.csv"
    frame_voice = report_dir / f"{prefix}_micro_frame_voice_v1.csv"
    stable_voices = report_dir / f"{prefix}_stable_voices_v1.csv"
    stable_mapping = report_dir / f"{prefix}_stable_voice_mapping_v1.csv"
    causal_hypotheses = report_dir / f"{prefix}_causal_hypotheses_v1.csv"
    causal_resolved = report_dir / f"{prefix}_causal_resolved_events_v1.csv"
    causal_frame_notes = report_dir / f"{prefix}_causal_frame_notes_v1.csv"
    causal_readable = report_dir / f"{prefix}_causal_readable_v1.csv"
    compare_csv = report_dir / f"{prefix}_tempo_aligned_polyphony_vs_midi_v1.csv"
    compare_causal_csv = report_dir / f"{prefix}_tempo_aligned_polyphony_vs_midi_causal_v1.csv"

    if mode in {"full", "baseline-only", "causal-only"}:
        _run_step(
            title="5. Directed causality graph (Block002_pipeline)",
            module="music12.blocks.Block002_pipeline.micro_directed_causality_graph_cli",
            tag=f"{prefix}_micro_directed_pipeline_v1",
            module_args=[
                "--micro_family_csv", str(families),
                "--out_directed_edges_csv", str(directed_edges),
                "--out_nodes_csv", str(directed_nodes),
                "--out_meta_json", str(report_dir / f"{prefix}_micro_directed_meta_v1.json"),
                "--out_summary_txt", str(report_dir / f"{prefix}_micro_directed_summary_v1.txt"),
                "--max_nodes_per_frame", "12",
                "--lag_min_frames", "1",
                "--lag_max_frames", "6",
                "--min_causal_frames", "3",
                "--min_causal_weight", "0.008",
                "--same_degree_weight_scale", "0.35",
            ],
            project_root=project_root,
            python_exe=python_exe,
        )

        _run_step(
            title="6. Causal role decomposition (Block002_pipeline)",
            module="music12.blocks.Block002_pipeline.micro_causal_role_decomposition_cli",
            tag=f"{prefix}_micro_causal_roles_pipeline_v1",
            module_args=[
                "--directed_edges_csv", str(directed_edges),
                "--out_roles_csv", str(causal_roles),
                "--out_note_centers_csv", str(causal_centers),
                "--out_meta_json", str(report_dir / f"{prefix}_micro_causal_roles_meta_v1.json"),
                "--out_summary_txt", str(report_dir / f"{prefix}_micro_causal_roles_summary_v1.txt"),
                "--min_center_score", "0.010",
                "--bridge_center_min_score", "0.020",
                "--bridge_center_min_asymmetry", "0.10",
            ],
            project_root=project_root,
            python_exe=python_exe,
        )

        _run_step(
            title="7. Simultaneous note disentanglement (Block002_pipeline)",
            module="music12.blocks.Block002_pipeline.micro_simultaneous_note_disentangler_cli",
            tag=f"{prefix}_micro_simul_pipeline_v1",
            module_args=[
                "--micro_family_csv", str(families),
                "--causal_centers_csv", str(causal_centers),
                "--roles_csv", str(causal_roles),
                "--out_frame_notes_csv", str(simul_frames),
                "--out_readable_csv", str(simul_readable),
                "--out_meta_json", str(report_dir / f"{prefix}_micro_simul_meta_v1.json"),
                "--out_summary_txt", str(report_dir / f"{prefix}_micro_simul_summary_v1.txt"),
                "--min_center_score", "0.010",
                "--min_family_score", "0.16",
                "--min_structural_support", "0.34",
                "--min_structural_root_micro_count", "6",
                "--min_structural_diversity", "4",
                "--max_notes_per_frame", "10",
                "--max_per_degree", "1",
                "--max_structural_companions_per_center", "1",
                "--max_structural_companions_without_center", "1",
            ],
            project_root=project_root,
            python_exe=python_exe,
        )

        _run_step(
            title="8. Voice continuity (Block002_pipeline)",
            module="music12.blocks.Block002_pipeline.micro_voice_continuity_tracker_cli",
            tag=f"{prefix}_micro_voice_pipeline_v1",
            module_args=[
                "--frame_notes_csv", str(simul_frames),
                "--out_voice_events_csv", str(voice_events),
                "--out_voice_summary_csv", str(voice_summary),
                "--out_frame_voice_csv", str(frame_voice),
                "--out_meta_json", str(report_dir / f"{prefix}_micro_voice_meta_v1.json"),
                "--out_summary_txt", str(report_dir / f"{prefix}_micro_voice_summary_v1.txt"),
                "--max_pitch_jump", "5",
                "--max_gap_frames", "3",
                "--min_voice_len_frames", "6",
                "--min_exciter_frames", "2",
                "--min_exciter_ratio", "0.08",
                "--max_structural_companion_ratio", "0.92",
            ],
            project_root=project_root,
            python_exe=python_exe,
        )

        _run_step(
            title="9. Voice identity stabilization (Block002_pipeline)",
            module="music12.blocks.Block002_pipeline.voice_identity_stabilizer_cli",
            tag=f"{prefix}_stable_voices_pipeline_v1",
            module_args=[
                "--voice_events_csv", str(voice_events),
                "--out_stable_voices_csv", str(stable_voices),
                "--out_mapping_csv", str(stable_mapping),
                "--out_meta_json", str(report_dir / f"{prefix}_stable_voices_meta_v1.json"),
                "--out_summary_txt", str(report_dir / f"{prefix}_stable_voices_summary_v1.txt"),
                "--max_merge_gap_frames", "12",
                "--max_merge_pitch_jump", "4",
                "--min_stable_duration_frames", "12",
            ],
            project_root=project_root,
            python_exe=python_exe,
        )

    if mode in {"full", "causal-only"}:
        _run_step(
            title="10. Causal note hypothesis resolution (Block002_pipeline)",
            module="music12.blocks.Block002_pipeline.causal_note_hypothesis_resolver_cli",
            tag=f"{prefix}_causal_hypothesis_pipeline_v1",
            module_args=[
                "--stable_voices_csv", str(stable_voices),
                "--frame_notes_csv", str(simul_frames),
                "--causal_roles_csv", str(causal_roles),
                "--causal_centers_csv", str(causal_centers),
                "--micro_family_csv", str(families),
                "--out_hypotheses_csv", str(causal_hypotheses),
                "--out_resolved_events_csv", str(causal_resolved),
                "--out_frame_notes_csv", str(causal_frame_notes),
                "--out_readable_csv", str(causal_readable),
                "--out_meta_json", str(report_dir / f"{prefix}_causal_hypothesis_meta_v1.json"),
                "--out_summary_txt", str(report_dir / f"{prefix}_causal_hypothesis_summary_v1.txt"),
                "--min_hypothesis_score", "0.55",
                "--min_confidence", "0.44",
                "--min_local_candidate_probability", "0.24",
                "--min_proto_probability", "0.22",
                "--min_proto_score", "0.68",
                "--top_k", "4",
                "--fps", "60",
            ],
            project_root=project_root,
            python_exe=python_exe,
        )

        _run_step(
            title="12. Tempo-aligned MIDI comparison (causal hypothesis experiment)",
            module="music12.blocks.Block002_pipeline.tempo_aligned_polyphony_vs_midi_cli",
            tag=f"{prefix}_tempo_aligned_causal_pipeline_v1",
            module_args=[
                "--detected_frame_notes_csv", str(causal_frame_notes),
                "--reference_events_csv", str(reference_events_csv),
                "--out_frame_compare_csv", str(compare_causal_csv),
                "--out_summary_json", str(report_dir / f"{prefix}_tempo_aligned_polyphony_vs_midi_causal_meta_v1.json"),
                "--out_summary_txt", str(report_dir / f"{prefix}_tempo_aligned_polyphony_vs_midi_causal_summary_v1.txt"),
                "--detected_duration_sec", str(detected_duration_sec),
                "--reference_duration_sec", str(reference_duration_sec),
                "--fps", "60",
            ],
            project_root=project_root,
            python_exe=python_exe,
        )

    if mode in {"full", "baseline-only"}:
        _run_step(
            title="11. Tempo-aligned MIDI comparison (raw micro_simul baseline)",
            module="music12.blocks.Block002_pipeline.tempo_aligned_polyphony_vs_midi_cli",
            tag=f"{prefix}_tempo_aligned_pipeline_v1",
            module_args=[
                "--detected_frame_notes_csv", str(simul_frames),
                "--reference_events_csv", str(reference_events_csv),
                "--out_frame_compare_csv", str(compare_csv),
                "--out_summary_json", str(report_dir / f"{prefix}_tempo_aligned_polyphony_vs_midi_meta_v1.json"),
                "--out_summary_txt", str(report_dir / f"{prefix}_tempo_aligned_polyphony_vs_midi_summary_v1.txt"),
                "--detected_duration_sec", str(detected_duration_sec),
                "--reference_duration_sec", str(reference_duration_sec),
                "--fps", "60",
            ],
            project_root=project_root,
            python_exe=python_exe,
        )

        _run_step(
            title="13. Polyphony diagnostics (raw baseline)",
            module="music12.blocks.Block002_pipeline.polyphony_error_diagnostics_cli",
            tag=f"{prefix}_polyphony_diagnostics_pipeline_v1",
            module_args=[
                "--frame_compare_csv", str(compare_csv),
                "--out_error_summary_csv", str(report_dir / f"{prefix}_polyphony_error_summary_v1.csv"),
                "--out_readable_csv", str(report_dir / f"{prefix}_polyphony_readable_compare_v1.csv"),
                "--out_problem_windows_csv", str(report_dir / f"{prefix}_polyphony_problem_windows_v1.csv"),
                "--out_meta_json", str(report_dir / f"{prefix}_polyphony_error_diagnostics_meta_v1.json"),
                "--out_summary_txt", str(report_dir / f"{prefix}_polyphony_error_diagnostics_summary_v1.txt"),
                "--problem_min_error", "4",
            ],
            project_root=project_root,
            python_exe=python_exe,
        )

    print()
    print(f"BLOCK002 PIPELINE MODE COMPLETE: {mode}")
    print(f"report_dir = {report_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Universal runner for the Block002_pipeline tail starting from micro_families."
    )
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--report-dir", required=True)
    ap.add_argument("--reference-events-csv", required=True)
    ap.add_argument("--probe-meta-json", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--python-exe", default=str(DEFAULT_PYTHON_EXE))
    ap.add_argument(
        "--mode",
        choices=["full", "baseline-only", "causal-only"],
        default="full",
    )
    args = ap.parse_args()

    run_pipeline(
        project_root=Path(args.project_root),
        python_exe=Path(args.python_exe),
        report_dir=Path(args.report_dir),
        reference_events_csv=Path(args.reference_events_csv),
        probe_meta_json=Path(args.probe_meta_json),
        prefix=args.prefix,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
