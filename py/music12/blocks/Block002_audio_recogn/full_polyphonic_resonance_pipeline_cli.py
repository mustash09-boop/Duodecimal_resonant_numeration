# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(title: str, cmd: list[str]) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Universal full polyphonic resonance pipeline for any WAV."
    )

    ap.add_argument("--wav", required=True)
    ap.add_argument("--report_dir", required=True)
    ap.add_argument("--prefix", required=True)

    ap.add_argument("--project_root", required=True)
    ap.add_argument("--reference_events_csv", default="")
    ap.add_argument("--reference_duration_sec", type=float, default=0.0)
    ap.add_argument("--detected_duration_sec", type=float, default=0.0)

    ap.add_argument("--octave_min", type=int, default=5)
    ap.add_argument("--octave_max", type=int, default=11)
    ap.add_argument("--detail_depth", type=int, default=1)
    ap.add_argument("--projection_depth", type=int, default=1)
    ap.add_argument("--fps", type=float, default=60.0)

    args = ap.parse_args()

    report = Path(args.report_dir)
    report.mkdir(parents=True, exist_ok=True)

    py = sys.executable

    p = args.prefix

    matrix = report / f"{p}_probe_matrix_micro_full.csv"
    times = report / f"{p}_probe_times_micro_full.csv"
    coords = report / f"{p}_probe_coords_micro_full.csv"
    meta = report / f"{p}_probe_meta_micro_full.json"

    framewise = report / f"{p}_framewise_candidates_micro_v1.csv"
    framewise_readable = report / f"{p}_framewise_candidates_micro_v1_readable.csv"

    clusters = report / f"{p}_micro_clusters_v1.csv"
    clusters_readable = report / f"{p}_micro_clusters_v1_readable.csv"

    families = report / f"{p}_micro_families_v1.csv"
    family_frame = report / f"{p}_micro_family_frame_summary_v1.csv"

    directed_edges = report / f"{p}_micro_directed_edges_v1.csv"
    directed_nodes = report / f"{p}_micro_directed_nodes_v1.csv"

    causal_roles = report / f"{p}_micro_causal_roles_v1.csv"
    causal_centers = report / f"{p}_micro_causal_note_centers_v1.csv"

    simul_frames = report / f"{p}_micro_simul_frame_notes_v1.csv"
    simul_readable = report / f"{p}_micro_simul_readable_v1.csv"

    voice_events = report / f"{p}_micro_voice_events_v1.csv"
    voice_summary = report / f"{p}_micro_voice_summary_v1.csv"
    frame_voice = report / f"{p}_micro_frame_voice_v1.csv"

    stable_voices = report / f"{p}_stable_voices_v1.csv"
    stable_mapping = report / f"{p}_stable_voice_mapping_v1.csv"

    run_step("1. Micro resonance scan", [
        py, "-m", "music12.blocks.Block002_audio_recogn.resonance_probe12_scan_cli",
        "--wav", args.wav,
        "--out_matrix_csv", str(matrix),
        "--out_meta_json", str(meta),
        "--out_times_csv", str(times),
        "--out_coords_csv", str(coords),
        "--octave_min", str(args.octave_min),
        "--octave_max", str(args.octave_max),
        "--detail_depth", str(args.detail_depth),
        "--projection_depth", str(args.projection_depth),
        "--time_step_seconds", "0.0166667",
        "--window_seconds", "0.05",
        "--window_type", "hamming",
    ])

    run_step("2. Candidate inference", [
        py, "-m", "music12.blocks.Block002_audio_recogn.resonance_candidate_inference_cli",
        "--matrix_csv", str(matrix),
        "--times_csv", str(times),
        "--coords_csv", str(coords),
        "--out_framewise_csv", str(framewise),
        "--out_framewise_readable_csv", str(framewise_readable),
        "--out_meta_json", str(report / f"{p}_framewise_candidates_micro_v1_meta.json"),
        "--energy_threshold", "0.003",
        "--top_n_candidates", "96",
        "--max_polyphonic_candidates", "48",
        "--analysis_min_hz", "30",
        "--analysis_max_hz", "12000",
    ])

    run_step("3. Micro candidate clustering", [
        py, "-m", "music12.blocks.Block002_audio_recogn.micro_candidate_cluster_cli",
        "--framewise_csv", str(framewise),
        "--out_cluster_csv", str(clusters),
        "--out_cluster_readable_csv", str(clusters_readable),
        "--out_meta_json", str(report / f"{p}_micro_clusters_meta_v1.json"),
        "--out_summary_txt", str(report / f"{p}_micro_clusters_summary_v1.txt"),
    ])

    run_step("4. Micro harmonic families", [
        py, "-m", "music12.blocks.Block002_audio_recogn.micro_harmonic_family_builder_cli",
        "--micro_clusters_csv", str(clusters),
        "--out_family_csv", str(families),
        "--out_frame_summary_csv", str(family_frame),
        "--out_meta_json", str(report / f"{p}_micro_family_meta_v1.json"),
        "--out_summary_txt", str(report / f"{p}_micro_family_summary_v1.txt"),
        "--anchor_token", "9.A'-",
        "--anchor_hz", "440",
        "--max_harmonic", "8",
        "--tolerance_cents", "35",
        "--max_families_per_frame", "12",
    ])

    run_step("5. Directed causality graph", [
        py, "-m", "music12.blocks.Block002_audio_recogn.micro_directed_causality_graph_cli",
        "--micro_family_csv", str(families),
        "--out_directed_edges_csv", str(directed_edges),
        "--out_nodes_csv", str(directed_nodes),
        "--out_meta_json", str(report / f"{p}_micro_directed_meta_v1.json"),
        "--out_summary_txt", str(report / f"{p}_micro_directed_summary_v1.txt"),
        "--max_nodes_per_frame", "12",
        "--lag_min_frames", "1",
        "--lag_max_frames", "6",
        "--min_causal_frames", "5",
        "--min_causal_weight", "0.015",
    ])

    run_step("6. Causal role decomposition", [
        py, "-m", "music12.blocks.Block002_audio_recogn.micro_causal_role_decomposition_cli",
        "--directed_edges_csv", str(directed_edges),
        "--out_roles_csv", str(causal_roles),
        "--out_note_centers_csv", str(causal_centers),
        "--out_meta_json", str(report / f"{p}_micro_causal_roles_meta_v1.json"),
        "--out_summary_txt", str(report / f"{p}_micro_causal_roles_summary_v1.txt"),
        "--min_center_score", "0.015",
    ])

    run_step("7. Simultaneous note disentanglement", [
        py, "-m", "music12.blocks.Block002_audio_recogn.micro_simultaneous_note_disentangler_cli",
        "--micro_family_csv", str(families),
        "--causal_centers_csv", str(causal_centers),
        "--out_frame_notes_csv", str(simul_frames),
        "--out_readable_csv", str(simul_readable),
        "--out_meta_json", str(report / f"{p}_micro_simul_meta_v1.json"),
        "--out_summary_txt", str(report / f"{p}_micro_simul_summary_v1.txt"),
        "--min_center_score", "0.015",
        "--min_family_score", "0.20",
        "--max_notes_per_frame", "8",
        "--max_per_degree", "1",
    ])

    run_step("8. Voice continuity", [
        py, "-m", "music12.blocks.Block002_audio_recogn.micro_voice_continuity_tracker_cli",
        "--frame_notes_csv", str(simul_frames),
        "--out_voice_events_csv", str(voice_events),
        "--out_voice_summary_csv", str(voice_summary),
        "--out_frame_voice_csv", str(frame_voice),
        "--out_meta_json", str(report / f"{p}_micro_voice_meta_v1.json"),
        "--out_summary_txt", str(report / f"{p}_micro_voice_summary_v1.txt"),
        "--max_pitch_jump", "5",
        "--max_gap_frames", "3",
        "--min_voice_len_frames", "6",
    ])

    run_step("9. Voice identity stabilization", [
        py, "-m", "music12.blocks.Block002_audio_recogn.voice_identity_stabilizer_cli",
        "--voice_events_csv", str(voice_events),
        "--out_stable_voices_csv", str(stable_voices),
        "--out_mapping_csv", str(stable_mapping),
        "--out_meta_json", str(report / f"{p}_stable_voices_meta_v1.json"),
        "--out_summary_txt", str(report / f"{p}_stable_voices_summary_v1.txt"),
        "--max_merge_gap_frames", "12",
        "--max_merge_pitch_jump", "4",
        "--min_stable_duration_frames", "12",
    ])

    if args.reference_events_csv and args.reference_duration_sec > 0 and args.detected_duration_sec > 0:
        compare_csv = report / f"{p}_tempo_aligned_polyphony_vs_midi_v1.csv"

        run_step("10. Optional MIDI reference comparison", [
            py, "-m", "music12.blocks.Block002_audio_recogn.tempo_aligned_polyphony_vs_midi_cli",
            "--detected_frame_notes_csv", str(simul_frames),
            "--reference_events_csv", args.reference_events_csv,
            "--out_frame_compare_csv", str(compare_csv),
            "--out_summary_json", str(report / f"{p}_tempo_aligned_polyphony_vs_midi_meta_v1.json"),
            "--out_summary_txt", str(report / f"{p}_tempo_aligned_polyphony_vs_midi_summary_v1.txt"),
            "--detected_duration_sec", str(args.detected_duration_sec),
            "--reference_duration_sec", str(args.reference_duration_sec),
            "--fps", str(args.fps),
        ])

        run_step("11. Optional polyphony diagnostics", [
            py, "-m", "music12.blocks.Block002_audio_recogn.polyphony_error_diagnostics_cli",
            "--frame_compare_csv", str(compare_csv),
            "--out_error_summary_csv", str(report / f"{p}_polyphony_error_summary_v1.csv"),
            "--out_readable_csv", str(report / f"{p}_polyphony_readable_compare_v1.csv"),
            "--out_problem_windows_csv", str(report / f"{p}_polyphony_problem_windows_v1.csv"),
            "--out_meta_json", str(report / f"{p}_polyphony_error_diagnostics_meta_v1.json"),
            "--out_summary_txt", str(report / f"{p}_polyphony_error_diagnostics_summary_v1.txt"),
            "--problem_min_error", "4",
        ])

    print("\nFULL POLYPHONIC RESONANCE PIPELINE COMPLETE")
    print(f"report_dir = {report}")


if __name__ == "__main__":
    main()