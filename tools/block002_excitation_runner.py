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

    result = subprocess.run(cmd, cwd=project_root, env=env, check=False)
    if result.returncode != 0:
        raise SystemExit(f"Step failed: {title}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the new Block002 excitation-first probe pipeline from probe_* artifacts."
    )
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--report-dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--midi-meta-json", default="")
    ap.add_argument("--python-exe", default=str(DEFAULT_PYTHON_EXE))
    args = ap.parse_args()

    project_root = Path(args.project_root)
    report_dir = Path(args.report_dir)
    prefix = str(args.prefix)
    python_exe = Path(args.python_exe)
    midi_meta_json = Path(args.midi_meta_json) if str(args.midi_meta_json).strip() else None

    probe_matrix = report_dir / f"{prefix}_probe_matrix_micro_full.csv"
    probe_times = report_dir / f"{prefix}_probe_times_micro_full.csv"
    probe_coords = report_dir / f"{prefix}_probe_coords_micro_full.csv"

    excitation_seeds = report_dir / f"{prefix}_excitation_seeds_v1.csv"
    proto_exciters = report_dir / f"{prefix}_proto_exciters_v1.csv"
    micro_families = report_dir / f"{prefix}_micro_families_v1.csv"
    branch_analysis = report_dir / f"{prefix}_exciter_branch_analysis_v1.csv"
    pitched_proto = report_dir / f"{prefix}_pitched_proto_exciters_v1.csv"
    event_proto = report_dir / f"{prefix}_event_proto_exciters_v1.csv"
    event_field_proto = report_dir / f"{prefix}_event_field_proto_exciters_v1.csv"
    event_only_proto = report_dir / f"{prefix}_event_only_proto_exciters_v1.csv"
    unresolved_proto = report_dir / f"{prefix}_unresolved_proto_exciters_v1.csv"
    notechain_proto = report_dir / f"{prefix}_notechain_proto_exciters_v1.csv"
    notechain_rescue_proto = report_dir / f"{prefix}_notechain_rescue_proto_exciters_v1.csv"
    notechain_proto_combined = report_dir / f"{prefix}_notechain_proto_exciters_combined_v1.csv"
    primary_chain_frames = report_dir / f"{prefix}_primary_note_chain_frames_v1.csv"
    primary_chains = report_dir / f"{prefix}_primary_note_chains_v1.csv"
    controlled_sustain_frames = report_dir / f"{prefix}_controlled_sustain_frames_v1.csv"
    controlled_sustain_chains = report_dir / f"{prefix}_controlled_sustain_chains_v1.csv"
    event_field_frames = report_dir / f"{prefix}_event_field_frames_v1.csv"
    event_field_entities = report_dir / f"{prefix}_event_field_entities_v1.csv"
    event_field_group_frames = report_dir / f"{prefix}_event_field_groups_frames_v1.csv"
    event_field_group_entities = report_dir / f"{prefix}_event_field_groups_v1.csv"
    fused_events = report_dir / f"{prefix}_cross_branch_fused_events_v1.csv"
    fused_anchored_events = report_dir / f"{prefix}_cross_branch_fused_events_anchored_v1.csv"
    fused_onset_groups = report_dir / f"{prefix}_cross_branch_fused_onset_groups_v1.csv"

    _run_step(
        title="1. Excitation seed extraction (Block002_pipeline)",
        module="music12.blocks.Block002_pipeline.excitation_seed_extractor_cli",
        tag=f"{prefix}_excitation_seed_v1",
        module_args=[
            "--probe-matrix-csv", str(probe_matrix),
            "--probe-times-csv", str(probe_times),
            "--probe-coords-csv", str(probe_coords),
            "--out-seeds-csv", str(excitation_seeds),
            "--out-summary-txt", str(report_dir / f"{prefix}_excitation_seed_summary_v1.txt"),
            "--out-meta-json", str(report_dir / f"{prefix}_excitation_seed_meta_v1.json"),
        ],
        project_root=project_root,
        python_exe=python_exe,
    )

    _run_step(
        title="2. Proto exciter build (Block002_pipeline)",
        module="music12.blocks.Block002_pipeline.proto_exciter_builder_cli",
        tag=f"{prefix}_proto_exciter_v1",
        module_args=[
            "--excitation-seeds-csv", str(excitation_seeds),
            "--out-proto-exciters-csv", str(proto_exciters),
            "--out-summary-txt", str(report_dir / f"{prefix}_proto_exciter_summary_v1.txt"),
            "--out-meta-json", str(report_dir / f"{prefix}_proto_exciter_meta_v1.json"),
        ],
        project_root=project_root,
        python_exe=python_exe,
    )

    _run_step(
        title="3. Exciter branch classification (Block002_pipeline)",
        module="music12.blocks.Block002_pipeline.exciter_branch_classifier_cli",
        tag=f"{prefix}_exciter_branch_v1",
        module_args=[
            "--proto-exciters-csv", str(proto_exciters),
            "--micro-families-csv", str(micro_families),
            "--out-branch-analysis-csv", str(branch_analysis),
            "--out-pitched-proto-exciters-csv", str(pitched_proto),
            "--out-event-proto-exciters-csv", str(event_proto),
            "--out-event-field-proto-exciters-csv", str(event_field_proto),
            "--out-event-only-proto-exciters-csv", str(event_only_proto),
            "--out-unresolved-proto-exciters-csv", str(unresolved_proto),
            "--out-notechain-proto-exciters-csv", str(notechain_proto),
            "--out-summary-txt", str(report_dir / f"{prefix}_exciter_branch_summary_v1.txt"),
            "--out-meta-json", str(report_dir / f"{prefix}_exciter_branch_meta_v1.json"),
        ],
        project_root=project_root,
        python_exe=python_exe,
    )

    _run_step(
        title="4. Event resonance-field mapping (Block002_pipeline)",
        module="music12.blocks.Block002_pipeline.event_resonance_field_mapper_cli",
        tag=f"{prefix}_event_field_v1",
        module_args=[
            "--event-field-proto-exciters-csv", str(event_field_proto),
            "--micro-families-csv", str(micro_families),
            "--out-event-field-frames-csv", str(event_field_frames),
            "--out-event-field-entities-csv", str(event_field_entities),
            "--out-summary-txt", str(report_dir / f"{prefix}_event_field_summary_v1.txt"),
            "--out-meta-json", str(report_dir / f"{prefix}_event_field_meta_v1.json"),
        ],
        project_root=project_root,
        python_exe=python_exe,
    )

    _run_step(
        title="5. Event field onset-group merge (Block002_pipeline)",
        module="music12.blocks.Block002_pipeline.event_field_onset_group_merger_cli",
        tag=f"{prefix}_event_field_groups_v1",
        module_args=[
            "--event-field-entities-csv", str(event_field_entities),
            "--out-merged-entities-csv", str(event_field_group_entities),
            "--out-merged-frames-csv", str(event_field_group_frames),
            "--out-summary-txt", str(report_dir / f"{prefix}_event_field_groups_summary_v1.txt"),
            "--out-meta-json", str(report_dir / f"{prefix}_event_field_groups_meta_v1.json"),
        ],
        project_root=project_root,
        python_exe=python_exe,
    )

    _run_step(
        title="6. Event-field notechain rescue selection (Block002_pipeline)",
        module="music12.blocks.Block002_pipeline.event_field_notechain_rescue_selector_cli",
        tag=f"{prefix}_notechain_rescue_v1",
        module_args=[
            "--event-field-groups-csv", str(event_field_group_entities),
            "--event-field-proto-exciters-csv", str(event_field_proto),
            "--notechain-proto-exciters-csv", str(notechain_proto),
            "--out-rescue-proto-exciters-csv", str(notechain_rescue_proto),
            "--out-combined-notechain-proto-exciters-csv", str(notechain_proto_combined),
            "--out-summary-txt", str(report_dir / f"{prefix}_notechain_rescue_summary_v1.txt"),
            "--out-meta-json", str(report_dir / f"{prefix}_notechain_rescue_meta_v1.json"),
        ],
        project_root=project_root,
        python_exe=python_exe,
    )

    _run_step(
        title="7. Primary note chain build (Block002_pipeline)",
        module="music12.blocks.Block002_pipeline.primary_note_chain_builder_cli",
        tag=f"{prefix}_primary_note_chain_v1",
        module_args=[
            "--proto-exciters-csv", str(notechain_proto_combined),
            "--all-proto-exciters-csv", str(proto_exciters),
            "--branch-analysis-csv", str(branch_analysis),
            "--micro-families-csv", str(micro_families),
            "--out-chain-frames-csv", str(primary_chain_frames),
            "--out-chains-csv", str(primary_chains),
            "--allow-single-frame-rescue",
            "--out-summary-txt", str(report_dir / f"{prefix}_primary_note_chain_summary_v1.txt"),
            "--out-meta-json", str(report_dir / f"{prefix}_primary_note_chain_meta_v1.json"),
        ],
        project_root=project_root,
        python_exe=python_exe,
    )

    _run_step(
        title="8. Controlled sustain / box transfer (Block002_pipeline)",
        module="music12.blocks.Block002_pipeline.controlled_sustain_transfer_mapper_cli",
        tag=f"{prefix}_controlled_sustain_v1",
        module_args=[
            "--primary-chains-csv", str(primary_chains),
            "--primary-chain-frames-csv", str(primary_chain_frames),
            "--micro-families-csv", str(micro_families),
            "--out-extended-frames-csv", str(controlled_sustain_frames),
            "--out-extended-chains-csv", str(controlled_sustain_chains),
            "--out-summary-txt", str(report_dir / f"{prefix}_controlled_sustain_summary_v1.txt"),
            "--out-meta-json", str(report_dir / f"{prefix}_controlled_sustain_meta_v1.json"),
        ],
        project_root=project_root,
        python_exe=python_exe,
    )

    _run_step(
        title="9. Cross-branch event fusion (Block002_pipeline)",
        module="music12.blocks.Block002_pipeline.cross_branch_event_fusion_cli",
        tag=f"{prefix}_cross_branch_fusion_v1",
        module_args=[
            "--notechain-chains-csv", str(controlled_sustain_chains),
            "--event-field-groups-csv", str(event_field_group_entities),
            "--out-fused-events-csv", str(fused_events),
            "--out-summary-txt", str(report_dir / f"{prefix}_cross_branch_fusion_summary_v1.txt"),
            "--out-meta-json", str(report_dir / f"{prefix}_cross_branch_fusion_meta_v1.json"),
        ],
        project_root=project_root,
        python_exe=python_exe,
    )

    _run_step(
        title="10. Fused event onset-anchor resolution (Block002_pipeline)",
        module="music12.blocks.Block002_pipeline.fused_event_onset_anchor_resolver_cli",
        tag=f"{prefix}_fused_onset_anchor_v1",
        module_args=[
            "--fused-events-csv", str(fused_events),
            "--out-anchored-events-csv", str(fused_anchored_events),
            "--out-onset-groups-csv", str(fused_onset_groups),
            "--out-summary-txt", str(report_dir / f"{prefix}_fused_onset_anchor_summary_v1.txt"),
            "--out-meta-json", str(report_dir / f"{prefix}_fused_onset_anchor_meta_v1.json"),
        ],
        project_root=project_root,
        python_exe=python_exe,
    )

    if midi_meta_json is not None:
        _run_step(
            title="11. Pipeline target alignment audit (Block002_pipeline)",
            module="music12.blocks.Block002_pipeline.pipeline_target_alignment_audit_cli",
            tag=f"{prefix}_target_alignment_v1",
            module_args=[
                "--midi-meta-json", str(midi_meta_json),
                "--notechain-chains-csv", str(controlled_sustain_chains),
                "--notechain-frames-csv", str(controlled_sustain_frames),
                "--event-field-entities-csv", str(event_field_group_entities),
                "--event-field-frames-csv", str(event_field_group_frames),
                "--fused-events-csv", str(fused_events),
                "--fused-onset-groups-csv", str(fused_onset_groups),
                "--out-summary-txt", str(report_dir / f"{prefix}_pipeline_target_alignment_summary_v1.txt"),
                "--out-meta-json", str(report_dir / f"{prefix}_pipeline_target_alignment_meta_v1.json"),
            ],
            project_root=project_root,
            python_exe=python_exe,
        )


if __name__ == "__main__":
    main()
