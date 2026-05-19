# Public Code Selection v1

## Purpose

This note defines what should be included in the next public-facing code layer
and what should remain private or be cleaned first.

## Good candidates for the next public commit

### Core package

- `py/music12/core/`
- `py/music12/audio/`
- `py/music12/ops/`

These are structural and conceptual foundations and should be publishable with
minimal risk.

### Main block code

- `py/music12/blocks/Block001_score_scan/`
- `py/music12/blocks/Block002_pipeline/`
- `py/music12/blocks/Block003_verify/`
- `py/music12/blocks/Block004_real_instruments/`
- `py/music12/blocks/Block005_job_orchestrator/`

These are the main research modules. They are worth publishing, but a small
sanity pass is recommended first because some files contain local path
assumptions or cloud-specific entrypoints.

### Useful external tools

The following tool families look public-useful:

- runners for Block002 and Block004
- audits for event structure, instrument affinity, and identity
- stem builders and mask builders
- enrichment tools for MIDI metadata

In practice, most of `tools/` is publishable as research tooling.

## Publish later after light cleanup

### Files with hardcoded local paths or machine-specific assumptions

Examples found in the audit:

- `py/build_note_centers_v8.py`
- `py/build_polyphonic_note_timeline_v8.py`
- `py/build_coords_delta_view_v8.py`
- `py/merge_probe_v7.py`
- `py/merge_probe_v8.py`
- `py/rewrite_coords_v8.py`
- `tools/backup_codex_state.ps1`
- `tools/project_snapshot.py`
- `tools/maxwell_snapshot.py`
- `tools/run_bach_midi_audio_v3_from_probe.ps1`
- `tools/run_bach_midi_audio_v3_from_families_block002_pipeline.py`
- `tools/run_phase_cluster_realpiano1.py`
- `tools/run_realpiano1_full_scan.py`

These are not bad files. They just need one pass to replace hardcoded absolute
paths with configurable arguments or documented environment assumptions.

### Cloud-specific files

- `py/music12/blocks/Block005_job_orchestrator/provider_gcp_storage_cli.py`
- `py/music12/blocks/Block005_job_orchestrator/provider_gcp_batch.py`

These should stay, but should be documented as optional cloud backends rather
than assumed defaults.

## Keep private for now

### Internal research memory

- `docs/jim_memory/`
- `docs/reports/`

These are valuable internally, but they are not yet clean public-facing
documents.

### Archived and deprecated code

- `py/music12/blocks/Block002_audio_recogn/_archive_pre_micro/`
- `py/music12/blocks/Block002_audio_recogn/_deprecated_2026_05_11/`

These should not enter the first public code layer.

## Audit result summary

The codebase does **not** currently look blocked by secrets in the usual sense.

The main publication risk is different:

- hardcoded local file paths
- machine-specific helper scripts
- internal memory folders
- archived branches of the algorithm

## Recommended next public code commit

1. `py/music12/` main package, excluding deprecated/archive folders
2. selected `tools/` that are generic and argument-driven
3. one compact example showing how the code is intended to run
