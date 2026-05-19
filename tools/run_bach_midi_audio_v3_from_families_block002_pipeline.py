# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

from block002_pipeline_runner import DEFAULT_PYTHON_EXE, run_pipeline


PROJECT_ROOT = Path(r"E:\Duodecimal_resonant_numeration")
REPORT_DIR = PROJECT_ROOT / r"Block001_data\Bach_Invention_1\10_reports_midi_audio_v3"
REFERENCE_EVENTS_CSV = PROJECT_ROOT / r"Block001_data\Bach_Invention_1\00_sources\midi\bach_invention_1_midi_events_v1.csv"
PROBE_META_JSON = REPORT_DIR / "bach_midi_audio_probe_meta_micro_full.json"
PREFIX = "bach_midi_audio"


def main() -> None:
    run_pipeline(
        project_root=PROJECT_ROOT,
        python_exe=DEFAULT_PYTHON_EXE,
        report_dir=REPORT_DIR,
        reference_events_csv=REFERENCE_EVENTS_CSV,
        probe_meta_json=PROBE_META_JSON,
        prefix=PREFIX,
        mode="full",
    )


if __name__ == "__main__":
    main()
