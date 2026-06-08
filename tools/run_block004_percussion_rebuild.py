from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(r"E:\Duodecimal_resonant_numeration")
PY_ROOT = PROJECT_ROOT / "py"
PERC_ROOT = PROJECT_ROOT / "Block004_data" / "percussion"
REPORTS_DIR = PROJECT_ROOT / "docs" / "reports"
RUN_TAG = "block004_percussion_rebuild_2026-06-07"
LOG_PATH = REPORTS_DIR / f"{RUN_TAG}.log"
STATE_PATH = REPORTS_DIR / f"{RUN_TAG}_state.json"
LOCK_PATH = REPORTS_DIR / f"{RUN_TAG}.lock"


def utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_log(message: str) -> None:
    ensure_parent(LOG_PATH)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(message.rstrip() + "\n")


def write_state(state: dict) -> None:
    ensure_parent(STATE_PATH)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def build_env() -> dict[str, str]:
    env = dict(os.environ)
    old = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = str(PY_ROOT) if not old else str(PY_ROOT) + os.pathsep + old
    return env


def run_cmd(label: str, cmd: list[str], state: dict) -> None:
    append_log(f"START {label}")
    append_log("RUN " + " ".join(f'"{part}"' if " " in str(part) else str(part) for part in cmd))
    state["current_stage"] = label
    write_state(state)

    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=build_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        append_log(line.rstrip())

    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"{label} failed with exit code {rc}")

    state["completed_stages"].append(label)
    append_log(f"DONE {label}")
    write_state(state)


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if LOCK_PATH.exists():
        print(f"Lock exists: {LOCK_PATH}")
        return 1
    LOCK_PATH.write_text(str(os.getpid()), encoding="utf-8")

    audio_dir = PERC_ROOT / "00_sources" / "audio_notes_wav"
    manifest_csv = PERC_ROOT / "20_manifest" / "percussion_manifest_events.csv"
    core_list = PERC_ROOT / "20_manifest" / "percussion_core_list.txt"
    reports_root = PERC_ROOT / "10_reports"
    spiral3d_dir = PERC_ROOT / "50_spiral3d"
    lineage_dir = PERC_ROOT / "55_cluster_lineage_spiral3d"
    passports_dir = PERC_ROOT / "40_passports"

    state = {
        "status": "running",
        "started_at": utc_now(),
        "finished_at": "",
        "current_stage": "starting",
        "completed_stages": [],
        "failed_stage": "",
        "outputs": {
            "manifest_csv": str(manifest_csv),
            "core_list": str(core_list),
            "reports_root": str(reports_root),
            "spiral3d_dir": str(spiral3d_dir),
            "lineage_dir": str(lineage_dir),
            "passports_dir": str(passports_dir),
        },
    }
    write_state(state)
    append_log(f"START RUN {RUN_TAG}")

    try:
        run_cmd(
            "manifest",
            [
                sys.executable,
                "-m",
                "music12.blocks.Block004_real_instruments.percussion_manifest12_cli",
                "--input_dir",
                str(audio_dir),
                "--out_csv",
                str(manifest_csv),
                "--out_core_list",
                str(core_list),
                "--instrument_family",
                "percussion",
            ],
            state,
        )

        run_cmd(
            "event_pipeline",
            [
                sys.executable,
                "-m",
                "music12.blocks.Block004_real_instruments.percussion_event_pipeline_cli",
                "--manifest_csv",
                str(manifest_csv),
                "--reports_root",
                str(reports_root),
            ],
            state,
        )

        run_cmd(
            "spiral3d",
            [
                sys.executable,
                "-m",
                "music12.blocks.Block004_real_instruments.percussion_spiral3d_builder_cli",
                "--instrument_name",
                "percussion",
                "--reports_root",
                str(reports_root),
                "--out_dir",
                str(spiral3d_dir),
            ],
            state,
        )

        run_cmd(
            "cluster_lineage_spiral3d",
            [
                sys.executable,
                "-m",
                "music12.blocks.Block004_real_instruments.percussion_cluster_lineage_builder_cli",
                "--instrument_name",
                "percussion",
                "--spiral3d_dir",
                str(spiral3d_dir),
                "--out_dir",
                str(lineage_dir),
            ],
            state,
        )

        run_cmd(
            "passports",
            [
                sys.executable,
                "-m",
                "music12.blocks.Block004_real_instruments.percussion_passport_builder_cli",
                "--reports_root",
                str(reports_root),
                "--out_dir",
                str(passports_dir),
            ],
            state,
        )

        run_cmd(
            "morphology_compare",
            [sys.executable, str(PROJECT_ROOT / "tools" / "build_percussion_event_morphology_compare.py")],
            state,
        )

        run_cmd(
            "augment_passports_with_event_morphology",
            [sys.executable, str(PROJECT_ROOT / "tools" / "augment_percussion_passports_with_event_morphology.py")],
            state,
        )

        run_cmd(
            "augment_passports_with_cluster_lineage",
            [sys.executable, str(PROJECT_ROOT / "tools" / "augment_percussion_passports_with_cluster_lineage.py")],
            state,
        )

        state["status"] = "completed"
        state["current_stage"] = "finished"
        state["finished_at"] = utc_now()
        write_state(state)
        append_log("FINISH status=completed")
        return 0
    except Exception as exc:
        state["status"] = "failed"
        state["failed_stage"] = state.get("current_stage", "")
        state["finished_at"] = utc_now()
        write_state(state)
        append_log(f"FAIL {exc}")
        raise
    finally:
        if LOCK_PATH.exists():
            LOCK_PATH.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
