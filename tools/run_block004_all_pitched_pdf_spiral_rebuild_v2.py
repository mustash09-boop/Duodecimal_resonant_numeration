from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(r"E:\Duodecimal_resonant_numeration")
PY_ROOT = PROJECT_ROOT / "py"
BLOCK_ROOT = PROJECT_ROOT / "Block004_data"
REPORTS_DIR = PROJECT_ROOT / "docs" / "reports"
LOG_PATH = REPORTS_DIR / "block004_all_pitched_pdf_spiral_rebuild_2026-06-05_v2.log"
STATE_PATH = REPORTS_DIR / "block004_all_pitched_pdf_spiral_rebuild_2026-06-05_v2_state.json"
LOCK_PATH = REPORTS_DIR / "block004_all_pitched_pdf_spiral_rebuild_2026-06-05_v2.lock"

ANCHOR_TOKEN = "9.A-"
ANCHOR_HZ = "440"
EXCLUDE = {"percussion", "_multi_instrument_compare"}
PIPELINE_STAGES = [
    "box",
    "box_split",
    "note_box_profile",
    "spiral3d",
    "harmonic_chain_spiral3d",
    "relation",
    "passport",
]


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def ensure_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def append_log(message: str) -> None:
    line = f"[{now_text()}] {message}"
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def write_state(
    *,
    status: str,
    current_instrument: str,
    current_stage: str,
    instrument_index: int,
    instrument_count: int,
    completed: list[dict],
    skipped: list[dict],
    failed: list[dict],
) -> None:
    payload = {
        "status": status,
        "current_instrument": current_instrument,
        "current_stage": current_stage,
        "instrument_index": instrument_index,
        "instrument_count": instrument_count,
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "updated_at": now_iso(),
        "log_path": str(LOG_PATH),
    }
    STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def env_with_pythonpath() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "").strip()
    py_root = str(PY_ROOT)
    if existing:
        if py_root not in existing.split(os.pathsep):
            env["PYTHONPATH"] = py_root + os.pathsep + existing
    else:
        env["PYTHONPATH"] = py_root
    return env


def read_lock_pid() -> int | None:
    if not LOCK_PATH.exists():
        return None
    try:
        raw = LOCK_PATH.read_text(encoding="utf-8").strip()
        return int(raw)
    except Exception:
        return None


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_lock() -> None:
    existing = read_lock_pid()
    if existing and existing != os.getpid() and pid_alive(existing):
        raise RuntimeError(f"Another rebuild runner is already active (pid={existing})")
    LOCK_PATH.write_text(str(os.getpid()), encoding="utf-8")


def release_lock() -> None:
    existing = read_lock_pid()
    if existing == os.getpid():
        LOCK_PATH.unlink(missing_ok=True)


def run_command(cmd: list[str]) -> None:
    append_log("RUN " + " ".join(f'"{part}"' if " " in str(part) else str(part) for part in cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env_with_pythonpath(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line.rstrip() + "\n")
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed with return code {rc}")


def run_pipeline_stage(
    *,
    inst: str,
    audio_dir: Path,
    manifest_csv: Path,
    reports_root: Path,
    stage_name: str,
) -> None:
    run_command(
        [
            sys.executable,
            "-m",
            "music12.blocks.Block004_real_instruments.instrument_pipeline_runner_cli",
            "--instrument_name",
            inst,
            "--audio_dir",
            str(audio_dir),
            "--manifest_csv",
            str(manifest_csv),
            "--reports_root",
            str(reports_root),
            "--stages",
            stage_name,
            "--anchor_token",
            ANCHOR_TOKEN,
            "--anchor_hz",
            ANCHOR_HZ,
        ]
    )


def resolve_audio_dir(instrument_root: Path) -> Path | None:
    for rel in ("00_sources/audio_notes_wav", "00_sources/audio_notes"):
        candidate = instrument_root / rel
        if candidate.exists():
            return candidate
    return None


def resolve_manifest_csv(instrument_root: Path, instrument_name: str) -> Path | None:
    manifest_dir = instrument_root / "20_manifest"
    exact = manifest_dir / f"{instrument_name}_manifest_12.csv"
    if exact.exists():
        return exact

    preferred = sorted(
        p for p in manifest_dir.glob("*manifest_12.csv")
        if "subset" not in p.name.lower() and "fixed" not in p.name.lower()
    )
    if preferred:
        return preferred[0]

    any_csv = sorted(manifest_dir.glob("*.csv"))
    if any_csv:
        return any_csv[0]
    return None


def iter_instruments() -> list[Path]:
    return sorted(
        [p for p in BLOCK_ROOT.iterdir() if p.is_dir() and p.name not in EXCLUDE],
        key=lambda p: p.name.lower(),
    )


def main() -> int:
    ensure_dirs()
    acquire_lock()
    try:
        instruments = iter_instruments()
        completed: list[dict] = []
        skipped: list[dict] = []
        failed: list[dict] = []

        append_log("BLOCK004 pitched rebuild v2 started")
        append_log(f"ProjectRoot: {PROJECT_ROOT}")
        append_log(f"Anchor: {ANCHOR_TOKEN} @ {ANCHOR_HZ}")
        append_log("Instruments: " + ", ".join(p.name for p in instruments))

        write_state(
            status="running",
            current_instrument="",
            current_stage="bootstrap",
            instrument_index=0,
            instrument_count=len(instruments),
            completed=completed,
            skipped=skipped,
            failed=failed,
        )

        for idx, inst_dir in enumerate(instruments, start=1):
            inst = inst_dir.name
            reports_root = inst_dir / "10_reports"
            audio_dir = resolve_audio_dir(inst_dir)
            manifest_csv = resolve_manifest_csv(inst_dir, inst)

            if not reports_root.exists() or audio_dir is None or manifest_csv is None:
                skipped.append(
                    {
                        "instrument": inst,
                        "reports_root": str(reports_root),
                        "audio_dir": str(audio_dir) if audio_dir else "",
                        "manifest_csv": str(manifest_csv) if manifest_csv else "",
                        "reason": "missing_required_path",
                        "skipped_at": now_iso(),
                    }
                )
                append_log(f"SKIP {inst}: missing required path")
                write_state(
                    status="running",
                    current_instrument=inst,
                    current_stage="skipped",
                    instrument_index=idx,
                    instrument_count=len(instruments),
                    completed=completed,
                    skipped=skipped,
                    failed=failed,
                )
                continue

            try:
                append_log(f"START {inst} ({idx}/{len(instruments)})")
                append_log(f"reports_root={reports_root}")
                append_log(f"audio_dir={audio_dir}")
                append_log(f"manifest_csv={manifest_csv}")

                write_state(
                    status="running",
                    current_instrument=inst,
                    current_stage="reports_from_existing_dense",
                    instrument_index=idx,
                    instrument_count=len(instruments),
                    completed=completed,
                    skipped=skipped,
                    failed=failed,
                )
                run_command(
                    [
                        sys.executable,
                        "-m",
                        "music12.blocks.Block004_real_instruments.reports_from_existing_dense_cli",
                        "--reports_root",
                        str(reports_root),
                        "--anchor_token",
                        ANCHOR_TOKEN,
                        "--anchor_hz",
                        ANCHOR_HZ,
                    ]
                )

                for stage_name in PIPELINE_STAGES:
                    write_state(
                        status="running",
                        current_instrument=inst,
                        current_stage=stage_name,
                        instrument_index=idx,
                        instrument_count=len(instruments),
                        completed=completed,
                        skipped=skipped,
                        failed=failed,
                    )
                    run_pipeline_stage(
                        inst=inst,
                        audio_dir=audio_dir,
                        manifest_csv=manifest_csv,
                        reports_root=reports_root,
                        stage_name=stage_name,
                    )

                completed.append(
                    {
                        "instrument": inst,
                        "completed_at": now_iso(),
                        "reports_root": str(reports_root),
                        "manifest_csv": str(manifest_csv),
                    }
                )
                append_log(f"DONE {inst}")
                write_state(
                    status="running",
                    current_instrument=inst,
                    current_stage="completed",
                    instrument_index=idx,
                    instrument_count=len(instruments),
                    completed=completed,
                    skipped=skipped,
                    failed=failed,
                )
            except Exception as exc:  # noqa: BLE001
                failed.append(
                    {
                        "instrument": inst,
                        "failed_at": now_iso(),
                        "error": str(exc),
                        "reports_root": str(reports_root),
                        "manifest_csv": str(manifest_csv),
                    }
                )
                append_log(f"FAIL {inst}: {exc}")
                write_state(
                    status="running",
                    current_instrument=inst,
                    current_stage="failed",
                    instrument_index=idx,
                    instrument_count=len(instruments),
                    completed=completed,
                    skipped=skipped,
                    failed=failed,
                )

        final_status = "completed_with_failures" if failed else "completed"
        write_state(
            status=final_status,
            current_instrument="",
            current_stage="finished",
            instrument_index=len(instruments),
            instrument_count=len(instruments),
            completed=completed,
            skipped=skipped,
            failed=failed,
        )
        append_log(
            f"FINISH status={final_status}; completed={len(completed)}; skipped={len(skipped)}; failed={len(failed)}"
        )
        return 0
    finally:
        release_lock()


if __name__ == "__main__":
    raise SystemExit(main())
