from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(r"E:\Duodecimal_resonant_numeration")
BLOCK004_ROOT = PROJECT_ROOT / "Block004_data"
PYTHONPATH = str(PROJECT_ROOT / "py")
CLI_SCRIPT = PROJECT_ROOT / "py" / "music12" / "blocks" / "Block004_real_instruments" / "harmonic_chain_spiral3d_builder_cli.py"
RUN_STATE = PROJECT_ROOT / "_long_runs" / "block004_harmonic_chain_all_tonal"


def infer_instrument_name(dataset_dir: Path) -> str:
    range_dir = dataset_dir / "20_range_research"
    passport_jsons = sorted(range_dir.glob("*__instrument_passport.json"))
    if passport_jsons:
        try:
            data = json.loads(passport_jsons[0].read_text(encoding="utf-8"))
            name = str(data.get("instrument_name", "")).strip()
            if name:
                return name
        except Exception:
            pass

    manifests = sorted((dataset_dir / "20_manifest").glob("*.csv"))
    for m in manifests:
        name = m.name
        name = name.replace("_fixed_subset_manifest_12.csv", "")
        name = name.replace("_manifest_12.csv", "")
        if name:
            return name.lower()
    return dataset_dir.name.lower().replace("-", "_")


def tonal_dataset_dirs() -> list[Path]:
    out: list[Path] = []
    for d in sorted(BLOCK004_ROOT.iterdir()):
        if not d.is_dir():
            continue
        if d.name.startswith("_"):
            continue
        if d.name == "percussion":
            continue
        if not (d / "50_spiral3d").exists():
            continue
        out.append(d)
    return out


def source_points_count(spiral_dir: Path) -> int:
    return len(list(spiral_dir.glob("*__spiral3d_points.csv")))


def output_points_count(out_dir: Path) -> int:
    if not out_dir.exists():
        return 0
    return len(list(out_dir.glob("*_points.csv")))


def write_progress(status: dict) -> None:
    RUN_STATE.mkdir(parents=True, exist_ok=True)
    (RUN_STATE / "progress.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    datasets = tonal_dataset_dirs()
    status: dict = {
        "state": "running",
        "project_root": str(PROJECT_ROOT),
        "datasets_total": len(datasets),
        "datasets_done": 0,
        "current_dataset": "",
        "datasets": [],
    }

    env = dict(**os.environ)
    env["PYTHONPATH"] = PYTHONPATH

    work_items: list[tuple[int, Path, str, Path, Path]] = []
    for i, dataset_dir in enumerate(datasets, start=1):
        instrument_name = infer_instrument_name(dataset_dir)
        spiral_dir = dataset_dir / "50_spiral3d"
        out_dir = dataset_dir / "55_harmonic_chain_spiral3d"
        src_count = source_points_count(spiral_dir)
        out_count = output_points_count(out_dir)
        is_done = src_count > 0 and out_count >= src_count

        item = {
            "dataset_dir": str(dataset_dir),
            "instrument_name": instrument_name,
            "spiral3d_dir": str(spiral_dir),
            "out_dir": str(out_dir),
            "source_points": src_count,
            "output_points": out_count,
            "state": "done" if is_done else "pending",
            "index": i,
        }
        status["datasets"].append(item)
        work_items.append((i, dataset_dir, instrument_name, spiral_dir, out_dir))

    status["datasets_done"] = sum(1 for item in status["datasets"] if item["state"] == "done")
    write_progress(status)

    for i, dataset_dir, instrument_name, spiral_dir, out_dir in work_items:
        item = status["datasets"][i - 1]
        if item["state"] == "done":
            continue

        status["current_dataset"] = dataset_dir.name
        item["state"] = "running"
        write_progress(status)

        cmd = [
            sys.executable,
            str(CLI_SCRIPT),
            "--instrument_name",
            instrument_name,
            "--spiral3d_dir",
            str(spiral_dir),
            "--out_dir",
            str(out_dir),
            "--skip_existing",
        ]
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)

        item["returncode"] = int(proc.returncode)
        item["output_points"] = output_points_count(out_dir)
        item["state"] = "done" if proc.returncode == 0 else "failed"
        status["datasets_done"] = sum(1 for x in status["datasets"] if x["state"] == "done")
        write_progress(status)

        if proc.returncode != 0:
            status["state"] = "failed"
            write_progress(status)
            raise SystemExit(proc.returncode)

    status["state"] = "done"
    status["current_dataset"] = ""
    write_progress(status)


if __name__ == "__main__":
    main()
