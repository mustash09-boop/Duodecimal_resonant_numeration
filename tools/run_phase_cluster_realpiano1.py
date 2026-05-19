from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\Alex\Documents\Duodecimal_resonant_numeration")
PY_ROOT = PROJECT_ROOT / "py"

INPUT_DIR = PROJECT_ROOT / r"Laboratory_research\target_root_convergence\csv"
OUTPUT_DIR = PROJECT_ROOT / r"Laboratory_research\phase_clusters"

PYTHON_EXE = sys.executable

ENV = dict(os.environ)
ENV["PYTHONPATH"] = f"{PROJECT_ROOT};{PY_ROOT}"


# ============================================================
# HELPERS
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: list[str]) -> None:
    print("\nRUN:")
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
    subprocess.run(cmd, check=True, env=ENV)


def run_maxwell(module: str, tag: str, extra_args: list[str]) -> None:
    cmd = [
        PYTHON_EXE,
        "-m", "music12.demons.demon_maxwell_cli",
        "-m", module,
        "--task-class", "module_run",
        "--project-root", str(PROJECT_ROOT),
        "--logdir", "_demon_logs",
        "--tag", tag,
        "--",
        *extra_args,
    ]
    run_cmd(cmd)


def is_valid_csv(path: Path, required_columns: list[str] | None = None) -> bool:
    if not path.exists():
        return False
    if path.stat().st_size <= 0:
        return False

    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            if required_columns:
                for col in required_columns:
                    if col not in fieldnames:
                        return False
            first_row = next(reader, None)
            if first_row is None:
                return False
    except Exception:
        return False

    return True


def task_is_complete(note_candidates_csv: Path, report_txt: Path, meta_json: Path) -> bool:
    if not is_valid_csv(
        note_candidates_csv,
        required_columns=[
            "rank",
            "note_token",
            "phase_coherence_score",
            "note_confidence",
            "source_cluster_id",
        ],
    ):
        return False

    if not report_txt.exists() or report_txt.stat().st_size <= 0:
        return False

    if not meta_json.exists() or meta_json.stat().st_size <= 0:
        return False

    return True


def find_input_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*__target_root_convergence.csv"))
    return files


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run phase clustering for all RealPiano_1 target_root_convergence CSV files"
    )
    parser.add_argument(
        "--mode",
        choices=["resume", "force"],
        default="resume",
        help="resume = skip completed notes, force = rerun all notes",
    )
    parser.add_argument("--phase_threshold_deg", type=float, default=25.0)
    parser.add_argument("--phase_lock_threshold_deg", type=float, default=20.0)
    parser.add_argument("--radial_threshold", type=float, default=1.5)
    parser.add_argument("--max_time_gap_60", type=int, default=2)
    parser.add_argument("--min_cluster_size", type=int, default=2)
    parser.add_argument("--max_clusters", type=int, default=16)
    parser.add_argument("--max_notes", type=int, default=8)
    parser.add_argument("--min_phase_coherence_score", type=float, default=0.08)
    parser.add_argument("--min_note_confidence", type=float, default=0.01)
    parser.add_argument(
        "--energy_mode",
        choices=["combined", "convergence", "stabilization"],
        default="combined",
    )
    args = parser.parse_args()

    ensure_dir(OUTPUT_DIR)

    files = find_input_files(INPUT_DIR)
    print(f"TOTAL FILES: {len(files)}")
    print(f"MODE: {args.mode}")

    manifest_rows: list[dict[str, str]] = []

    for input_csv in files:
        prefix = input_csv.name.replace("__target_root_convergence.csv", "")

        phase_clusters_csv = OUTPUT_DIR / f"{prefix}__phase_clusters.csv"
        note_candidates_csv = OUTPUT_DIR / f"{prefix}__phase_note_candidates.csv"
        report_txt = OUTPUT_DIR / f"{prefix}__phase_cluster_report.txt"
        meta_json = OUTPUT_DIR / f"{prefix}__phase_cluster_meta.json"

        print("\n==============================")
        print(f"PROCESS: {prefix}")
        print("==============================")

        if args.mode == "resume" and task_is_complete(note_candidates_csv, report_txt, meta_json):
            print(f"SKIP COMPLETE: {prefix}")

            manifest_rows.append(
                {
                    "prefix": prefix,
                    "input_csv": str(input_csv),
                    "phase_clusters_csv": str(phase_clusters_csv),
                    "note_candidates_csv": str(note_candidates_csv),
                    "report_txt": str(report_txt),
                    "meta_json": str(meta_json),
                    "status": "SKIPPED_COMPLETE",
                }
            )
            continue

        try:
            run_maxwell(
                "music12.blocks.Block002_audio_recogn.dsp_phase_cluster_cli",
                f"{prefix}_phase_cluster",
                [
                    "--input_csv", str(input_csv),
                    "--out_phase_clusters_csv", str(phase_clusters_csv),
                    "--out_note_candidates_csv", str(note_candidates_csv),
                    "--out_txt", str(report_txt),
                    "--out_meta_json", str(meta_json),

                    "--phase_threshold_deg", str(args.phase_threshold_deg),
                    "--phase_lock_threshold_deg", str(args.phase_lock_threshold_deg),
                    "--radial_threshold", str(args.radial_threshold),
                    "--max_time_gap_60", str(args.max_time_gap_60),
                    "--min_cluster_size", str(args.min_cluster_size),
                    "--max_clusters", str(args.max_clusters),
                    "--max_notes", str(args.max_notes),
                    "--min_phase_coherence_score", str(args.min_phase_coherence_score),
                    "--min_note_confidence", str(args.min_note_confidence),
                    "--energy_mode", str(args.energy_mode),
                ],
            )

            manifest_rows.append(
                {
                    "prefix": prefix,
                    "input_csv": str(input_csv),
                    "phase_clusters_csv": str(phase_clusters_csv),
                    "note_candidates_csv": str(note_candidates_csv),
                    "report_txt": str(report_txt),
                    "meta_json": str(meta_json),
                    "status": "OK",
                }
            )

        except Exception as e:
            print(f"\nERROR on {prefix}: {e}")

            manifest_rows.append(
                {
                    "prefix": prefix,
                    "input_csv": str(input_csv),
                    "phase_clusters_csv": str(phase_clusters_csv),
                    "note_candidates_csv": str(note_candidates_csv),
                    "report_txt": str(report_txt),
                    "meta_json": str(meta_json),
                    "status": f"ERROR: {e}",
                }
            )

    manifest_csv = OUTPUT_DIR / "RealPiano_1__phase_cluster_manifest.csv"
    with manifest_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prefix",
                "input_csv",
                "phase_clusters_csv",
                "note_candidates_csv",
                "report_txt",
                "meta_json",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("\n==============================")
    print("DONE")
    print(f"Manifest: {manifest_csv}")
    print("==============================")


if __name__ == "__main__":
    main()