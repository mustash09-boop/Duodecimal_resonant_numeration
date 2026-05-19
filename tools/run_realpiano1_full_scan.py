from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

from music12.core.report_naming12 import NoteIdentity, note_report_dir, report_file


# ==============================
# CONFIG
# ==============================

PROJECT_ROOT = Path(r"C:\Users\Alex\Documents\Duodecimal_resonant_numeration")
PY_ROOT = PROJECT_ROOT / "py"

AUDIO_DIR = PROJECT_ROOT / r"Block004_data\RealPiano_1\00_sources\audio_notes"
LIST_FILE = AUDIO_DIR / "List_piano1.txt"
REPORTS_ROOT = PROJECT_ROOT / r"Block004_data\RealPiano_1\00_sources\reports"

DATASET_NAME = "RealPiano_1"

PYTHON_EXE = sys.executable

ENV = dict(os.environ)
ENV["PYTHONPATH"] = str(PY_ROOT)


# ==============================
# HELPERS
# ==============================

def read_list_file(list_file: Path) -> list[str]:
    lines: list[str] = []
    for raw in list_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith(";"):
            continue
        lines.append(line)
    return lines


def find_wav_for_line(line: str, audio_dir: Path) -> Path:
    direct = audio_dir / line
    if direct.exists() and direct.suffix.lower() == ".wav":
        return direct

    if not direct.suffix:
        direct_wav = audio_dir / f"{line}.wav"
        if direct_wav.exists():
            return direct_wav

    candidates = sorted(audio_dir.glob(f"*_{line}.wav"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise RuntimeError(f"Несколько WAV для '{line}': {candidates}")

    raise FileNotFoundError(f"WAV не найден для '{line}'")


def parse_wav_name(wav_path: Path) -> tuple[str, str]:
    stem = wav_path.stem
    m = re.match(r"^(?P<num>\d+)_(?P<note>.+)$", stem)
    if not m:
        raise ValueError(f"Имя файла не в формате NNN_note.wav: {wav_path.name}")
    return m.group("num"), m.group("note")


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
        "--task-class", "audio_analysis",
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


def note_is_complete(stabilized_with_phase_csv: Path) -> bool:
    return is_valid_csv(
        stabilized_with_phase_csv,
        required_columns=[
            "representative_rc_note",
            "representative_rc_hz",
            "phase_deg",
            "radial_level",
        ],
    )


# ==============================
# MAIN
# ==============================

def main() -> None:
    parser = argparse.ArgumentParser(description="Full RealPiano_1 scan runner with resume support")
    parser.add_argument(
        "--mode",
        choices=["resume", "force"],
        default="resume",
        help="resume = skip completed notes, force = rerun all notes",
    )
    args = parser.parse_args()

    ensure_dir(REPORTS_ROOT)

    rows: list[dict[str, str]] = []
    lines = read_list_file(LIST_FILE)

    print(f"TOTAL FILES: {len(lines)}")
    print(f"MODE: {args.mode}")

    for line in lines:
        try:
            wav_path = find_wav_for_line(line, AUDIO_DIR)
            num, note = parse_wav_name(wav_path)

            print("\n==============================")
            print(f"PROCESS: {num} | {note}")
            print("==============================")

            note_id = NoteIdentity(
                index=num,
                instrument=DATASET_NAME,
                note_token=note,
            )

            report_dir = note_report_dir(REPORTS_ROOT, note_id)
            ensure_dir(report_dir)

            # ---- Step 0 outputs: probe ----
            probe_matrix_csv = str(report_file(report_dir, note_id, "__probe_matrix.csv"))
            probe_times_csv = str(report_file(report_dir, note_id, "__probe_times.csv"))
            probe_coords_csv = str(report_file(report_dir, note_id, "__probe_coords.csv"))
            probe_meta_json = str(report_file(report_dir, note_id, "__probe_meta.json"))

            # ---- Step 1 outputs: framewise rc ----
            framewise_csv = str(report_file(report_dir, note_id, "__framewise.csv"))
            framewise_readable_csv = str(report_file(report_dir, note_id, "__framewise_readable.csv"))
            framewise_meta_json = str(report_file(report_dir, note_id, "__framewise_meta.json"))

            # ---- Step 2 outputs: theoretical chain matching ----
            framewise_with_theory_csv = str(report_file(report_dir, note_id, "__framewise__with_theory.csv"))
            framewise_with_theory_meta_json = str(report_file(report_dir, note_id, "__framewise__with_theory_meta.json"))

            # ---- Step 3 outputs: stabilization ----
            stabilized_csv = str(report_file(report_dir, note_id, "__stabilized.csv"))
            stabilized_meta_json = str(report_file(report_dir, note_id, "__stabilized_meta.json"))

            # ---- Step 4 outputs: stabilized + phase ----
            stabilized_with_phase_csv = str(report_file(report_dir, note_id, "__stabilized__with_phase.csv"))
            stabilized_with_phase_meta_json = str(report_file(report_dir, note_id, "__stabilized__with_phase_meta.json"))

            final_csv_path = Path(stabilized_with_phase_csv)

            if args.mode == "resume" and note_is_complete(final_csv_path):
                print(f"SKIP COMPLETE: {num} | {note}")

                rows.append({
                    "num": num,
                    "instrument": DATASET_NAME,
                    "note": note,
                    "wav": str(wav_path),
                    "report_dir": str(report_dir),

                    "probe_matrix_csv": probe_matrix_csv,
                    "probe_times_csv": probe_times_csv,
                    "probe_coords_csv": probe_coords_csv,
                    "probe_meta_json": probe_meta_json,

                    "framewise_csv": framewise_csv,
                    "framewise_readable_csv": framewise_readable_csv,
                    "framewise_meta_json": framewise_meta_json,

                    "framewise_with_theory_csv": framewise_with_theory_csv,
                    "framewise_with_theory_meta_json": framewise_with_theory_meta_json,

                    "stabilized_csv": stabilized_csv,
                    "stabilized_meta_json": stabilized_meta_json,

                    "stabilized_with_phase_csv": stabilized_with_phase_csv,
                    "stabilized_with_phase_meta_json": stabilized_with_phase_meta_json,

                    "status": "SKIPPED_COMPLETE",
                })
                continue

            # ==============================
            # STEP 0: resonance probe (Maxwell)
            # ==============================
            run_maxwell(
                "music12.blocks.Block002_audio_recogn.resonance_probe12_scan_cli",
                f"{num}_{note}_probe",
                [
                    "--wav", str(wav_path),
                    "--out_matrix_csv", probe_matrix_csv,
                    "--out_meta_json", probe_meta_json,
                    "--out_times_csv", probe_times_csv,
                    "--out_coords_csv", probe_coords_csv,
                    "--octave_min", "1",
                    "--octave_max", "C",
                ],
            )

            # ==============================
            # STEP 1: framewise rc inference (Maxwell)
            # ==============================
            run_maxwell(
                "music12.blocks.Block002_audio_recogn.resonance_f0_inference_cli",
                f"{num}_{note}_framewise_rc",
                [
                    "--matrix_csv", probe_matrix_csv,
                    "--times_csv", probe_times_csv,
                    "--coords_csv", probe_coords_csv,
                    "--out_framewise_csv", framewise_csv,
                    "--out_framewise_readable_csv", framewise_readable_csv,
                    "--out_meta_json", framewise_meta_json,
                ],
            )

            # ==============================
            # STEP 2: theoretical chain window match (Maxwell)
            # ==============================
            run_maxwell(
                "music12.blocks.Block002_audio_recogn.theoretical_chain_window_match_cli",
                f"{num}_{note}_theory",
                [
                    "--framewise_csv", framewise_csv,
                    "--out_csv", framewise_with_theory_csv,
                    "--out_meta_json", framewise_with_theory_meta_json,
                    "--window_frames", "2",
                ],
            )

            # ==============================
            # STEP 3: stabilization with theory (Maxwell)
            # ==============================
            run_maxwell(
                "music12.blocks.Block002_audio_recogn.stabilize_f0_chain_cli",
                f"{num}_{note}_stabilize",
                [
                    "--framewise_with_theory_csv", framewise_with_theory_csv,
                    "--out_stabilized_csv", stabilized_csv,
                    "--out_meta_json", stabilized_meta_json,
                    "--window_frames", "4",
                ],
            )

            # ==============================
            # STEP 4: add phase to stabilized (Maxwell)
            # ==============================
            run_maxwell(
                "music12.blocks.Block002_audio_recogn.add_phase_to_stabilized_cli",
                f"{num}_{note}_phase",
                [
                    "--stabilized_csv", stabilized_csv,
                    "--out_csv", stabilized_with_phase_csv,
                    "--out_meta_json", stabilized_with_phase_meta_json,
                ],
            )

            rows.append({
                "num": num,
                "instrument": DATASET_NAME,
                "note": note,
                "wav": str(wav_path),
                "report_dir": str(report_dir),

                "probe_matrix_csv": probe_matrix_csv,
                "probe_times_csv": probe_times_csv,
                "probe_coords_csv": probe_coords_csv,
                "probe_meta_json": probe_meta_json,

                "framewise_csv": framewise_csv,
                "framewise_readable_csv": framewise_readable_csv,
                "framewise_meta_json": framewise_meta_json,

                "framewise_with_theory_csv": framewise_with_theory_csv,
                "framewise_with_theory_meta_json": framewise_with_theory_meta_json,

                "stabilized_csv": stabilized_csv,
                "stabilized_meta_json": stabilized_meta_json,

                "stabilized_with_phase_csv": stabilized_with_phase_csv,
                "stabilized_with_phase_meta_json": stabilized_with_phase_meta_json,

                "status": "OK",
            })

        except Exception as e:
            print(f"\nERROR on {line}: {e}")

            rows.append({
                "num": "",
                "instrument": DATASET_NAME,
                "note": line,
                "wav": "",
                "report_dir": "",

                "probe_matrix_csv": "",
                "probe_times_csv": "",
                "probe_coords_csv": "",
                "probe_meta_json": "",

                "framewise_csv": "",
                "framewise_readable_csv": "",
                "framewise_meta_json": "",

                "framewise_with_theory_csv": "",
                "framewise_with_theory_meta_json": "",

                "stabilized_csv": "",
                "stabilized_meta_json": "",

                "stabilized_with_phase_csv": "",
                "stabilized_with_phase_meta_json": "",

                "status": f"ERROR: {e}",
            })

    manifest_csv = REPORTS_ROOT / f"{DATASET_NAME}__manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "num",
                "instrument",
                "note",
                "wav",
                "report_dir",

                "probe_matrix_csv",
                "probe_times_csv",
                "probe_coords_csv",
                "probe_meta_json",

                "framewise_csv",
                "framewise_readable_csv",
                "framewise_meta_json",

                "framewise_with_theory_csv",
                "framewise_with_theory_meta_json",

                "stabilized_csv",
                "stabilized_meta_json",

                "stabilized_with_phase_csv",
                "stabilized_with_phase_meta_json",

                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n==============================")
    print("DONE")
    print(f"Manifest: {manifest_csv}")
    print("==============================")


if __name__ == "__main__":
    main()