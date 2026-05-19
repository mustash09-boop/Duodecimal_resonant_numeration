from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(r"E:\Duodecimal_resonant_numeration")


def extract_note(folder_name: str) -> str:
    m = re.search(r"([1-9ABC]+\.[1-9ABC]+-)$", folder_name, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot extract note from folder: {folder_name}")
    return m.group(1).upper()


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def process_folder(folder: Path, box_csv: Path) -> None:
    dense_files = sorted(folder.glob("*__dense.csv"))
    if not dense_files:
        print(f"[SKIP] no dense: {folder.name}")
        return

    dense_csv = dense_files[0]
    note = extract_note(folder.name)

    clean_csv = folder / f"{folder.name}__dense_unified_clean.csv"
    removed_csv = folder / f"{folder.name}__dense_unified_removed_box.csv"
    clean_summary = folder / f"{folder.name}__dense_unified_clean_summary.txt"

    spiral_csv = folder / f"{folder.name}__spiral12_clean_points.csv"
    spiral_png = folder / f"{folder.name}__spiral12_clean.png"

    print(f"\n=== {folder.name} -> {note} ===")

    run([
        sys.executable, "-m", "music12.blocks.Block004_real_instruments.unified_dense_note_cleaner_cli",
        "--dense_csv", str(dense_csv),
        "--box_csv", str(box_csv),
        "--out_clean_csv", str(clean_csv),
        "--out_removed_csv", str(removed_csv),
        "--out_summary_txt", str(clean_summary),
        "--expected_note", note,
        "--anchor_token", "9.A-",
        "--anchor_hz", "440",
        "--max_harmonic", "12",
        "--max_freq_hz", "21000",
        "--protect_tolerance_cents_low", "45",
        "--protect_tolerance_cents_mid", "32",
        "--protect_tolerance_cents_high", "28",
        "--box_tolerance_hz", "2.5",
        "--min_box_percent_notes", "70",
        "--min_box_amp", "0",
        "--min_box_hz", "10",
        "--max_box_hz", "320",
    ])

    run([
        sys.executable, "-m", "music12.blocks.Block002_audio_recogn.spiral12_from_dense_clean_cli",
        "--dense_csv", str(clean_csv),
        "--out_csv", str(spiral_csv),
        "--out_png", str(spiral_png),
        "--anchor_token", "9.A-",
        "--anchor_hz", "440",
        "--title", f"{folder.name} 12-radix clean spiral",
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports_root", required=True)
    ap.add_argument("--box_csv", required=True)
    args = ap.parse_args()

    reports_root = Path(args.reports_root).resolve()
    box_csv = Path(args.box_csv).resolve()

    for folder in sorted([p for p in reports_root.iterdir() if p.is_dir()]):
        process_folder(folder, box_csv)


if __name__ == "__main__":
    main()