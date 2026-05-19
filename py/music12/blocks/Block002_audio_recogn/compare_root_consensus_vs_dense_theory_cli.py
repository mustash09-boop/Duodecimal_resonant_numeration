from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, Any, List


# =========================
# УТИЛИТЫ
# =========================
def stem_of(name: str) -> str:
    return Path(name).stem.replace("'", "").replace('"', "")


def safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def cents_delta(a_hz: float, b_hz: float) -> float:
    if a_hz <= 0 or b_hz <= 0:
        return 0.0
    return 1200.0 * math.log2(a_hz / b_hz)


# =========================
# MANIFEST
# =========================
def load_manifest(manifest_csv: Path) -> Dict[str, Dict[str, Any]]:
    mapping = {}

    with manifest_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("parse_status", "OK") != "OK":
                continue

            stem = stem_of(row["original_filename"])
            mapping[stem] = {
                "note12": row.get("note12", "").strip(),
                "freq_hz": safe_float(row.get("frequency_hz", 0.0)),
            }

    return mapping


# =========================
# ROOT SUMMARY PARSER
# =========================
def parse_summary(summary_path: Path) -> Dict[str, Any]:
    if not summary_path.exists():
        return {}

    text = summary_path.read_text(encoding="utf-8")

    root_token = ""
    root_hz = 0.0

    for line in text.splitlines():
        if "consensus_root_token" in line:
            root_token = line.split(":")[-1].strip()
        if "consensus_root_hz" in line:
            root_hz = safe_float(line.split(":")[-1].strip())

    return {
        "token": root_token,
        "hz": root_hz,
    }


# =========================
# THEORY MATCH
# =========================
def compute_match_ratio(theory_csv: Path) -> float:
    if not theory_csv.exists():
        return 0.0

    total = 0
    match = 0

    with theory_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if row.get("status") == "MATCH":
                match += 1

    return (match / total) if total else 0.0


# =========================
# КЛАССИФИКАЦИЯ
# =========================
def classify(expected_hz: float, detected_hz: float) -> str:
    if detected_hz <= 0:
        return "NO_ROOT"

    delta = cents_delta(detected_hz, expected_hz)

    abs_delta = abs(delta)

    if abs_delta < 25:
        return "OK"

    if abs_delta < 80:
        return "MICRO_SHIFT"

    # октавная ошибка
    ratio = detected_hz / expected_hz
    if abs(math.log2(ratio) - round(math.log2(ratio))) < 0.1:
        return "OCTAVE_ERROR"

    # гармоника
    if 1.8 < ratio < 12:
        return "HARMONIC_ERROR"

    return "WRONG"


# =========================
# ОБРАБОТКА ОДНОЙ НОТЫ
# =========================
def process_folder(folder: Path, manifest_row: Dict[str, Any]) -> Dict[str, Any]:
    stem = folder.name

    expected_note = manifest_row["note12"]
    expected_hz = manifest_row["freq_hz"]

    summary = parse_summary(folder / f"{stem}__root_consensus_summary.txt")

    detected_token = summary.get("token", "")
    detected_hz = summary.get("hz", 0.0)

    match_ratio = compute_match_ratio(folder / f"{stem}__dense_vs_theory.csv")

    status = classify(expected_hz, detected_hz)

    delta = cents_delta(detected_hz, expected_hz) if detected_hz else 0.0

    return {
        "note": stem,
        "expected_note": expected_note,
        "detected_token": detected_token,
        "expected_hz": expected_hz,
        "detected_hz": detected_hz,
        "delta_cents": round(delta, 2),
        "match_ratio": round(match_ratio, 3),
        "status": status,
    }


# =========================
# СБОР
# =========================
def collect(reports_root: Path, manifest: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []

    for folder in reports_root.iterdir():
        if not folder.is_dir():
            continue

        stem = folder.name

        if stem not in manifest:
            continue

        rows.append(process_folder(folder, manifest[stem]))

    return rows


# =========================
# CSV
# =========================
def write_csv(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# =========================
# TXT
# =========================
def write_txt(path: Path, rows: List[Dict[str, Any]]):
    lines = []
    lines.append("VIOLIN ROOT VALIDATION")
    lines.append("=" * 100)
    lines.append("")

    stats = {}

    for r in rows:
        stats[r["status"]] = stats.get(r["status"], 0) + 1

        lines.append(
            f"{r['note']}: exp={r['expected_note']} "
            f"det={r['detected_token']} "
            f"Δ={r['delta_cents']}c "
            f"match={r['match_ratio']} "
            f"{r['status']}"
        )

    lines.append("")
    lines.append("SUMMARY")
    for k, v in stats.items():
        lines.append(f"{k:15} : {v}")

    path.write_text("\n".join(lines), encoding="utf-8")


# =========================
# MAIN
# =========================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--reports_root", required=True)
    ap.add_argument("--manifest_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_txt", required=True)

    args = ap.parse_args()

    manifest = load_manifest(Path(args.manifest_csv))
    rows = collect(Path(args.reports_root), manifest)

    write_csv(Path(args.out_csv), rows)
    write_txt(Path(args.out_txt), rows)

    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()