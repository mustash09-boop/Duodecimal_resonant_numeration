from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def safe_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def expected_note_from_dirname(name: str) -> str:
    """
    Примеры:
      001_piano_midi_5.A-  -> 5.A-
      049_piano_midi_9.A-  -> 9.A-
      001__RealPiano_1__5.A- -> 5.A-
    """
    base = name.strip()

    m = re.match(r"^\d+_piano_midi_(.+)$", base)
    if m:
        return m.group(1)

    m = re.match(r"^\d+__[^_]+(?:_[^_]+)*__(.+)$", base)
    if m:
        return m.group(1)

    m = re.match(r"^\d+_(.+)$", base)
    if m:
        return m.group(1)

    return ""


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_one_folder(folder: Path) -> dict[str, Any] | None:
    name = folder.name
    expected_note = expected_note_from_dirname(name)

    summary_json = folder / f"{name}__chain_summary.json"
    summary_txt = folder / f"{name}__chain_summary.txt"
    candidates_csv = folder / f"{name}__chain_candidates.csv"

    if not summary_json.exists():
        return None

    data = load_json(summary_json)
    best = data.get("best_chain", {}) or {}

    harmonic_indices_found = best.get("harmonic_indices_found", "")
    harmonic_indices_missing = best.get("harmonic_indices_missing", "")

    row = {
        "folder_name": name,
        "expected_note": expected_note,
        "detected_root_note": safe_str(best.get("root_note_token", "")),
        "root_hz": safe_float(best.get("root_hz", 0.0)),
        "harmonic_count_found": safe_int(best.get("harmonic_count_found", 0)),
        "harmonic_indices_found": safe_str(harmonic_indices_found),
        "harmonic_indices_missing": safe_str(harmonic_indices_missing),
        "chain_energy_sum": safe_float(best.get("chain_energy_sum", 0.0)),
        "weighted_support_score": safe_float(best.get("weighted_support_score", 0.0)),
        "subharmonic_penalty": safe_float(best.get("subharmonic_penalty", 0.0)),
        "root_plausibility_score": safe_float(best.get("root_plausibility_score", 0.0)),
        "chain_score": safe_float(best.get("chain_score", 0.0)),
        "summary_json": str(summary_json),
        "summary_txt": str(summary_txt) if summary_txt.exists() else "",
        "candidates_csv": str(candidates_csv) if candidates_csv.exists() else "",
    }

    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = [
        "folder_name",
        "expected_note",
        "detected_root_note",
        "root_hz",
        "harmonic_count_found",
        "harmonic_indices_found",
        "harmonic_indices_missing",
        "chain_energy_sum",
        "weighted_support_score",
        "subharmonic_penalty",
        "root_plausibility_score",
        "chain_score",
        "summary_json",
        "summary_txt",
        "candidates_csv",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_txt(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("CHAIN COLLECTION REPORT")
    lines.append("=" * 100)
    lines.append("")

    if not rows:
        lines.append("No chain summaries found.")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    for row in rows:
        lines.append(f"[{row['folder_name']}]")
        lines.append(f"expected_note           : {row['expected_note']}")
        lines.append(f"detected_root_note      : {row['detected_root_note']}")
        lines.append(f"root_hz                 : {row['root_hz']}")
        lines.append(f"harmonic_count_found    : {row['harmonic_count_found']}")
        lines.append(f"harmonic_indices_found  : {row['harmonic_indices_found']}")
        lines.append(f"harmonic_indices_missing: {row['harmonic_indices_missing']}")
        lines.append(f"chain_energy_sum        : {row['chain_energy_sum']}")
        lines.append(f"weighted_support_score  : {row['weighted_support_score']}")
        lines.append(f"subharmonic_penalty     : {row['subharmonic_penalty']}")
        lines.append(f"root_plausibility_score : {row['root_plausibility_score']}")
        lines.append(f"chain_score             : {row['chain_score']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Collect chain summary results from report folders into one CSV and TXT."
    )
    ap.add_argument("--reports_root", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_txt", required=True)
    args = ap.parse_args()

    reports_root = Path(args.reports_root).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_txt = Path(args.out_txt).resolve()

    rows: list[dict[str, Any]] = []

    for folder in sorted(reports_root.iterdir()):
        if not folder.is_dir():
            continue
        row = collect_one_folder(folder)
        if row is not None:
            rows.append(row)

    rows.sort(key=lambda x: x["folder_name"])

    write_csv(out_csv, rows)
    write_txt(out_txt, rows)

    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote TXT: {out_txt}")
    print(f"Collected folders: {len(rows)}")


if __name__ == "__main__":
    main()