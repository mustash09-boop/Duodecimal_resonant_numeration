# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any, List


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _extract_note_token_from_folder(name: str) -> str:
    """
    Supports folders like:
      049_piano_midi_9.A-
      049_9.A-
    Returns:
      9.A'-
    """
    stem = name.strip()

    parts = stem.split("_")
    raw = parts[-1] if parts else stem

    raw = raw.replace("__", "_").strip()

    if "'" in raw:
        return raw

    if raw.endswith("-"):
        return raw[:-1] + "'-"

    return raw + "'-"


def _find_first(folder: Path, patterns: List[str]) -> Path | None:
    for pat in patterns:
        hits = sorted(folder.glob(pat))
        if hits:
            return hits[0]
    return None


def _summarize_report_folder(folder: Path) -> Dict[str, Any]:
    note_token = _extract_note_token_from_folder(folder.name)

    dense_clean = _find_first(folder, ["*__dense_unified_clean.csv"])
    root_summary = _find_first(folder, ["*__root_consensus_summary.txt"])
    root_candidates = _find_first(folder, ["*__root_consensus_candidates.csv"])
    spiral_points = _find_first(folder, ["*__spiral12_clean_points.csv"])

    summary_text = _read_text(root_summary) if root_summary else ""

    has_dense = dense_clean is not None and dense_clean.exists()
    has_root_summary = root_summary is not None and root_summary.exists()
    has_root_candidates = root_candidates is not None and root_candidates.exists()
    has_spiral_points = spiral_points is not None and spiral_points.exists()

    file_count = len([p for p in folder.iterdir() if p.is_file()])

    return {
        "note_token": note_token,
        "report_folder": str(folder),
        "folder_name": folder.name,
        "file_count": file_count,
        "has_dense_unified_clean": int(has_dense),
        "has_root_consensus_summary": int(has_root_summary),
        "has_root_consensus_candidates": int(has_root_candidates),
        "has_spiral12_clean_points": int(has_spiral_points),
        "dense_unified_clean_csv": str(dense_clean) if dense_clean else "",
        "root_consensus_summary_txt": str(root_summary) if root_summary else "",
        "root_consensus_candidates_csv": str(root_candidates) if root_candidates else "",
        "spiral12_clean_points_csv": str(spiral_points) if spiral_points else "",
        "summary_preview": summary_text[:500].replace("\n", " ").replace("\r", " "),
    }


def _load_range_research(range_dir: Path) -> Dict[str, Dict[str, str]]:
    """
    Soft loader: reads any CSV in 20_range_research and indexes rows by note-like columns.
    Does not assume exact old schema.
    """
    out: Dict[str, Dict[str, str]] = {}

    if not range_dir.exists():
        return out

    csv_files = sorted(range_dir.glob("*.csv"))

    for csv_path in csv_files:
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    note = (
                        row.get("note_token")
                        or row.get("note12")
                        or row.get("token")
                        or row.get("expected_note_token")
                        or row.get("expected_note")
                        or ""
                    ).strip()

                    if not note:
                        continue

                    if "'" not in note and note.endswith("-"):
                        note = note[:-1] + "'-"

                    if note not in out:
                        out[note] = {}

                    prefix = csv_path.stem
                    for k, v in row.items():
                        out[note][f"{prefix}.{k}"] = v

        except Exception:
            continue

    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build piano_midi1 single-note reference index for polyphonic recognition diagnostics."
    )

    ap.add_argument("--reports_dir", required=True)
    ap.add_argument("--range_research_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    args = ap.parse_args()

    reports_dir = Path(args.reports_dir)
    range_research_dir = Path(args.range_research_dir)

    out_csv = Path(args.out_csv)
    out_meta_json = Path(args.out_meta_json)
    out_summary_txt = Path(args.out_summary_txt)

    range_index = _load_range_research(range_research_dir)

    rows = []

    for folder in sorted(reports_dir.iterdir()):
        if not folder.is_dir():
            continue

        row = _summarize_report_folder(folder)

        note = row["note_token"]
        range_data = range_index.get(note, {})

        row["has_range_research"] = int(bool(range_data))
        row["range_research_fields_json"] = json.dumps(range_data, ensure_ascii=False)

        rows.append(row)

    fieldnames = [
        "note_token",
        "folder_name",
        "report_folder",
        "file_count",
        "has_dense_unified_clean",
        "has_root_consensus_summary",
        "has_root_consensus_candidates",
        "has_spiral12_clean_points",
        "has_range_research",
        "dense_unified_clean_csv",
        "root_consensus_summary_txt",
        "root_consensus_candidates_csv",
        "spiral12_clean_points_csv",
        "range_research_fields_json",
        "summary_preview",
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    meta = {
        "stage": "piano_midi_reference_index",
        "reports_dir": str(reports_dir),
        "range_research_dir": str(range_research_dir),
        "out_csv": str(out_csv),
        "note_count": len(rows),
        "notes_with_dense": sum(int(r["has_dense_unified_clean"]) for r in rows),
        "notes_with_root_summary": sum(int(r["has_root_consensus_summary"]) for r in rows),
        "notes_with_range_research": sum(int(r["has_range_research"]) for r in rows),
    }

    out_meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    txt = []
    txt.append("PIANO MIDI REFERENCE INDEX")
    txt.append("=" * 72)
    txt.append(f"reports_dir               : {reports_dir}")
    txt.append(f"range_research_dir        : {range_research_dir}")
    txt.append(f"out_csv                   : {out_csv}")
    txt.append(f"note_count                : {meta['note_count']}")
    txt.append(f"notes_with_dense          : {meta['notes_with_dense']}")
    txt.append(f"notes_with_root_summary   : {meta['notes_with_root_summary']}")
    txt.append(f"notes_with_range_research : {meta['notes_with_range_research']}")
    txt.append("")
    txt.append("Purpose:")
    txt.append("  Build a single-note piano_midi1 reference layer for Bach polyphonic diagnostics.")
    txt.append("  This does not infer notes by itself; it indexes prior single-note analysis.")
    txt.append("")

    out_summary_txt.write_text("\n".join(txt), encoding="utf-8")

    print("piano midi reference index complete")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()