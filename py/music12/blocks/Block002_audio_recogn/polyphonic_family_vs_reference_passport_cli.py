# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _load_reference_index(path: Path) -> Dict[str, Dict[str, Any]]:
    out = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            note = str(row.get("note_token", "")).strip()
            if note:
                out[note] = row
    return out


def _parse_family_members(raw: str) -> List[str]:
    return [x.strip() for x in str(raw or "").split() if x.strip()]


def _passport_score(family_root: str, members: List[str], ref: Dict[str, Any]) -> float:
    score = 0.0

    if ref:
        score += 0.35

    if _safe_int(ref.get("has_dense_unified_clean", 0), 0):
        score += 0.20

    if _safe_int(ref.get("has_root_consensus_summary", 0), 0):
        score += 0.20

    if _safe_int(ref.get("has_range_research", 0), 0):
        score += 0.20

    if family_root in members:
        score += 0.05

    return min(score, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare polyphonic root families against single-note piano_midi1 reference passport."
    )

    ap.add_argument("--family_csv", required=True)
    ap.add_argument("--reference_index_csv", required=True)

    ap.add_argument("--out_confirmed_csv", required=True)
    ap.add_argument("--out_frame_summary_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--min_passport_score", type=float, default=0.70)

    args = ap.parse_args()

    family_csv = Path(args.family_csv)
    ref_csv = Path(args.reference_index_csv)

    out_confirmed_csv = Path(args.out_confirmed_csv)
    out_frame_summary_csv = Path(args.out_frame_summary_csv)
    out_meta_json = Path(args.out_meta_json)
    out_summary_txt = Path(args.out_summary_txt)

    ref_index = _load_reference_index(ref_csv)

    confirmed_rows = []
    frame_map: Dict[int, List[Dict[str, Any]]] = {}

    with family_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            frame_index = _safe_int(row.get("frame_index", ""), 0)
            time_sec = _safe_float(row.get("time_sec", ""), 0.0)
            family_rank = _safe_int(row.get("family_rank", ""), 0)

            family_root = str(row.get("family_root_note", "")).strip()
            members = _parse_family_members(row.get("family_members", ""))

            ref = ref_index.get(family_root, {})
            score = _passport_score(family_root, members, ref)

            confirmed = score >= args.min_passport_score

            out = {
                "frame_index": frame_index,
                "time_sec": f"{time_sec:.9f}",
                "family_rank": family_rank,
                "family_root_note": family_root,
                "family_members": " ".join(members),
                "passport_score": f"{score:.6f}",
                "passport_confirmed": int(confirmed),
                "reference_found": int(bool(ref)),
                "reference_folder": ref.get("report_folder", ""),
                "has_dense_unified_clean": ref.get("has_dense_unified_clean", ""),
                "has_root_consensus_summary": ref.get("has_root_consensus_summary", ""),
                "has_range_research": ref.get("has_range_research", ""),
                "reference_summary_preview": ref.get("summary_preview", ""),
            }

            confirmed_rows.append(out)
            frame_map.setdefault(frame_index, []).append(out)

    frame_rows = []

    for frame_index in sorted(frame_map):
        rows = frame_map[frame_index]
        time_sec = rows[0]["time_sec"]

        confirmed = [r for r in rows if _safe_int(r["passport_confirmed"], 0) == 1]

        frame_rows.append({
            "frame_index": frame_index,
            "time_sec": time_sec,
            "confirmed_note_count": len(confirmed),
            "confirmed_notes": " | ".join(
                f"{r['family_root_note']}:{r['passport_score']}"
                for r in confirmed
            ),
            "candidate_family_count": len(rows),
        })

    out_confirmed_csv.parent.mkdir(parents=True, exist_ok=True)

    confirmed_fields = [
        "frame_index",
        "time_sec",
        "family_rank",
        "family_root_note",
        "family_members",
        "passport_score",
        "passport_confirmed",
        "reference_found",
        "reference_folder",
        "has_dense_unified_clean",
        "has_root_consensus_summary",
        "has_range_research",
        "reference_summary_preview",
    ]

    with out_confirmed_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=confirmed_fields)
        w.writeheader()
        w.writerows(confirmed_rows)

    summary_fields = [
        "frame_index",
        "time_sec",
        "confirmed_note_count",
        "confirmed_notes",
        "candidate_family_count",
    ]

    with out_frame_summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(frame_rows)

    meta = {
        "stage": "polyphonic_family_vs_reference_passport",
        "inputs": {
            "family_csv": str(family_csv),
            "reference_index_csv": str(ref_csv),
        },
        "outputs": {
            "confirmed_csv": str(out_confirmed_csv),
            "frame_summary_csv": str(out_frame_summary_csv),
            "meta_json": str(out_meta_json),
            "summary_txt": str(out_summary_txt),
        },
        "parameters": {
            "min_passport_score": args.min_passport_score,
        },
        "result": {
            "confirmed_rows": len(confirmed_rows),
            "frame_rows": len(frame_rows),
            "reference_note_count": len(ref_index),
            "max_confirmed_notes_in_frame": max(
                (_safe_int(r["confirmed_note_count"], 0) for r in frame_rows),
                default=0,
            ),
        },
    }

    out_meta_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    txt = []
    txt.append("POLYPHONIC FAMILY VS REFERENCE PASSPORT")
    txt.append("=" * 72)
    txt.append(f"family_csv                  : {family_csv}")
    txt.append(f"reference_index_csv         : {ref_csv}")
    txt.append(f"confirmed_csv               : {out_confirmed_csv}")
    txt.append(f"frame_summary_csv           : {out_frame_summary_csv}")
    txt.append(f"reference_note_count        : {len(ref_index)}")
    txt.append(f"confirmed_rows              : {len(confirmed_rows)}")
    txt.append(f"frame_rows                  : {len(frame_rows)}")
    txt.append(f"max_confirmed_notes_in_frame: {meta['result']['max_confirmed_notes_in_frame']}")
    txt.append("")
    txt.append("Principle:")
    txt.append("  This stage does not decide instrument identity yet.")
    txt.append("  It checks whether each Bach polyphonic root family has a corresponding")
    txt.append("  single-note piano_midi1 passport entry.")
    txt.append("")

    out_summary_txt.write_text("\n".join(txt), encoding="utf-8")

    print("polyphonic family vs reference passport complete")
    print(json.dumps(meta["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()