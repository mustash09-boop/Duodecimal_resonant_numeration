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


def _members(raw: str) -> List[str]:
    return [x.strip() for x in str(raw or "").split() if x.strip()]


def _load_family_rows(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def _group_by_frame_and_family(rows: List[Dict[str, Any]]) -> Dict[tuple[int, int], List[Dict[str, Any]]]:
    grouped: Dict[tuple[int, int], List[Dict[str, Any]]] = {}
    for r in rows:
        frame = _safe_int(r.get("frame_index", ""), 0)
        rank = _safe_int(r.get("family_rank", ""), 0)
        grouped.setdefault((frame, rank), []).append(r)
    return grouped


def _range_mode_from_token(note: str) -> str:
    # coarse, project-specific practical approximation
    if note.startswith("5.") or note.startswith("6."):
        return "low"
    if note.startswith("A.") or note.startswith("B.") or note.startswith("C.") or note.startswith("11."):
        return "high"
    return "mid"


def _behavior_score(
    *,
    family_root: str,
    family_members: List[str],
    family_score: float,
    ref: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Conservative behavioural score.

    This first version does not pretend to fully compare curves yet.
    It scores:
    - reference existence
    - availability of dense/root/range data
    - family member richness
    - root present in its own family
    - range-dependent expectations
    """

    reference_found = bool(ref)

    has_dense = _safe_int(ref.get("has_dense_unified_clean", 0), 0)
    has_root = _safe_int(ref.get("has_root_consensus_summary", 0), 0)
    has_range = _safe_int(ref.get("has_range_research", 0), 0)
    has_spiral = _safe_int(ref.get("has_spiral12_clean_points", 0), 0)

    member_count = len(set(family_members))
    root_in_family = family_root in family_members

    range_mode = _range_mode_from_token(family_root)

    score = 0.0

    if reference_found:
        score += 0.15
    if has_dense:
        score += 0.15
    if has_root:
        score += 0.15
    if has_range:
        score += 0.15
    if has_spiral:
        score += 0.10

    if root_in_family:
        score += 0.10

    # harmonic richness
    if range_mode == "low":
        # low notes often need multiple upper supports
        if member_count >= 3:
            score += 0.15
        elif member_count >= 2:
            score += 0.08
    elif range_mode == "high":
        # high notes may have fewer visible harmonics
        if member_count >= 1:
            score += 0.12
    else:
        if member_count >= 3:
            score += 0.12
        elif member_count >= 2:
            score += 0.08

    # family score contribution, capped
    score += min(max(family_score, 0.0), 0.35) * 0.25

    return {
        "behavior_score": min(score, 1.0),
        "range_mode": range_mode,
        "member_count": member_count,
        "root_in_family": int(root_in_family),
        "reference_found": int(reference_found),
        "has_dense_unified_clean": has_dense,
        "has_root_consensus_summary": has_root,
        "has_range_research": has_range,
        "has_spiral12_clean_points": has_spiral,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score polyphonic root families against piano_midi1 behavioural reference passport."
    )

    ap.add_argument("--family_csv", required=True)
    ap.add_argument("--reference_index_csv", required=True)

    ap.add_argument("--out_scored_csv", required=True)
    ap.add_argument("--out_frame_summary_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--min_behavior_score", type=float, default=0.72)
    ap.add_argument("--max_notes_per_frame", type=int, default=8)

    args = ap.parse_args()

    family_csv = Path(args.family_csv)
    ref_csv = Path(args.reference_index_csv)

    out_scored_csv = Path(args.out_scored_csv)
    out_frame_summary_csv = Path(args.out_frame_summary_csv)
    out_meta_json = Path(args.out_meta_json)
    out_summary_txt = Path(args.out_summary_txt)

    ref_index = _load_reference_index(ref_csv)
    family_rows = _load_family_rows(family_csv)
    grouped = _group_by_frame_and_family(family_rows)

    scored_rows = []
    frame_map: Dict[int, List[Dict[str, Any]]] = {}

    for (frame_index, family_rank), rows in sorted(grouped.items()):
        first = rows[0]

        time_sec = _safe_float(first.get("time_sec", ""), 0.0)
        family_root = str(first.get("family_root_note", "")).strip()
        family_score = _safe_float(first.get("family_score", ""), 0.0)

        members = []
        for r in rows:
            m = str(r.get("member_note", "")).strip()
            if m:
                members.append(m)

        members = sorted(set(members))

        ref = ref_index.get(family_root, {})

        behavior = _behavior_score(
            family_root=family_root,
            family_members=members,
            family_score=family_score,
            ref=ref,
        )

        confirmed = behavior["behavior_score"] >= args.min_behavior_score

        out = {
            "frame_index": frame_index,
            "time_sec": f"{time_sec:.9f}",
            "family_rank": family_rank,
            "family_root_note": family_root,
            "family_score": f"{family_score:.9f}",
            "family_members": " ".join(members),

            "behavior_score": f"{behavior['behavior_score']:.9f}",
            "behavior_confirmed": int(confirmed),

            "range_mode": behavior["range_mode"],
            "member_count": behavior["member_count"],
            "root_in_family": behavior["root_in_family"],

            "reference_found": behavior["reference_found"],
            "has_dense_unified_clean": behavior["has_dense_unified_clean"],
            "has_root_consensus_summary": behavior["has_root_consensus_summary"],
            "has_range_research": behavior["has_range_research"],
            "has_spiral12_clean_points": behavior["has_spiral12_clean_points"],

            "reference_folder": ref.get("report_folder", ""),
        }

        scored_rows.append(out)
        frame_map.setdefault(frame_index, []).append(out)

    frame_rows = []

    for frame_index in sorted(frame_map):
        rows = frame_map[frame_index]
        time_sec = rows[0]["time_sec"]

        confirmed = [
            r for r in rows
            if _safe_int(r["behavior_confirmed"], 0) == 1
        ]

        confirmed = sorted(
            confirmed,
            key=lambda r: _safe_float(r["behavior_score"], 0.0),
            reverse=True,
        )[:args.max_notes_per_frame]

        frame_rows.append({
            "frame_index": frame_index,
            "time_sec": time_sec,
            "confirmed_note_count": len(confirmed),
            "confirmed_notes": " | ".join(
                f"{r['family_root_note']}:{r['behavior_score']}"
                for r in confirmed
            ),
            "candidate_family_count": len(rows),
        })

    out_scored_csv.parent.mkdir(parents=True, exist_ok=True)

    scored_fields = [
        "frame_index",
        "time_sec",
        "family_rank",
        "family_root_note",
        "family_score",
        "family_members",
        "behavior_score",
        "behavior_confirmed",
        "range_mode",
        "member_count",
        "root_in_family",
        "reference_found",
        "has_dense_unified_clean",
        "has_root_consensus_summary",
        "has_range_research",
        "has_spiral12_clean_points",
        "reference_folder",
    ]

    with out_scored_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=scored_fields)
        w.writeheader()
        w.writerows(scored_rows)

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
        "stage": "behavioral_passport_similarity",
        "inputs": {
            "family_csv": str(family_csv),
            "reference_index_csv": str(ref_csv),
        },
        "outputs": {
            "scored_csv": str(out_scored_csv),
            "frame_summary_csv": str(out_frame_summary_csv),
            "meta_json": str(out_meta_json),
            "summary_txt": str(out_summary_txt),
        },
        "parameters": {
            "min_behavior_score": args.min_behavior_score,
            "max_notes_per_frame": args.max_notes_per_frame,
        },
        "result": {
            "scored_rows": len(scored_rows),
            "frame_rows": len(frame_rows),
            "reference_note_count": len(ref_index),
            "max_confirmed_notes_in_frame": max(
                (_safe_int(r["confirmed_note_count"], 0) for r in frame_rows),
                default=0,
            ),
            "frames_with_confirmed_notes": sum(
                1 for r in frame_rows if _safe_int(r["confirmed_note_count"], 0) > 0
            ),
        },
    }

    out_meta_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    txt = []
    txt.append("BEHAVIORAL PASSPORT SIMILARITY")
    txt.append("=" * 72)
    txt.append(f"family_csv                  : {family_csv}")
    txt.append(f"reference_index_csv         : {ref_csv}")
    txt.append(f"scored_csv                  : {out_scored_csv}")
    txt.append(f"frame_summary_csv           : {out_frame_summary_csv}")
    txt.append(f"reference_note_count        : {len(ref_index)}")
    txt.append(f"scored_rows                 : {len(scored_rows)}")
    txt.append(f"frame_rows                  : {len(frame_rows)}")
    txt.append(f"max_confirmed_notes_in_frame: {meta['result']['max_confirmed_notes_in_frame']}")
    txt.append(f"frames_with_confirmed_notes : {meta['result']['frames_with_confirmed_notes']}")
    txt.append("")
    txt.append("Principle:")
    txt.append("  This stage begins behavioural passport scoring.")
    txt.append("  It does not merely check whether a note exists in the reference index.")
    txt.append("  It uses range mode, family richness, root presence and available passport layers.")
    txt.append("")

    out_summary_txt.write_text("\n".join(txt), encoding="utf-8")

    print("behavioral passport similarity complete")
    print(json.dumps(meta["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()