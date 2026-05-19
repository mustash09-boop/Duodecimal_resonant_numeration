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


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _time_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _same_pitch_or_octave(a: str, b: str) -> bool:
    """
    Compare only degree symbol:
      7.3'- == 8.3'- == 9.3'-
    """
    try:
        da = a.split(".", 1)[1].split("'", 1)[0]
        db = b.split(".", 1)[1].split("'", 1)[0]
        return da == db
    except Exception:
        return False


def _get_reference_id(row: Dict[str, Any], fallback_index: int) -> int:
    """
    Reference CSVs in this project may use different event id column names.
    We support the common variants and fall back to row number.
    """
    for key in ("event_id", "event_index", "id"):
        value = row.get(key, "")
        if str(value).strip() != "":
            return _safe_int(value, fallback_index)
    return fallback_index


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare persistent detected notes against MIDI reference events."
    )

    ap.add_argument("--persistent_events_csv", required=True)
    ap.add_argument("--reference_events_csv", required=True)

    ap.add_argument("--out_matches_csv", required=True)
    ap.add_argument("--out_summary_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--min_overlap_sec", type=float, default=0.03)

    args = ap.parse_args()

    persistent_csv = Path(args.persistent_events_csv)
    reference_csv = Path(args.reference_events_csv)

    out_matches = Path(args.out_matches_csv)
    out_summary = Path(args.out_summary_csv)
    out_meta = Path(args.out_meta_json)
    out_txt = Path(args.out_summary_txt)

    detected = _load_csv(persistent_csv)
    reference = _load_csv(reference_csv)

    # Add stable internal reference ids so one MIDI event cannot be reused endlessly.
    reference_with_ids = []
    for i, row in enumerate(reference, start=1):
        r = dict(row)
        r["_reference_id"] = _get_reference_id(row, i)
        reference_with_ids.append(r)

    match_rows = []

    matched_reference_ids = set()
    matched_detected_ids = set()
    used_reference_ids = set()

    octave_matches = 0
    exact_matches = 0

    for d in detected:
        did = _safe_int(d.get("event_id"), 0)

        dnote = str(d.get("note_token", "")).strip()
        ds = _safe_float(d.get("time_start_sec"), 0.0)
        de = _safe_float(d.get("time_end_sec"), 0.0)

        best = None
        best_overlap = 0.0

        for r in reference_with_ids:
            rid = _safe_int(r.get("_reference_id"), 0)

            # IMPORTANT:
            # one MIDI reference event can only be matched once
            if rid in used_reference_ids:
                continue

            rnote = str(r.get("note_token", "")).strip()
            rs = _safe_float(r.get("time_start_sec"), 0.0)
            re = _safe_float(r.get("time_end_sec"), 0.0)

            overlap = _time_overlap(ds, de, rs, re)

            if overlap < args.min_overlap_sec:
                continue

            if not _same_pitch_or_octave(dnote, rnote):
                continue

            if overlap > best_overlap:
                best_overlap = overlap
                best = r

        if best is None:
            match_rows.append({
                "detected_event_id": did,
                "detected_note": dnote,
                "detected_start": f"{ds:.9f}",
                "detected_end": f"{de:.9f}",
                "reference_id": "",
                "reference_note": "",
                "reference_start": "",
                "reference_end": "",
                "overlap_sec": "0.0",
                "match_type": "NO_MATCH",
            })
            continue

        rid = _safe_int(best.get("_reference_id"), 0)

        rnote = str(best.get("note_token", "")).strip()
        rs = _safe_float(best.get("time_start_sec"), 0.0)
        re = _safe_float(best.get("time_end_sec"), 0.0)

        matched_reference_ids.add(rid)
        matched_detected_ids.add(did)
        used_reference_ids.add(rid)

        if dnote == rnote:
            mtype = "EXACT"
            exact_matches += 1
        else:
            mtype = "OCTAVE_RELATED"
            octave_matches += 1

        match_rows.append({
            "detected_event_id": did,
            "detected_note": dnote,
            "detected_start": f"{ds:.9f}",
            "detected_end": f"{de:.9f}",
            "reference_id": rid,
            "reference_note": rnote,
            "reference_start": f"{rs:.9f}",
            "reference_end": f"{re:.9f}",
            "overlap_sec": f"{best_overlap:.9f}",
            "match_type": mtype,
        })

    precision = len(matched_detected_ids) / max(len(detected), 1)
    recall = len(matched_reference_ids) / max(len(reference), 1)

    summary_rows = [{
        "detected_events": len(detected),
        "reference_events": len(reference),
        "matched_detected_events": len(matched_detected_ids),
        "matched_reference_events": len(matched_reference_ids),
        "exact_matches": exact_matches,
        "octave_related_matches": octave_matches,
        "precision": f"{precision:.6f}",
        "recall": f"{recall:.6f}",
    }]

    out_matches.parent.mkdir(parents=True, exist_ok=True)

    match_fields = [
        "detected_event_id",
        "detected_note",
        "detected_start",
        "detected_end",
        "reference_id",
        "reference_note",
        "reference_start",
        "reference_end",
        "overlap_sec",
        "match_type",
    ]

    with out_matches.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=match_fields)
        w.writeheader()
        w.writerows(match_rows)

    summary_fields = [
        "detected_events",
        "reference_events",
        "matched_detected_events",
        "matched_reference_events",
        "exact_matches",
        "octave_related_matches",
        "precision",
        "recall",
    ]

    with out_summary.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(summary_rows)

    meta = {
        "stage": "persistent_notes_vs_midi_reference",
        "matching": "greedy_one_to_one_reference_matching",
        "inputs": {
            "persistent_events_csv": str(persistent_csv),
            "reference_events_csv": str(reference_csv),
        },
        "outputs": {
            "matches_csv": str(out_matches),
            "summary_csv": str(out_summary),
            "meta_json": str(out_meta),
            "summary_txt": str(out_txt),
        },
        "parameters": {
            "min_overlap_sec": args.min_overlap_sec,
        },
        "result": summary_rows[0],
    }

    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    txt = []
    txt.append("PERSISTENT NOTES VS MIDI REFERENCE")
    txt.append("=" * 72)
    txt.append(f"persistent_events_csv    : {persistent_csv}")
    txt.append(f"reference_events_csv     : {reference_csv}")
    txt.append("")
    txt.append("Matching mode:")
    txt.append("  greedy one-to-one reference matching")
    txt.append("  one MIDI reference event can only be matched once")
    txt.append("")
    txt.append(f"detected_events          : {len(detected)}")
    txt.append(f"reference_events         : {len(reference)}")
    txt.append(f"matched_detected_events  : {len(matched_detected_ids)}")
    txt.append(f"matched_reference_events : {len(matched_reference_ids)}")
    txt.append(f"exact_matches            : {exact_matches}")
    txt.append(f"octave_related_matches   : {octave_matches}")
    txt.append(f"precision                : {precision:.6f}")
    txt.append(f"recall                   : {recall:.6f}")
    txt.append("")
    txt.append("Principle:")
    txt.append("  Compare persistent resonance-derived note events")
    txt.append("  against MIDI reference musical events.")
    txt.append("")

    out_txt.write_text("\n".join(txt), encoding="utf-8")

    print("persistent notes vs midi reference complete")
    print(json.dumps(summary_rows[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()