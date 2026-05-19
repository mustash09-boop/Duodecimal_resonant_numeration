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


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _normalize(note: str) -> str:
    s = str(note or "").strip()
    if not s:
        return ""
    if "'" in s:
        return s.split("'", 1)[0] + "'-"
    if s.endswith("-"):
        return s[:-1] + "'-"
    return s + "'-"


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    left = max(a0, b0)
    right = min(a1, b1)
    return max(0.0, right - left)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare persistent micro resonance identities vs MIDI reference."
    )

    ap.add_argument("--persistent_csv", required=True)
    ap.add_argument("--reference_csv", required=True)

    ap.add_argument("--out_compare_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--min_overlap_sec", type=float, default=0.05)

    args = ap.parse_args()

    persistent = _load_csv(Path(args.persistent_csv))
    reference = _load_csv(Path(args.reference_csv))

    rows = []

    matched = 0
    false_positive = 0
    missed = 0

    ref_used = set()

    for p in persistent:
        p_note = _normalize(p.get("note_token", ""))

        p0 = _safe_float(p.get("time_start_sec"), 0.0)
        p1 = _safe_float(p.get("time_end_sec"), 0.0)

        best = None
        best_overlap = 0.0

        for idx, r in enumerate(reference):
            r_note = _normalize(r.get("note_token", ""))

            if p_note != r_note:
                continue

            r0 = _safe_float(r.get("time_start_sec"), 0.0)
            r1 = _safe_float(r.get("time_end_sec"), 0.0)

            ov = _overlap(p0, p1, r0, r1)

            if ov > best_overlap:
                best_overlap = ov
                best = (idx, r)

        if best is not None and best_overlap >= args.min_overlap_sec:
            matched += 1
            ref_used.add(best[0])

            status = "MATCH"
        else:
            false_positive += 1
            status = "FALSE_POSITIVE"

        rows.append({
            "status": status,
            "persistent_note": p_note,
            "persistent_start": f"{p0:.9f}",
            "persistent_end": f"{p1:.9f}",
            "best_overlap_sec": f"{best_overlap:.9f}",
            "persistent_mean_score": p.get("mean_score", ""),
        })

    for idx, r in enumerate(reference):
        if idx not in ref_used:
            missed += 1

    out_csv = Path(args.out_compare_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fields = list(rows[0].keys()) if rows else [
            "status",
            "persistent_note",
            "persistent_start",
            "persistent_end",
            "best_overlap_sec",
            "persistent_mean_score",
        ]

        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    precision = matched / max(matched + false_positive, 1)
    recall = matched / max(matched + missed, 1)

    meta = {
        "stage": "micro_vs_midi_reference",
        "inputs": {
            "persistent_csv": args.persistent_csv,
            "reference_csv": args.reference_csv,
        },
        "outputs": {
            "compare_csv": args.out_compare_csv,
            "meta_json": args.out_meta_json,
            "summary_txt": args.out_summary_txt,
        },
        "parameters": {
            "min_overlap_sec": args.min_overlap_sec,
        },
        "result": {
            "persistent_events": len(persistent),
            "reference_events": len(reference),
            "matched": matched,
            "false_positive": false_positive,
            "missed": missed,
            "precision": precision,
            "recall": recall,
        },
    }

    Path(args.out_meta_json).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    txt = [
        "MICRO VS MIDI REFERENCE",
        "=" * 72,
        f"persistent_csv  : {args.persistent_csv}",
        f"reference_csv   : {args.reference_csv}",
        "",
        f"persistent_events : {len(persistent)}",
        f"reference_events  : {len(reference)}",
        "",
        f"matched           : {matched}",
        f"false_positive    : {false_positive}",
        f"missed            : {missed}",
        "",
        f"precision         : {precision:.6f}",
        f"recall            : {recall:.6f}",
        "",
        "Principle:",
        "  Compare persistent resonance identities",
        "  against MIDI reference structure.",
        "",
    ]

    Path(args.out_summary_txt).write_text(
        "\n".join(txt),
        encoding="utf-8",
    )

    print("micro vs midi comparison complete")
    print(json.dumps(meta["result"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()