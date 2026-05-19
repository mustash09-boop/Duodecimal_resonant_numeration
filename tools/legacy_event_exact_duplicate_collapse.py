# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        w.writerows(rows)


def _soft_group_count(rows: List[Dict[str, Any]], frame_key: str, window: int) -> int:
    frames = sorted(_safe_int(r.get(frame_key), 0) for r in rows)
    if not frames:
        return 0
    groups = 1
    anchor = frames[0]
    for frame in frames[1:]:
        if frame - anchor > window:
            groups += 1
            anchor = frame
    return groups


def _winner_key(r: Dict[str, Any]) -> Tuple[float, float, int, str]:
    return (
        _safe_float(r.get("internal_coherence_score"), 0.0),
        _safe_float(r.get("mean_score"), 0.0),
        _safe_int(r.get("frame_count"), 0),
        str(r.get("merged_event_id", "")),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Collapse exact same-note same-span duplicate lifecycle events.")
    ap.add_argument("--events-csv", required=True)
    ap.add_argument("--out-events-csv", required=True)
    ap.add_argument("--out-mapping-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--onset-window", type=int, default=3)
    args = ap.parse_args()

    rows = _load_csv(Path(args.events_csv))
    if not rows:
        raise SystemExit("No rows in input events csv")

    groups: Dict[Tuple[str, int, int], List[Dict[str, Any]]] = {}
    for r in rows:
        key = (
            str(r.get("candidate_note", "")).strip(),
            _safe_int(r.get("birth_frame"), 0),
            _safe_int(r.get("end_frame"), 0),
        )
        groups.setdefault(key, []).append(r)

    kept_rows: List[Dict[str, Any]] = []
    mapping_rows: List[Dict[str, Any]] = []
    duplicate_group_count = 0
    duplicate_event_count = 0

    for key, bucket in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        winner = max(bucket, key=_winner_key)
        if len(bucket) > 1:
            duplicate_group_count += 1
            duplicate_event_count += len(bucket)
        kept_rows.append(winner)
        winner_id = str(winner.get("merged_event_id", winner.get("event_id", "")))
        for src in bucket:
            mapping_rows.append(
                {
                    "collapse_group_note": key[0],
                    "collapse_birth_frame": key[1],
                    "collapse_end_frame": key[2],
                    "kept_event_id": winner_id,
                    "source_event_id": str(src.get("merged_event_id", src.get("event_id", ""))),
                    "is_kept": "1" if src is winner else "0",
                }
            )

    kept_rows.sort(key=lambda r: (_safe_int(r.get("birth_frame"), 0), str(r.get("candidate_note", ""))))

    _write_csv(Path(args.out_events_csv), kept_rows, kept_rows[0].keys())
    _write_csv(
        Path(args.out_mapping_csv),
        mapping_rows,
        [
            "collapse_group_note",
            "collapse_birth_frame",
            "collapse_end_frame",
            "kept_event_id",
            "source_event_id",
            "is_kept",
        ],
    )

    summary_lines = [
        "LEGACY EVENT EXACT DUPLICATE COLLAPSE",
        "=" * 72,
        f"events_csv                  : {args.events_csv}",
        f"input_events                : {len(rows)}",
        f"kept_events                 : {len(kept_rows)}",
        f"removed_events              : {len(rows) - len(kept_rows)}",
        f"exact_duplicate_groups      : {duplicate_group_count}",
        f"events_inside_dup_groups    : {duplicate_event_count}",
        f"soft_onset_groups_kept      : {_soft_group_count(kept_rows, 'birth_frame', args.onset_window)}",
    ]
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "inputs": {"events_csv": args.events_csv},
                "outputs": {
                    "events_csv": args.out_events_csv,
                    "mapping_csv": args.out_mapping_csv,
                    "summary_txt": args.out_summary_txt,
                },
                "result": {
                    "input_events": len(rows),
                    "kept_events": len(kept_rows),
                    "removed_events": len(rows) - len(kept_rows),
                    "exact_duplicate_groups": duplicate_group_count,
                    "events_inside_dup_groups": duplicate_event_count,
                    "soft_onset_groups_kept": _soft_group_count(kept_rows, "birth_frame", args.onset_window),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
