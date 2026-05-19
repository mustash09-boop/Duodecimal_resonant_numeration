# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


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
        str(r.get("merged_event_id", r.get("event_id", ""))),
    )


def _near_duplicate(
    a: Dict[str, Any],
    b: Dict[str, Any],
    *,
    min_overlap: float,
    max_start_diff: int,
    max_end_diff: int,
) -> bool:
    if str(a.get("candidate_note", "")).strip() != str(b.get("candidate_note", "")).strip():
        return False
    ab = _safe_int(a.get("birth_frame"), 0)
    ae = _safe_int(a.get("end_frame"), 0)
    bb = _safe_int(b.get("birth_frame"), 0)
    be = _safe_int(b.get("end_frame"), 0)
    inter = max(0, min(ae, be) - max(ab, bb) + 1)
    if inter <= 0:
        return False
    union = max(ae, be) - min(ab, bb) + 1
    overlap = inter / max(union, 1)
    return (
        overlap >= min_overlap
        and abs(ab - bb) <= max_start_diff
        and abs(ae - be) <= max_end_diff
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Collapse cautious near-duplicate same-note lifecycle events.")
    ap.add_argument("--events-csv", required=True)
    ap.add_argument("--out-events-csv", required=True)
    ap.add_argument("--out-mapping-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--onset-window", type=int, default=3)
    ap.add_argument("--min-overlap", type=float, default=0.90)
    ap.add_argument("--max-start-diff", type=int, default=2)
    ap.add_argument("--max-end-diff", type=int, default=2)
    args = ap.parse_args()

    rows = _load_csv(Path(args.events_csv))
    if not rows:
        raise SystemExit("No rows in input events csv")

    n = len(rows)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    rows_sorted_idx = sorted(range(n), key=lambda i: (rows[i].get("candidate_note", ""), _safe_int(rows[i].get("birth_frame"), 0), _safe_int(rows[i].get("end_frame"), 0)))
    for pos, i in enumerate(rows_sorted_idx):
        a = rows[i]
        anote = str(a.get("candidate_note", "")).strip()
        ae = _safe_int(a.get("end_frame"), 0)
        for j in rows_sorted_idx[pos + 1:]:
            b = rows[j]
            bnote = str(b.get("candidate_note", "")).strip()
            if bnote != anote:
                if bnote > anote:
                    break
                continue
            bb = _safe_int(b.get("birth_frame"), 0)
            if bb > ae + args.max_start_diff:
                break
            if _near_duplicate(
                a,
                b,
                min_overlap=args.min_overlap,
                max_start_diff=args.max_start_diff,
                max_end_diff=args.max_end_diff,
            ):
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    kept_rows: List[Dict[str, Any]] = []
    mapping_rows: List[Dict[str, Any]] = []
    near_groups = 0
    removed = 0

    for root, idxs in sorted(groups.items(), key=lambda kv: min(_safe_int(rows[i].get("birth_frame"), 0) for i in kv[1])):
        bucket = [rows[i] for i in idxs]
        winner = max(bucket, key=_winner_key)
        if len(bucket) > 1:
            near_groups += 1
            removed += len(bucket) - 1
        kept_rows.append(winner)
        winner_id = str(winner.get("merged_event_id", winner.get("event_id", "")))
        for src in bucket:
            mapping_rows.append(
                {
                    "kept_event_id": winner_id,
                    "source_event_id": str(src.get("merged_event_id", src.get("event_id", ""))),
                    "candidate_note": str(src.get("candidate_note", "")),
                    "birth_frame": str(src.get("birth_frame", "")),
                    "end_frame": str(src.get("end_frame", "")),
                    "is_kept": "1" if src is winner else "0",
                }
            )

    kept_rows.sort(key=lambda r: (_safe_int(r.get("birth_frame"), 0), str(r.get("candidate_note", ""))))

    _write_csv(Path(args.out_events_csv), kept_rows, kept_rows[0].keys())
    _write_csv(
        Path(args.out_mapping_csv),
        mapping_rows,
        ["kept_event_id", "source_event_id", "candidate_note", "birth_frame", "end_frame", "is_kept"],
    )

    summary_lines = [
        "LEGACY EVENT NEAR DUPLICATE COLLAPSE",
        "=" * 72,
        f"events_csv             : {args.events_csv}",
        f"input_events           : {len(rows)}",
        f"kept_events            : {len(kept_rows)}",
        f"removed_events         : {removed}",
        f"near_duplicate_groups  : {near_groups}",
        f"soft_onset_groups_kept : {_soft_group_count(kept_rows, 'birth_frame', args.onset_window)}",
        f"min_overlap            : {args.min_overlap}",
        f"max_start_diff         : {args.max_start_diff}",
        f"max_end_diff           : {args.max_end_diff}",
    ]
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "inputs": {"events_csv": args.events_csv},
                "parameters": {
                    "min_overlap": args.min_overlap,
                    "max_start_diff": args.max_start_diff,
                    "max_end_diff": args.max_end_diff,
                },
                "result": {
                    "input_events": len(rows),
                    "kept_events": len(kept_rows),
                    "removed_events": removed,
                    "near_duplicate_groups": near_groups,
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
