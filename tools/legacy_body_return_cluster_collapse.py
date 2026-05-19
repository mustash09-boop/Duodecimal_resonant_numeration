# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
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


def _winner_key(r: Dict[str, Any]) -> Tuple[float, float, int, float, str]:
    return (
        _safe_float(r.get("internal_coherence_score"), 0.0),
        _safe_float(r.get("mean_score"), 0.0),
        _safe_int(r.get("frame_count"), 0),
        _safe_float(r.get("birth_score"), 0.0),
        str(r.get("merged_event_id", "")),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Collapse only multi-event same-note clusters dominated by instrument body return.")
    ap.add_argument("--acoustic-audit-csv", required=True)
    ap.add_argument("--midi-meta-json", required=True)
    ap.add_argument("--out-events-csv", required=True)
    ap.add_argument("--out-support-csv", required=True)
    ap.add_argument("--out-mapping-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--onset-window", type=int, default=3)
    args = ap.parse_args()

    rows = _load_csv(Path(args.acoustic_audit_csv))
    midi_meta = json.loads(Path(args.midi_meta_json).read_text(encoding="utf-8"))
    by_id = {str(r.get("merged_event_id", "")).strip(): r for r in rows}

    target_causes = {"LIKELY_INSTRUMENT_BODY_RETURN", "LIKELY_INTERNAL_WAVE", "LIKELY_TRUE_REEXCITATION"}
    parent = list(range(len(rows)))

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

    id_to_idx = {str(r.get("merged_event_id", "")).strip(): i for i, r in enumerate(rows)}
    for i, r in enumerate(rows):
        if str(r.get("acoustic_cause_class", "")) not in target_causes:
            continue
        note = str(r.get("candidate_note", "")).strip()
        for key in ("same_note_prev_id", "same_note_next_id"):
            nid = str(r.get(key, "")).strip()
            if not nid or nid not in by_id:
                continue
            other = by_id[nid]
            if str(other.get("candidate_note", "")).strip() != note:
                continue
            if str(other.get("acoustic_cause_class", "")) not in target_causes:
                continue
            union(i, id_to_idx[nid])

    comps: Dict[int, List[int]] = defaultdict(list)
    for i in range(len(rows)):
        comps[find(i)].append(i)

    kept_rows: List[Dict[str, Any]] = []
    support_rows: List[Dict[str, Any]] = []
    mapping_rows: List[Dict[str, Any]] = []
    collapsed_cluster_count = 0
    collapsed_removed_count = 0

    handled: set[int] = set()

    for root, idxs in comps.items():
        if len(idxs) == 1:
            continue
        bucket = [rows[i] for i in idxs]
        cause_counts = Counter(str(r.get("acoustic_cause_class", "")) for r in bucket)
        dominant = cause_counts.most_common(1)[0][0]
        if dominant != "LIKELY_INSTRUMENT_BODY_RETURN":
            continue

        collapsed_cluster_count += 1
        winner = max(bucket, key=_winner_key)
        winner_id = str(winner.get("merged_event_id", ""))

        for src in bucket:
            sid = str(src.get("merged_event_id", ""))
            mapping_rows.append(
                {
                    "cluster_root": str(root),
                    "dominant_cause": dominant,
                    "kept_event_id": winner_id,
                    "source_event_id": sid,
                    "source_cause": str(src.get("acoustic_cause_class", "")),
                    "is_kept": "1" if sid == winner_id else "0",
                }
            )
            if sid == winner_id:
                rr = dict(src)
                rr["acoustic_event_role"] = "BODY_RETURN_CLUSTER_REPRESENTATIVE"
                kept_rows.append(rr)
            else:
                rr = dict(src)
                rr["acoustic_event_role"] = "BODY_RETURN_SUPPORT"
                support_rows.append(rr)
                collapsed_removed_count += 1
            handled.add(id_to_idx[sid])

    for i, r in enumerate(rows):
        if i in handled:
            continue
        rr = dict(r)
        cause = str(r.get("acoustic_cause_class", ""))
        if cause == "PRIMARY_NOTE_BACKBONE":
            rr["acoustic_event_role"] = "PRIMARY_NOTE_EVENT"
        elif cause == "LIKELY_TRUE_REEXCITATION":
            rr["acoustic_event_role"] = "TRUE_REEXCITATION_EVENT"
        elif cause == "LIKELY_INTERNAL_WAVE":
            rr["acoustic_event_role"] = "INTERNAL_WAVE_EVENT"
        elif cause == "LIKELY_HALL_OR_FIELD_TRACE":
            rr["acoustic_event_role"] = "HALL_OR_FIELD_TRACE"
        elif cause == "LIKELY_SHORT_RESONANCE_TRACE":
            rr["acoustic_event_role"] = "SHORT_RESONANCE_TRACE"
        else:
            rr["acoustic_event_role"] = "UNRESOLVED_EVENT"
        kept_rows.append(rr)

    kept_rows.sort(key=lambda r: (_safe_int(r.get("birth_frame"), 0), str(r.get("candidate_note", ""))))
    support_rows.sort(key=lambda r: (_safe_int(r.get("birth_frame"), 0), str(r.get("candidate_note", ""))))

    _write_csv(Path(args.out_events_csv), kept_rows, kept_rows[0].keys())
    _write_csv(Path(args.out_support_csv), support_rows, support_rows[0].keys() if support_rows else kept_rows[0].keys())
    _write_csv(
        Path(args.out_mapping_csv),
        mapping_rows,
        ["cluster_root", "dominant_cause", "kept_event_id", "source_event_id", "source_cause", "is_kept"],
    )

    role_counts = Counter(str(r.get("acoustic_event_role", "")) for r in kept_rows)
    lines = [
        "LEGACY BODY RETURN CLUSTER COLLAPSE",
        "=" * 72,
        f"acoustic_audit_csv         : {args.acoustic_audit_csv}",
        f"input_events               : {len(rows)}",
        f"kept_events                : {len(kept_rows)}",
        f"support_events             : {len(support_rows)}",
        f"collapsed_clusters         : {collapsed_cluster_count}",
        f"removed_from_event_count   : {collapsed_removed_count}",
        "",
        f"target_event_count         : {midi_meta.get('event_count', 0)}",
        f"target_onset_groups        : {midi_meta.get('unique_onset_groups', 0)}",
        f"event_gap_to_target        : {len(kept_rows) - int(midi_meta.get('event_count', 0))}",
        f"soft_onset_groups_kept     : {_soft_group_count(kept_rows, 'birth_frame', args.onset_window)}",
        f"soft_onset_gap_to_target   : {_soft_group_count(kept_rows, 'birth_frame', args.onset_window) - int(midi_meta.get('unique_onset_groups', 0))}",
        "",
        "kept_role_counts:",
    ]
    for k, v in sorted(role_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {k}: {v}")

    Path(args.out_summary_txt).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "inputs": {
                    "acoustic_audit_csv": args.acoustic_audit_csv,
                    "midi_meta_json": args.midi_meta_json,
                },
                "result": {
                    "input_events": len(rows),
                    "kept_events": len(kept_rows),
                    "support_events": len(support_rows),
                    "collapsed_clusters": collapsed_cluster_count,
                    "removed_from_event_count": collapsed_removed_count,
                    "target_event_count": midi_meta.get("event_count", 0),
                    "target_onset_groups": midi_meta.get("unique_onset_groups", 0),
                    "event_gap_to_target": len(kept_rows) - int(midi_meta.get("event_count", 0)),
                    "soft_onset_groups_kept": _soft_group_count(kept_rows, "birth_frame", args.onset_window),
                    "role_counts": dict(sorted(role_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
