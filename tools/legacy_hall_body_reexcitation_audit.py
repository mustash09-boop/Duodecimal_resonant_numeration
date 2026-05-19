# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


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


def _neighbor_ratio(row: Dict[str, Any], by_id: Dict[str, Dict[str, Any]]) -> float:
    ratios: List[float] = []
    my_score = _safe_float(row.get("mean_score"), 0.0)
    for key in ("same_note_prev_id", "same_note_next_id"):
        nid = str(row.get(key, "")).strip()
        if not nid:
            continue
        other = by_id.get(nid)
        if not other:
            continue
        other_score = _safe_float(other.get("mean_score"), 0.0)
        if other_score > 0:
            ratios.append(my_score / other_score)
    if not ratios:
        return -1.0
    return min(ratios)


def _classify(row: Dict[str, Any], by_id: Dict[str, Dict[str, Any]]) -> str:
    residual = str(row.get("residual_fragmentation_class", ""))
    overlap = _safe_int(row.get("same_note_overlap_frames"), 0)
    gap = _safe_int(row.get("same_note_min_gap"), -1)
    group = _safe_int(row.get("birth_group_size"), 1)
    re_count = _safe_int(row.get("re_excitation_count"), 0)
    duration = _safe_int(row.get("duration_frames"), 0)
    frame_count = _safe_int(row.get("frame_count"), 0)
    coherence = _safe_float(row.get("internal_coherence_score"), 0.0)
    energy_span = _safe_float(row.get("relative_energy_span"), 0.0)
    mean_score = _safe_float(row.get("mean_score"), 0.0)
    birth_score = _safe_float(row.get("birth_score"), 0.0)
    final_score = _safe_float(row.get("final_score"), 0.0)
    ratio = _neighbor_ratio(row, by_id)

    if residual == "STABLE_BACKBONE":
        return "PRIMARY_NOTE_BACKBONE"

    if residual == "SAME_NOTE_NEAR_REBIRTH":
        if overlap >= 8 and (re_count >= 3 or energy_span >= 0.25 or coherence >= 0.82):
            return "LIKELY_INTERNAL_WAVE"
        if group >= 3 and birth_score >= 3.45 and mean_score >= 3.70:
            return "LIKELY_TRUE_REEXCITATION"
        if overlap >= 3 and energy_span < 0.22 and ratio >= 0.78:
            return "LIKELY_INSTRUMENT_BODY_RETURN"
        if overlap >= 3:
            return "LIKELY_INSTRUMENT_BODY_RETURN"
        return "UNRESOLVED_NEAR_RETURN"

    if residual == "VERY_SHORT_EVENT":
        if duration <= 4 and frame_count <= 4 and mean_score < 3.15 and final_score <= birth_score:
            return "LIKELY_HALL_OR_FIELD_TRACE"
        if gap >= 1 and gap <= 6 and ratio >= 0.0 and ratio < 0.75:
            return "LIKELY_HALL_RETURN"
        return "LIKELY_SHORT_RESONANCE_TRACE"

    if residual == "WEAK_CLUSTER_MEMBER":
        if group >= 2 and mean_score < 3.45:
            return "LIKELY_HALL_OR_FIELD_TRACE"
        if group >= 2 and birth_score >= 3.45:
            return "LIKELY_TRUE_REEXCITATION"
        return "LIKELY_SHORT_RESONANCE_TRACE"

    if residual == "INTERNAL_WAVE_HEAVY":
        return "LIKELY_INTERNAL_WAVE"

    if residual == "TRACE_OR_FRAGMENT":
        return "LIKELY_HALL_OR_FIELD_TRACE"

    if residual == "DENSE_ONSET_CLUSTER":
        return "LIKELY_TRUE_REEXCITATION"

    return "UNRESOLVED"


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit likely hall/body/re-excitation causes in residual live-piano events.")
    ap.add_argument("--residual-audit-csv", required=True)
    ap.add_argument("--out-audit-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    args = ap.parse_args()

    rows = _load_csv(Path(args.residual_audit_csv))
    by_id = {str(r.get("merged_event_id", "")).strip(): r for r in rows}

    out_rows: List[Dict[str, Any]] = []
    cause_counts: Dict[str, int] = defaultdict(int)
    residual_to_cause: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for r in rows:
        cause = _classify(r, by_id)
        cause_counts[cause] += 1
        residual_to_cause[str(r.get("residual_fragmentation_class", ""))][cause] += 1
        rr = dict(r)
        rr["neighbor_score_ratio"] = f"{_neighbor_ratio(r, by_id):.9f}"
        rr["acoustic_cause_class"] = cause
        out_rows.append(rr)

    out_rows.sort(
        key=lambda r: (
            {"LIKELY_INTERNAL_WAVE": 0, "LIKELY_INSTRUMENT_BODY_RETURN": 1, "LIKELY_TRUE_REEXCITATION": 2, "LIKELY_HALL_RETURN": 3, "LIKELY_HALL_OR_FIELD_TRACE": 4, "LIKELY_SHORT_RESONANCE_TRACE": 5, "PRIMARY_NOTE_BACKBONE": 6, "UNRESOLVED_NEAR_RETURN": 7, "UNRESOLVED": 8}.get(str(r.get("acoustic_cause_class", "")), 9),
            _safe_int(r.get("birth_frame"), 0),
            str(r.get("candidate_note", "")),
        )
    )

    _write_csv(Path(args.out_audit_csv), out_rows, out_rows[0].keys())

    lines = [
        "LEGACY HALL / BODY / RE-EXCITATION AUDIT",
        "=" * 72,
        f"residual_audit_csv : {args.residual_audit_csv}",
        f"input_events       : {len(rows)}",
        "",
        "acoustic_cause_class_counts:",
    ]
    for k, v in sorted(cause_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("residual_to_cause_breakdown:")
    for residual, inner in sorted(residual_to_cause.items()):
        lines.append(f"  {residual}:")
        for k, v in sorted(inner.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"    {k}: {v}")

    Path(args.out_summary_txt).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "inputs": {"residual_audit_csv": args.residual_audit_csv},
                "result": {
                    "input_events": len(rows),
                    "acoustic_cause_class_counts": dict(sorted(cause_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
                    "residual_to_cause_breakdown": {
                        rk: dict(sorted(rv.items(), key=lambda kv: (-kv[1], kv[0])))
                        for rk, rv in sorted(residual_to_cause.items())
                    },
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
