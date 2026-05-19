# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return default


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit how event-field routed proto-exciters overlap with EXACT_BIRTH + MISSED_SUSTAIN bridge-takeover windows."
    )
    ap.add_argument("--branch-analysis-csv", required=True)
    ap.add_argument("--proto-exciters-csv", required=True)
    ap.add_argument("--breakdown-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-overlaps-csv", required=True)
    ap.add_argument("--out-meta-json", required=True)
    args = ap.parse_args()

    analysis_rows = _load_csv(Path(args.branch_analysis_csv))
    proto_rows = _load_csv(Path(args.proto_exciters_csv))
    breakdown_rows = _load_csv(Path(args.breakdown_csv))

    proto_by_id = {str(row.get("proto_exciter_id", "")): row for row in proto_rows}
    event_field_analysis = [
        row for row in analysis_rows
        if str(row.get("route_label", "")).strip() in {"event_only", "event_field_candidate"}
    ]
    fallback_analysis = [
        row for row in analysis_rows
        if str(row.get("route_label", "")).strip() == "notechain_fallback"
    ]

    overlaps: list[dict[str, Any]] = []
    overlap_type_counter: Counter[str] = Counter()

    for event in breakdown_rows:
        start_frame = _safe_int(event.get("start_frame"), 0)
        end_frame = _safe_int(event.get("end_frame"), start_frame)
        collapse_type = str(event.get("collapse_type", "")).strip()
        for row in event_field_analysis:
            proto = proto_by_id.get(str(row.get("proto_exciter_id", "")))
            if not proto:
                continue
            proto_start = _safe_int(proto.get("start_frame"), 0)
            proto_end = _safe_int(proto.get("end_frame"), proto_start)
            if proto_start <= end_frame and proto_end >= start_frame:
                overlap_type_counter[str(row.get("route_label", "")).strip()] += 1
                overlaps.append(
                    {
                        "event_id": event.get("event_id", ""),
                        "expected_note": event.get("expected_note", ""),
                        "collapse_type": collapse_type,
                        "proto_exciter_id": proto.get("proto_exciter_id", ""),
                        "route_label": row.get("route_label", ""),
                        "route_reason": row.get("route_reason", ""),
                        "coarse_note": proto.get("coarse_note", ""),
                        "proto_start_frame": proto.get("start_frame", ""),
                        "proto_end_frame": proto.get("end_frame", ""),
                        "duration_frames": proto.get("duration_frames", ""),
                        "exciter_confidence": proto.get("exciter_confidence", ""),
                        "total_seed_score": proto.get("total_seed_score", ""),
                        "matched_frames": row.get("matched_frames", ""),
                        "support_ratio": row.get("support_ratio", ""),
                    }
                )

    overlaps.sort(key=lambda r: (_safe_int(r.get("event_id"), 0), _safe_int(r.get("proto_start_frame"), 0), _safe_int(r.get("proto_exciter_id"), 0)))
    out_overlaps = Path(args.out_overlaps_csv)
    out_overlaps.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "event_id",
        "expected_note",
        "collapse_type",
        "proto_exciter_id",
        "route_label",
        "route_reason",
        "coarse_note",
        "proto_start_frame",
        "proto_end_frame",
        "duration_frames",
        "exciter_confidence",
        "total_seed_score",
        "matched_frames",
        "support_ratio",
    ]
    with out_overlaps.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(overlaps)

    summary_lines = [
        "EVENT FIELD ROUTE / BRIDGE OVERLAP AUDIT",
        "=" * 72,
        f"event_field_analysis_rows : {len(event_field_analysis)}",
        f"notechain_fallback_rows   : {len(fallback_analysis)}",
        f"breakdown_events          : {len(breakdown_rows)}",
        f"event_field_overlap_rows  : {len(overlaps)}",
        f"distinct_overlap_events   : {len({str(r.get('event_id', '')) for r in overlaps})}",
        f"distinct_overlap_proto    : {len({str(r.get('proto_exciter_id', '')) for r in overlaps})}",
    ]
    for key in sorted(overlap_type_counter):
        summary_lines.append(f"{key:<24}: {overlap_type_counter[key]}")
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    meta = {
        "stage": "event_field_route_bridge_audit",
        "inputs": {
            "branch_analysis_csv": args.branch_analysis_csv,
            "proto_exciters_csv": args.proto_exciters_csv,
            "breakdown_csv": args.breakdown_csv,
        },
        "result": {
            "event_field_analysis_rows": len(event_field_analysis),
            "notechain_fallback_rows": len(fallback_analysis),
            "breakdown_events": len(breakdown_rows),
            "event_field_overlap_rows": len(overlaps),
            "distinct_overlap_events": len({str(r.get("event_id", "")) for r in overlaps}),
            "distinct_overlap_proto": len({str(r.get("proto_exciter_id", "")) for r in overlaps}),
            "overlap_type_counts": dict(overlap_type_counter),
        },
    }
    Path(args.out_meta_json).write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
