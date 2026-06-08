# -*- coding: ascii -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_progress(path_str: str, payload: dict[str, Any]) -> None:
    if not str(path_str).strip():
        return
    Path(path_str).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a data-grounded second-sustain owner by merging upper sustained observations with octave-lower support found in the same frames."
    )
    ap.add_argument("--ownership-observations-csv", required=True)
    ap.add_argument("--window-start-sec", type=float, required=True)
    ap.add_argument("--window-end-sec", type=float, required=True)
    ap.add_argument("--out-combined-csv", required=True)
    ap.add_argument("--out-support-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--progress-json", default="")
    args = ap.parse_args()

    target_owner = "SECOND_SUSTAIN_DATA_GROUNDED_OWNER"
    allowed_support_owners = {
        "LOCAL_TRANSIENT_OWNER",
        "LOCAL_EVENT_OWNER",
        "BODY_CONTINUATION_OWNER",
        "UNRESOLVED_BACKBONE_OWNER",
    }

    _write_progress(
        args.progress_json,
        {
            "status": "running",
            "phase": "loading_inputs",
            "window_start_sec": args.window_start_sec,
            "window_end_sec": args.window_end_sec,
        },
    )

    rows = _load_csv(Path(args.ownership_observations_csv))
    selected = [
        row for row in rows
        if args.window_start_sec <= _safe_float(row.get("time_sec"), -1.0) <= args.window_end_sec
    ]

    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_frame[_safe_int(row.get("frame_index"), 0)].append(row)

    combined_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    upper_count = 0
    linked_support_count = 0
    support_owner_counter: Counter[str] = Counter()
    support_coarse_counter: Counter[str] = Counter()
    linked_frame_counter: Counter[str] = Counter()

    for frame_index in sorted(by_frame.keys()):
        frame_rows = by_frame[frame_index]
        upper_rows = [r for r in frame_rows if str(r.get("owner_label", "")).strip() == "SECOND_SUSTAIN_OWNER"]
        if not upper_rows:
            continue
        upper_count += len(upper_rows)
        upper_centroid = sum(
            _safe_float(r.get("frequency_hz"), 0.0) * _safe_float(r.get("energy"), 0.0)
            for r in upper_rows
        ) / max(
            sum(_safe_float(r.get("energy"), 0.0) for r in upper_rows),
            1e-9,
        )
        support_low = upper_centroid * 0.46
        support_high = upper_centroid * 0.54

        for row in upper_rows:
            copied = dict(row)
            copied["owner_label"] = target_owner
            copied["owner_family"] = "SECOND_LAYER_OWNER"
            copied["owner_reasons_json"] = json.dumps(
                ["original_second_sustain_owner"],
                ensure_ascii=False,
            )
            copied["data_grounded_support_role"] = "UPPER_DIRECT_SUPPORT"
            copied["source_owner_label"] = str(row.get("owner_label", "")).strip()
            combined_rows.append(copied)

        support_candidates = []
        for row in frame_rows:
            owner_label = str(row.get("owner_label", "")).strip()
            if owner_label not in allowed_support_owners:
                continue
            freq_hz = _safe_float(row.get("frequency_hz"), 0.0)
            energy = _safe_float(row.get("energy"), 0.0)
            if not (support_low <= freq_hz <= support_high):
                continue
            if energy < 0.34:
                continue
            support_candidates.append(row)

        if support_candidates:
            linked_frame_counter["frames_with_support"] += 1
        else:
            linked_frame_counter["frames_without_support"] += 1

        for row in support_candidates:
            owner_label = str(row.get("owner_label", "")).strip()
            copied = dict(row)
            copied["owner_label"] = target_owner
            copied["owner_family"] = "SECOND_LAYER_OWNER"
            copied["owner_reasons_json"] = json.dumps(
                [
                    "data_grounded_suboctave_support",
                    f"upper_centroid_hz={upper_centroid:.6f}",
                    f"support_band_hz={support_low:.6f}-{support_high:.6f}",
                    f"source_owner={owner_label}",
                ],
                ensure_ascii=False,
            )
            copied["data_grounded_support_role"] = "LOWER_OCTAVE_SUPPORT"
            copied["source_owner_label"] = owner_label
            combined_rows.append(copied)
            support_rows.append(copied)
            linked_support_count += 1
            support_owner_counter[owner_label] += 1
            support_coarse_counter[str(row.get("observed_coarse_symbol", "")).strip()] += 1

    out_combined = Path(args.out_combined_csv)
    out_combined.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(combined_rows[0].keys()) if combined_rows else [
        "owner_label",
        "data_grounded_support_role",
        "source_owner_label",
    ]
    with out_combined.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in combined_rows:
            writer.writerow(row)

    out_support = Path(args.out_support_csv)
    support_fields = list(support_rows[0].keys()) if support_rows else fieldnames
    with out_support.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=support_fields)
        writer.writeheader()
        for row in support_rows:
            writer.writerow(row)

    summary_lines = [
        "WINDOW SECOND SUSTAIN DATA GROUNDED SUPPORT",
        "=" * 72,
        f"window_start_sec             : {args.window_start_sec:.6f}",
        f"window_end_sec               : {args.window_end_sec:.6f}",
        f"upper_direct_rows            : {upper_count}",
        f"linked_support_rows          : {linked_support_count}",
        f"combined_rows                : {len(combined_rows)}",
        "",
        "support_source_owner_counts:",
    ]
    for key, value in support_owner_counter.most_common():
        summary_lines.append(f"  {key}: {value}")
    summary_lines.extend(["", "support_coarse_counts:"])
    for key, value in support_coarse_counter.most_common(12):
        summary_lines.append(f"  {key}: {value}")
    summary_lines.extend(["", "frame_link_counts:"])
    for key, value in linked_frame_counter.most_common():
        summary_lines.append(f"  {key}: {value}")
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "stage": "window_second_sustain_data_grounded_support_builder",
                "inputs": {
                    "ownership_observations_csv": args.ownership_observations_csv,
                },
                "window": {
                    "start_sec": args.window_start_sec,
                    "end_sec": args.window_end_sec,
                },
                "result": {
                    "target_owner": target_owner,
                    "upper_direct_rows": upper_count,
                    "linked_support_rows": linked_support_count,
                    "combined_rows": len(combined_rows),
                    "support_source_owner_counts": dict(support_owner_counter),
                    "support_coarse_counts": dict(support_coarse_counter),
                    "frame_link_counts": dict(linked_frame_counter),
                },
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    _write_progress(
        args.progress_json,
        {
            "status": "done",
            "phase": "complete",
            "upper_direct_rows": upper_count,
            "linked_support_rows": linked_support_count,
            "combined_rows": len(combined_rows),
        },
    )


if __name__ == "__main__":
    main()
