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


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    n = len(values)
    mid = n // 2
    if n % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def _profile_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    freq = [_safe_float(r.get("mean_frequency_hz"), 0.0) for r in rows]
    obs = [_safe_int(r.get("observation_frame_count"), 0) for r in rows]
    dur = [_safe_int(r.get("duration_frames"), 0) for r in rows]
    traj = [_safe_int(r.get("trajectory_count"), 0) for r in rows]
    coarse = Counter(str(r.get("anchor_coarse_note", "")).strip() for r in rows)
    kind = Counter(str(r.get("chain_structure_class", "")).strip() for r in rows)
    role = Counter(str(r.get("resolved_role", "")).strip() for r in rows)
    return {
        "count": len(rows),
        "mean_frequency_hz": _mean(freq),
        "median_frequency_hz": _median(freq),
        "mean_observation_frames": _mean([float(x) for x in obs]),
        "mean_duration_frames": _mean([float(x) for x in dur]),
        "mean_trajectory_count": _mean([float(x) for x in traj]),
        "top_coarse_tokens": coarse.most_common(20),
        "chain_structure_counts": dict(kind),
        "resolved_role_counts": dict(role),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare main harmonic backbone against possible second sustained lineage by register, duration, coarse tokens and frame overlap."
    )
    ap.add_argument("--backbone-lineages-csv", required=True)
    ap.add_argument("--backbone-frame-lineages-csv", required=True)
    ap.add_argument("--out-compare-csv", required=True)
    ap.add_argument("--out-frame-overlap-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--progress-json", default="")
    args = ap.parse_args()

    _write_progress(
        args.progress_json,
        {
            "status": "running",
            "phase": "loading_inputs",
            "processed_rows": 0,
            "total_rows": 0,
        },
    )

    rows = _load_csv(Path(args.backbone_lineages_csv))
    frame_rows = _load_csv(Path(args.backbone_frame_lineages_csv))
    total_rows = len(rows)

    main_rows = [r for r in rows if str(r.get("backbone_lineage_class", "")).strip() == "MAIN_HARMONIC_BACKBONE"]
    second_rows = [r for r in rows if str(r.get("backbone_lineage_class", "")).strip() == "POSSIBLE_SECOND_SUSTAINED_LINEAGE"]

    _write_progress(
        args.progress_json,
        {
            "status": "running",
            "phase": "computing_compare",
            "processed_rows": total_rows,
            "total_rows": total_rows,
        },
    )

    main_profile = _profile_rows(main_rows)
    second_profile = _profile_rows(second_rows)

    compare_rows = [
        {
            "lineage_class": "MAIN_HARMONIC_BACKBONE",
            "count": main_profile["count"],
            "mean_frequency_hz": f"{main_profile['mean_frequency_hz']:.6f}",
            "median_frequency_hz": f"{main_profile['median_frequency_hz']:.6f}",
            "mean_observation_frames": f"{main_profile['mean_observation_frames']:.6f}",
            "mean_duration_frames": f"{main_profile['mean_duration_frames']:.6f}",
            "mean_trajectory_count": f"{main_profile['mean_trajectory_count']:.6f}",
            "top_coarse_tokens_json": json.dumps(main_profile["top_coarse_tokens"], ensure_ascii=False),
            "chain_structure_counts_json": json.dumps(main_profile["chain_structure_counts"], ensure_ascii=False),
            "resolved_role_counts_json": json.dumps(main_profile["resolved_role_counts"], ensure_ascii=False),
        },
        {
            "lineage_class": "POSSIBLE_SECOND_SUSTAINED_LINEAGE",
            "count": second_profile["count"],
            "mean_frequency_hz": f"{second_profile['mean_frequency_hz']:.6f}",
            "median_frequency_hz": f"{second_profile['median_frequency_hz']:.6f}",
            "mean_observation_frames": f"{second_profile['mean_observation_frames']:.6f}",
            "mean_duration_frames": f"{second_profile['mean_duration_frames']:.6f}",
            "mean_trajectory_count": f"{second_profile['mean_trajectory_count']:.6f}",
            "top_coarse_tokens_json": json.dumps(second_profile["top_coarse_tokens"], ensure_ascii=False),
            "chain_structure_counts_json": json.dumps(second_profile["chain_structure_counts"], ensure_ascii=False),
            "resolved_role_counts_json": json.dumps(second_profile["resolved_role_counts"], ensure_ascii=False),
        },
    ]

    overlap_rows: list[dict[str, Any]] = []
    for row in frame_rows:
        counts = json.loads(str(row.get("backbone_lineage_counts_json", "")).strip() or "{}")
        if not isinstance(counts, dict):
            continue
        main_count = _safe_int(counts.get("MAIN_HARMONIC_BACKBONE"), 0)
        second_count = _safe_int(counts.get("POSSIBLE_SECOND_SUSTAINED_LINEAGE"), 0)
        body_count = _safe_int(counts.get("BODY_CONTINUATION_LINEAGE"), 0)
        if main_count <= 0 and second_count <= 0:
            continue
        overlap_rows.append(
            {
                "frame_index": _safe_int(row.get("frame_index"), 0),
                "time_sec": row.get("time_sec", ""),
                "main_backbone_activity": main_count,
                "second_sustain_activity": second_count,
                "body_continuation_activity": body_count,
                "dominant_backbone_lineage": row.get("dominant_backbone_lineage", ""),
                "coexistence_mode": (
                    "MAIN_AND_SECOND_OVERLAP" if main_count > 0 and second_count > 0
                    else "MAIN_ONLY" if main_count > 0
                    else "SECOND_ONLY"
                ),
            }
        )

    coexistence_counter = Counter(str(r["coexistence_mode"]) for r in overlap_rows)

    out_compare = Path(args.out_compare_csv)
    out_compare.parent.mkdir(parents=True, exist_ok=True)
    compare_fields = [
        "lineage_class",
        "count",
        "mean_frequency_hz",
        "median_frequency_hz",
        "mean_observation_frames",
        "mean_duration_frames",
        "mean_trajectory_count",
        "top_coarse_tokens_json",
        "chain_structure_counts_json",
        "resolved_role_counts_json",
    ]
    with out_compare.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=compare_fields)
        writer.writeheader()
        for row in compare_rows:
            writer.writerow(row)

    out_overlap = Path(args.out_frame_overlap_csv)
    overlap_fields = [
        "frame_index",
        "time_sec",
        "main_backbone_activity",
        "second_sustain_activity",
        "body_continuation_activity",
        "dominant_backbone_lineage",
        "coexistence_mode",
    ]
    with out_overlap.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=overlap_fields)
        writer.writeheader()
        for row in overlap_rows:
            writer.writerow(row)

    summary_lines = [
        "MAIN BACKBONE VS SECOND SUSTAIN COMPARE",
        "=" * 72,
        f"main_backbone_count        : {main_profile['count']}",
        f"second_sustain_count       : {second_profile['count']}",
        f"main_mean_freq_hz          : {main_profile['mean_frequency_hz']:.6f}",
        f"second_mean_freq_hz        : {second_profile['mean_frequency_hz']:.6f}",
        f"main_mean_obs_frames       : {main_profile['mean_observation_frames']:.6f}",
        f"second_mean_obs_frames     : {second_profile['mean_observation_frames']:.6f}",
        f"main_mean_duration_frames  : {main_profile['mean_duration_frames']:.6f}",
        f"second_mean_duration_frames: {second_profile['mean_duration_frames']:.6f}",
        "",
        "coexistence_mode_counts:",
    ]
    for key, value in coexistence_counter.most_common():
        summary_lines.append(f"  {key}: {value}")
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "stage": "main_backbone_vs_second_sustain_compare",
                "inputs": {
                    "backbone_lineages_csv": args.backbone_lineages_csv,
                    "backbone_frame_lineages_csv": args.backbone_frame_lineages_csv,
                },
                "result": {
                    "main_backbone_profile": main_profile,
                    "second_sustain_profile": second_profile,
                    "coexistence_mode_counts": dict(coexistence_counter),
                    "frame_overlap_rows": len(overlap_rows),
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
            "processed_rows": total_rows,
            "total_rows": total_rows,
        },
    )


if __name__ == "__main__":
    main()
