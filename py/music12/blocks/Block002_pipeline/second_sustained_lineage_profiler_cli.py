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


def _register_band(mean_freq: float) -> str:
    if mean_freq < 400.0:
        return "LOW_REGISTER_BAND"
    if mean_freq < 800.0:
        return "MID_REGISTER_BAND"
    if mean_freq < 1200.0:
        return "UPPER_REGISTER_BAND"
    return "HIGH_REGISTER_BAND"


def _profile_class(row: dict[str, Any]) -> tuple[str, list[str]]:
    mean_freq = _safe_float(row.get("mean_frequency_hz"), 0.0)
    observation_frame_count = _safe_int(row.get("observation_frame_count"), 0)
    duration_frames = _safe_int(row.get("duration_frames"), 0)
    trajectory_count = _safe_int(row.get("trajectory_count"), 0)
    chain_kind = str(row.get("chain_structure_class", "")).strip()
    coarse_rank_div = _safe_int(row.get("coarse_rank_diversity"), 0)
    probe_div = _safe_int(row.get("probe_diversity"), 0)
    reasons: list[str] = []

    if mean_freq >= 1100.0 and observation_frame_count >= 16:
        profile = "HIGH_UPPER_SUSTAIN_PROFILE"
        reasons.append("high_upper_sustain")
    elif mean_freq >= 800.0 and observation_frame_count >= 12:
        profile = "UPPER_SUSTAIN_PROFILE"
        reasons.append("upper_sustain")
    elif chain_kind == "COHORT_DRIFT_BACKBONE_CHAIN" and trajectory_count >= 2:
        profile = "DRIFTED_SUSTAIN_PROFILE"
        reasons.append("drifted_sustain")
    else:
        profile = "LOCALIZED_SUSTAIN_PROFILE"
        reasons.append("localized_sustain")

    if duration_frames >= 24:
        reasons.append("duration_ge24")
    elif duration_frames >= 12:
        reasons.append("duration_ge12")
    if coarse_rank_div >= 3:
        reasons.append("rank_diverse")
    if probe_div >= 2:
        reasons.append("probe_diverse")
    return profile, reasons


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Profile the possible second sustained lineage layer by register, duration and internal sustain morphology."
    )
    ap.add_argument("--backbone-lineages-csv", required=True)
    ap.add_argument("--backbone-frame-lineages-csv", required=True)
    ap.add_argument("--out-profile-csv", required=True)
    ap.add_argument("--out-frame-profile-summary-csv", required=True)
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

    lineage_rows = _load_csv(Path(args.backbone_lineages_csv))
    frame_rows = _load_csv(Path(args.backbone_frame_lineages_csv))
    selected = [row for row in lineage_rows if str(row.get("backbone_lineage_class", "")).strip() == "POSSIBLE_SECOND_SUSTAINED_LINEAGE"]
    total_rows = len(selected)

    _write_progress(
        args.progress_json,
        {
            "status": "running",
            "phase": "profiling_second_sustain",
            "processed_rows": 0,
            "total_rows": total_rows,
        },
    )

    profile_rows: list[dict[str, Any]] = []
    profile_counter: Counter[str] = Counter()
    band_counter: Counter[str] = Counter()
    profile_by_chain_id: dict[int, str] = {}
    band_by_chain_id: dict[int, str] = {}

    for idx, row in enumerate(selected, start=1):
        mean_freq = _safe_float(row.get("mean_frequency_hz"), 0.0)
        band = _register_band(mean_freq)
        profile, reasons = _profile_class(row)
        chain_id = _safe_int(row.get("chain_id"), 0)
        profile_by_chain_id[chain_id] = profile
        band_by_chain_id[chain_id] = band
        profile_counter[profile] += 1
        band_counter[band] += 1

        new_row = dict(row)
        new_row["second_sustain_profile_class"] = profile
        new_row["register_band"] = band
        new_row["second_sustain_profile_reasons_json"] = json.dumps(reasons, ensure_ascii=False)
        profile_rows.append(new_row)

        if idx % 500 == 0 or idx == total_rows:
            _write_progress(
                args.progress_json,
                {
                    "status": "running",
                    "phase": "profiling_second_sustain",
                    "processed_rows": idx,
                    "total_rows": total_rows,
                },
            )

    frame_profile_rows: list[dict[str, Any]] = []
    for row in frame_rows:
        lineage_counts = json.loads(str(row.get("backbone_lineage_counts_json", "")).strip() or "{}")
        if not isinstance(lineage_counts, dict):
            continue
        if "POSSIBLE_SECOND_SUSTAINED_LINEAGE" not in lineage_counts:
            continue
        frame_profile_rows.append(
            {
                "frame_index": _safe_int(row.get("frame_index"), 0),
                "time_sec": row.get("time_sec", ""),
                "dominant_backbone_lineage": row.get("dominant_backbone_lineage", ""),
                "possible_second_sustain_activity": _safe_int(lineage_counts.get("POSSIBLE_SECOND_SUSTAINED_LINEAGE"), 0),
                "backbone_lineage_counts_json": row.get("backbone_lineage_counts_json", ""),
            }
        )

    out_csv = Path(args.out_profile_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    profile_fields = list(selected[0].keys()) + [
        "second_sustain_profile_class",
        "register_band",
        "second_sustain_profile_reasons_json",
    ] if selected else []
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=profile_fields)
        writer.writeheader()
        for row in profile_rows:
            writer.writerow({key: row.get(key, "") for key in profile_fields})

    out_frame_csv = Path(args.out_frame_profile_summary_csv)
    frame_fields = [
        "frame_index",
        "time_sec",
        "dominant_backbone_lineage",
        "possible_second_sustain_activity",
        "backbone_lineage_counts_json",
    ]
    with out_frame_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=frame_fields)
        writer.writeheader()
        for row in frame_profile_rows:
            writer.writerow({key: row.get(key, "") for key in frame_fields})

    summary_lines = [
        "SECOND SUSTAINED LINEAGE PROFILER",
        "=" * 72,
        "source_mode               : POSSIBLE_SECOND_SUSTAINED_PROFILE",
        f"input_second_lineage_rows : {len(selected)}",
        f"frame_profile_rows        : {len(frame_profile_rows)}",
        "",
        "profile_class_counts:",
    ]
    for key, value in profile_counter.most_common():
        summary_lines.append(f"  {key}: {value}")
    summary_lines.extend(["", "register_band_counts:"])
    for key, value in band_counter.most_common():
        summary_lines.append(f"  {key}: {value}")
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "stage": "second_sustained_lineage_profiler",
                "source_mode": "POSSIBLE_SECOND_SUSTAINED_PROFILE",
                "inputs": {
                    "backbone_lineages_csv": args.backbone_lineages_csv,
                    "backbone_frame_lineages_csv": args.backbone_frame_lineages_csv,
                },
                "result": {
                    "input_second_lineage_rows": len(selected),
                    "frame_profile_rows": len(frame_profile_rows),
                    "profile_class_counts": dict(profile_counter),
                    "register_band_counts": dict(band_counter),
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
