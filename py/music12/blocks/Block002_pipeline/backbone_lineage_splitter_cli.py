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


def _split_lineage(row: dict[str, Any]) -> tuple[str, list[str]]:
    role = str(row.get("resolved_role", "")).strip()
    chain_kind = str(row.get("chain_structure_class", "")).strip()
    mean_freq = _safe_float(row.get("mean_frequency_hz"), 0.0)
    observation_frame_count = _safe_int(row.get("observation_frame_count"), 0)
    trajectory_count = _safe_int(row.get("trajectory_count"), 0)
    refined_score = _safe_float(row.get("refined_confidence_score"), 0.0)
    coarse_rank_div = _safe_int(row.get("coarse_rank_diversity"), 0)
    pitch_rank_div = _safe_int(row.get("pitchclass_rank_diversity"), 0)
    probe_div = _safe_int(row.get("probe_diversity"), 0)
    suffix_div = _safe_int(row.get("micro_suffix_diversity"), 0)

    reasons: list[str] = []

    if mean_freq >= 700.0 and observation_frame_count >= 12:
        lineage = "POSSIBLE_SECOND_SUSTAINED_LINEAGE"
        reasons.append("upper_register_sustain")
    elif role == "VERY_LONG_BACKBONE_ROLE" and chain_kind == "COHORT_DRIFT_BACKBONE_CHAIN" and trajectory_count >= 2:
        lineage = "POSSIBLE_SECOND_SUSTAINED_LINEAGE"
        reasons.append("very_long_drift_backbone")
    elif chain_kind in {"EXACT_PROBE_COHORT_BACKBONE_CHAIN", "EXACT_MICRO_COHORT_CHAIN"}:
        lineage = "MAIN_HARMONIC_BACKBONE"
        reasons.append("exact_backbone_chain")
    elif refined_score >= 11.0 and mean_freq < 700.0 and trajectory_count >= 1 and coarse_rank_div <= 2:
        lineage = "MAIN_HARMONIC_BACKBONE"
        reasons.append("stable_mid_backbone")
    elif role in {"LONG_BACKBONE_ROLE", "PROBABLE_SUSTAIN_BACKBONE_ROLE", "VERY_LONG_BACKBONE_ROLE"} and chain_kind == "LOCAL_COHORT_TRAJECTORY_CHAIN":
        lineage = "BODY_CONTINUATION_LINEAGE"
        reasons.append("local_sustain_body")
    elif observation_frame_count >= 16 and suffix_div >= 4 and pitch_rank_div >= 2:
        lineage = "POSSIBLE_SECOND_SUSTAINED_LINEAGE"
        reasons.append("diverse_upper_sustain")
    else:
        lineage = "BODY_CONTINUATION_LINEAGE"
        reasons.append("default_body")

    if observation_frame_count >= 16:
        reasons.append("obs_ge16")
    elif observation_frame_count >= 8:
        reasons.append("obs_ge8")
    if trajectory_count >= 2:
        reasons.append("traj_ge2")
    if probe_div >= 2:
        reasons.append("probe_diverse")
    return lineage, reasons


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Split long/very-long backbone layer into main harmonic backbone, body continuation, and possible second sustained lineage."
    )
    ap.add_argument("--roles-csv", required=True)
    ap.add_argument("--frame-roles-csv", required=True)
    ap.add_argument("--chain-frames-csv", required=True)
    ap.add_argument("--out-lineages-csv", required=True)
    ap.add_argument("--out-frame-lineage-summary-csv", required=True)
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

    role_rows = _load_csv(Path(args.roles_csv))
    frame_role_rows = _load_csv(Path(args.frame_roles_csv))
    chain_frame_rows = _load_csv(Path(args.chain_frames_csv))

    keep_roles = {"LONG_BACKBONE_ROLE", "VERY_LONG_BACKBONE_ROLE", "PROBABLE_SUSTAIN_BACKBONE_ROLE"}
    selected = [row for row in role_rows if str(row.get("resolved_role", "")).strip() in keep_roles]
    total_rows = len(selected)

    _write_progress(
        args.progress_json,
        {
            "status": "running",
            "phase": "splitting_lineages",
            "processed_rows": 0,
            "total_rows": total_rows,
        },
    )

    lineage_rows: list[dict[str, Any]] = []
    lineage_counter: Counter[str] = Counter()
    lineage_by_chain_id: dict[int, str] = {}

    for idx, row in enumerate(selected, start=1):
        chain_id = _safe_int(row.get("chain_id"), 0)
        lineage, reasons = _split_lineage(row)
        lineage_by_chain_id[chain_id] = lineage
        lineage_counter[lineage] += 1

        new_row = dict(row)
        new_row["backbone_lineage_class"] = lineage
        new_row["backbone_lineage_reasons_json"] = json.dumps(reasons, ensure_ascii=False)
        lineage_rows.append(new_row)

        if idx % 1000 == 0 or idx == total_rows:
            _write_progress(
                args.progress_json,
                {
                    "status": "running",
                    "phase": "splitting_lineages",
                    "processed_rows": idx,
                    "total_rows": total_rows,
                },
            )

    frame_counts: dict[int, Counter[str]] = defaultdict(Counter)
    frame_times: dict[int, str] = {}
    for row in chain_frame_rows:
        chain_id = _safe_int(row.get("chain_id"), 0)
        lineage = lineage_by_chain_id.get(chain_id, "")
        if not lineage:
            continue
        frame_index = _safe_int(row.get("frame_index"), 0)
        frame_counts[frame_index][lineage] += 1
        frame_times[frame_index] = str(row.get("time_sec", ""))

    frame_summary_rows: list[dict[str, Any]] = []
    for frame_role in frame_role_rows:
        frame_index = _safe_int(frame_role.get("frame_index"), 0)
        counts = frame_counts.get(frame_index, Counter())
        if not counts:
            continue
        dominant = counts.most_common(1)[0][0]
        frame_summary_rows.append(
            {
                "frame_index": frame_index,
                "time_sec": frame_times.get(frame_index, frame_role.get("time_sec", "")),
                "dominant_backbone_lineage": dominant,
                "active_backbone_lineage_count": sum(counts.values()),
                "distinct_backbone_lineages": len(counts),
                "backbone_lineage_counts_json": json.dumps(dict(counts), ensure_ascii=False),
                "dominant_role": frame_role.get("dominant_role", ""),
                "dominant_role_family": frame_role.get("dominant_role_family", ""),
            }
        )

    out_csv = Path(args.out_lineages_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    lineage_fields = list(selected[0].keys()) + [
        "backbone_lineage_class",
        "backbone_lineage_reasons_json",
    ] if selected else []
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=lineage_fields)
        writer.writeheader()
        for row in lineage_rows:
            writer.writerow({key: row.get(key, "") for key in lineage_fields})

    out_frame_csv = Path(args.out_frame_lineage_summary_csv)
    frame_fields = [
        "frame_index",
        "time_sec",
        "dominant_backbone_lineage",
        "active_backbone_lineage_count",
        "distinct_backbone_lineages",
        "backbone_lineage_counts_json",
        "dominant_role",
        "dominant_role_family",
    ]
    with out_frame_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=frame_fields)
        writer.writeheader()
        for row in frame_summary_rows:
            writer.writerow({key: row.get(key, "") for key in frame_fields})

    summary_lines = [
        "BACKBONE LINEAGE SPLITTER",
        "=" * 72,
        "source_mode               : BACKBONE_LAYER_INTERNAL_SPLIT",
        f"input_backbone_rows       : {len(selected)}",
        f"input_chain_frame_rows    : {len(chain_frame_rows)}",
        f"frame_lineage_rows        : {len(frame_summary_rows)}",
        "",
        "backbone_lineage_counts:",
    ]
    for key, value in lineage_counter.most_common():
        summary_lines.append(f"  {key}: {value}")
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "stage": "backbone_lineage_splitter",
                "source_mode": "BACKBONE_LAYER_INTERNAL_SPLIT",
                "inputs": {
                    "roles_csv": args.roles_csv,
                    "frame_roles_csv": args.frame_roles_csv,
                    "chain_frames_csv": args.chain_frames_csv,
                },
                "result": {
                    "input_backbone_rows": len(selected),
                    "input_chain_frame_rows": len(chain_frame_rows),
                    "frame_lineage_rows": len(frame_summary_rows),
                    "backbone_lineage_counts": dict(lineage_counter),
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
