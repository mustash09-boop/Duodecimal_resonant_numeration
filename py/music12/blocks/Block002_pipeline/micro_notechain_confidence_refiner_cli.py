# -*- coding: ascii -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
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


def _parse_json_list(value: str) -> list[Any]:
    try:
        loaded = json.loads(str(value or "").strip() or "[]")
        return loaded if isinstance(loaded, list) else []
    except Exception:
        return []


def _write_progress(path_str: str, payload: dict[str, Any]) -> None:
    if not str(path_str).strip():
        return
    Path(path_str).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _top_share_from_counter_json(value: str) -> float:
    items = _parse_json_list(value)
    if not items:
        return 0.0
    weights: list[int] = []
    for item in items:
        if isinstance(item, list) and len(item) >= 2:
            weights.append(_safe_int(item[1], 0))
    total = sum(weights)
    if total <= 0:
        return 0.0
    return max(weights) / total


def _temporal_regime(observation_frame_count: int) -> str:
    if observation_frame_count <= 3:
        return "SHORT_TRANSIENT_REGIME"
    if observation_frame_count <= 7:
        return "MEDIUM_LOCAL_REGIME"
    if observation_frame_count <= 15:
        return "LONG_SUSTAIN_REGIME"
    return "VERY_LONG_SUSTAIN_REGIME"


def _refine_score(row: dict[str, Any]) -> tuple[float, str, str, list[str]]:
    observation_frame_count = _safe_int(row.get("observation_frame_count"), 0)
    duration_frames = _safe_int(row.get("duration_frames"), 0)
    trajectory_count = _safe_int(row.get("trajectory_count"), 0)
    chain_kind = str(row.get("chain_structure_class", "")).strip()
    old_conf = str(row.get("confirmation_level", "")).strip()

    micro_share = _top_share_from_counter_json(row.get("top_micro_note_hypotheses_json", ""))
    coarse_share = _top_share_from_counter_json(row.get("top_coarse_note_hypotheses_json", ""))
    rank_share = _top_share_from_counter_json(row.get("dominant_coarse_ranks_json", ""))
    probe_share = _top_share_from_counter_json(row.get("dominant_probes_json", ""))

    score = 0.0
    reasons: list[str] = []

    score += min(observation_frame_count / 3.0, 6.0)
    if observation_frame_count >= 8:
        score += 1.2
        reasons.append("obs_ge8")
    if observation_frame_count >= 16:
        score += 1.4
        reasons.append("obs_ge16")
    if duration_frames >= 12:
        score += 0.7
        reasons.append("duration_ge12")
    if duration_frames >= 24:
        score += 0.8
        reasons.append("duration_ge24")

    if trajectory_count >= 2:
        score += 0.9
        reasons.append("multi_traj")
    if trajectory_count >= 3:
        score += 0.9
        reasons.append("traj_ge3")

    if chain_kind == "COHORT_DRIFT_BACKBONE_CHAIN":
        score += 2.2
        reasons.append("drift_backbone")
    elif chain_kind == "EXACT_PROBE_COHORT_BACKBONE_CHAIN":
        score += 1.8
        reasons.append("probe_backbone")
    elif chain_kind == "EXACT_MICRO_COHORT_CHAIN":
        score += 2.5
        reasons.append("exact_micro_chain")
    elif chain_kind == "LOCAL_COHORT_TRAJECTORY_CHAIN":
        score += 0.3
        reasons.append("local_chain")

    if old_conf == "CONFIRMED_CHAIN_V3":
        score += 1.2
        reasons.append("old_confirmed")
    elif old_conf == "PROBABLE_CHAIN_V3":
        score += 0.5
        reasons.append("old_probable")

    score += micro_share * 2.0
    score += coarse_share * 1.2
    score += rank_share * 1.5
    score += probe_share * 1.0
    if micro_share >= 0.75:
        reasons.append("stable_micro")
    if rank_share >= 0.70:
        reasons.append("stable_rank")
    if probe_share >= 0.70:
        reasons.append("stable_probe")

    regime = _temporal_regime(observation_frame_count)
    if score >= 10.0:
        label = "REFINED_CONFIRMED_BACKBONE"
    elif score >= 7.0:
        label = "REFINED_PROBABLE_BACKBONE"
    elif observation_frame_count >= 8:
        label = "REFINED_LONG_LOCAL_CHAIN"
    else:
        label = "REFINED_WEAK_LOCAL_CHAIN"
    return score, label, regime, reasons


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Refine confidence for micro notechains v3 without changing the chain graph itself."
    )
    ap.add_argument("--notechains-csv", required=True)
    ap.add_argument("--out-refined-csv", required=True)
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

    rows = _load_csv(Path(args.notechains_csv))
    total_rows = len(rows)

    _write_progress(
        args.progress_json,
        {
            "status": "running",
            "phase": "refining_confidence",
            "processed_rows": 0,
            "total_rows": total_rows,
        },
    )

    refined_rows: list[dict[str, Any]] = []
    refined_counter: Counter[str] = Counter()
    regime_counter: Counter[str] = Counter()
    score_values: list[float] = []

    for idx, row in enumerate(rows, start=1):
        score, refined_label, regime, reasons = _refine_score(row)
        refined_counter[refined_label] += 1
        regime_counter[regime] += 1
        score_values.append(score)
        new_row = dict(row)
        new_row["refined_confidence_score"] = f"{score:.6f}"
        new_row["refined_confidence_label"] = refined_label
        new_row["temporal_regime"] = regime
        new_row["refined_confidence_reasons_json"] = json.dumps(reasons, ensure_ascii=False)
        refined_rows.append(new_row)

        if idx % 4000 == 0 or idx == total_rows:
            _write_progress(
                args.progress_json,
                {
                    "status": "running",
                    "phase": "refining_confidence",
                    "processed_rows": idx,
                    "total_rows": total_rows,
                },
            )

    out_csv = Path(args.out_refined_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) + [
        "refined_confidence_score",
        "refined_confidence_label",
        "temporal_regime",
        "refined_confidence_reasons_json",
    ] if rows else []
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in refined_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    summary_lines = [
        "MICRO NOTECHAIN CONFIDENCE REFINER",
        "=" * 72,
        "source_mode               : MICRO_NOTECHAIN_V3_CONFIDENCE_REFINED",
        f"input_chain_rows          : {total_rows}",
        f"mean_refined_score        : {(sum(score_values) / len(score_values)) if score_values else 0.0:.6f}",
        f"max_refined_score         : {max(score_values) if score_values else 0.0:.6f}",
        "",
        "refined_confidence_counts:",
    ]
    for key, value in refined_counter.most_common():
        summary_lines.append(f"  {key}: {value}")
    summary_lines.extend(["", "temporal_regime_counts:"])
    for key, value in regime_counter.most_common():
        summary_lines.append(f"  {key}: {value}")
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "stage": "micro_notechain_confidence_refiner",
                "source_mode": "MICRO_NOTECHAIN_V3_CONFIDENCE_REFINED",
                "inputs": {
                    "notechains_csv": args.notechains_csv,
                },
                "result": {
                    "input_chain_rows": total_rows,
                    "refined_confidence_counts": dict(refined_counter),
                    "temporal_regime_counts": dict(regime_counter),
                    "mean_refined_score": (sum(score_values) / len(score_values)) if score_values else 0.0,
                    "max_refined_score": max(score_values) if score_values else 0.0,
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
