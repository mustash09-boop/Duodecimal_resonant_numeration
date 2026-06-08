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


def _resolve_role(row: dict[str, Any]) -> tuple[str, str, list[str]]:
    temporal_regime = str(row.get("temporal_regime", "")).strip()
    refined_label = str(row.get("refined_confidence_label", "")).strip()
    chain_kind = str(row.get("chain_structure_class", "")).strip()
    observation_frame_count = _safe_int(row.get("observation_frame_count"), 0)
    trajectory_count = _safe_int(row.get("trajectory_count"), 0)
    reasons: list[str] = []

    if refined_label == "REFINED_CONFIRMED_BACKBONE":
        if temporal_regime == "VERY_LONG_SUSTAIN_REGIME":
            role = "VERY_LONG_BACKBONE_ROLE"
            family = "BACKBONE_SUSTAIN"
            reasons.append("confirmed_very_long")
        elif temporal_regime == "LONG_SUSTAIN_REGIME":
            role = "LONG_BACKBONE_ROLE"
            family = "BACKBONE_SUSTAIN"
            reasons.append("confirmed_long")
        else:
            role = "BACKBONE_ROLE"
            family = "BACKBONE_LOCAL"
            reasons.append("confirmed_backbone")
    elif refined_label == "REFINED_PROBABLE_BACKBONE":
        if temporal_regime in {"LONG_SUSTAIN_REGIME", "VERY_LONG_SUSTAIN_REGIME"}:
            role = "PROBABLE_SUSTAIN_BACKBONE_ROLE"
            family = "BACKBONE_SUSTAIN"
            reasons.append("probable_sustain")
        else:
            role = "PROBABLE_BACKBONE_ROLE"
            family = "BACKBONE_LOCAL"
            reasons.append("probable_backbone")
    else:
        if temporal_regime == "SHORT_TRANSIENT_REGIME":
            role = "SHORT_LOCAL_TRANSIENT_ROLE"
            family = "LOCAL_TRANSIENT"
            reasons.append("short_local")
        elif temporal_regime == "MEDIUM_LOCAL_REGIME":
            role = "MEDIUM_LOCAL_EVENT_ROLE"
            family = "LOCAL_EVENT"
            reasons.append("medium_local")
        elif temporal_regime == "LONG_SUSTAIN_REGIME":
            role = "LONG_LOCAL_SUSTAIN_ROLE"
            family = "LOCAL_SUSTAIN"
            reasons.append("long_local")
        else:
            role = "VERY_LONG_LOCAL_SUSTAIN_ROLE"
            family = "LOCAL_SUSTAIN"
            reasons.append("very_long_local")

    if chain_kind == "COHORT_DRIFT_BACKBONE_CHAIN":
        reasons.append("drift_backbone_chain")
    elif chain_kind == "EXACT_PROBE_COHORT_BACKBONE_CHAIN":
        reasons.append("probe_backbone_chain")
    elif chain_kind == "EXACT_MICRO_COHORT_CHAIN":
        reasons.append("exact_micro_chain")
    elif chain_kind == "LOCAL_COHORT_TRAJECTORY_CHAIN":
        reasons.append("local_cohort_chain")

    if observation_frame_count >= 16:
        reasons.append("obs_ge16")
    elif observation_frame_count >= 8:
        reasons.append("obs_ge8")

    if trajectory_count >= 3:
        reasons.append("traj_ge3")
    elif trajectory_count >= 2:
        reasons.append("traj_ge2")

    return role, family, reasons


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Resolve scene roles such as backbone vs local transient from refined micro notechains before any instrument passport step."
    )
    ap.add_argument("--refined-chains-csv", required=True)
    ap.add_argument("--chain-frames-csv", required=True)
    ap.add_argument("--out-roles-csv", required=True)
    ap.add_argument("--out-frame-role-summary-csv", required=True)
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

    chain_rows = _load_csv(Path(args.refined_chains_csv))
    frame_rows = _load_csv(Path(args.chain_frames_csv))
    total_rows = len(chain_rows)

    _write_progress(
        args.progress_json,
        {
            "status": "running",
            "phase": "resolving_roles",
            "processed_rows": 0,
            "total_rows": total_rows,
        },
    )

    role_rows: list[dict[str, Any]] = []
    role_counter: Counter[str] = Counter()
    family_counter: Counter[str] = Counter()
    role_by_chain_id: dict[int, str] = {}
    family_by_chain_id: dict[int, str] = {}

    for idx, row in enumerate(chain_rows, start=1):
        role, family, reasons = _resolve_role(row)
        chain_id = _safe_int(row.get("chain_id"), 0)
        role_by_chain_id[chain_id] = role
        family_by_chain_id[chain_id] = family
        role_counter[role] += 1
        family_counter[family] += 1

        new_row = dict(row)
        new_row["resolved_role"] = role
        new_row["resolved_role_family"] = family
        new_row["resolved_role_reasons_json"] = json.dumps(reasons, ensure_ascii=False)
        role_rows.append(new_row)

        if idx % 4000 == 0 or idx == total_rows:
            _write_progress(
                args.progress_json,
                {
                    "status": "running",
                    "phase": "resolving_roles",
                    "processed_rows": idx,
                    "total_rows": total_rows,
                },
            )

    frame_role_counter: dict[int, Counter[str]] = defaultdict(Counter)
    frame_family_counter: dict[int, Counter[str]] = defaultdict(Counter)
    frame_time_sec: dict[int, str] = {}
    for row in frame_rows:
        frame_index = _safe_int(row.get("frame_index"), 0)
        chain_id = _safe_int(row.get("chain_id"), 0)
        role = role_by_chain_id.get(chain_id, "UNRESOLVED_ROLE")
        family = family_by_chain_id.get(chain_id, "UNRESOLVED_FAMILY")
        frame_role_counter[frame_index][role] += 1
        frame_family_counter[frame_index][family] += 1
        frame_time_sec[frame_index] = str(row.get("time_sec", ""))

    frame_role_rows: list[dict[str, Any]] = []
    for frame_index in sorted(frame_role_counter):
        role_counts = frame_role_counter[frame_index]
        family_counts = frame_family_counter[frame_index]
        dominant_role = role_counts.most_common(1)[0][0] if role_counts else ""
        dominant_family = family_counts.most_common(1)[0][0] if family_counts else ""
        frame_role_rows.append(
            {
                "frame_index": frame_index,
                "time_sec": frame_time_sec.get(frame_index, ""),
                "active_role_count": sum(role_counts.values()),
                "distinct_roles": len(role_counts),
                "dominant_role": dominant_role,
                "dominant_role_family": dominant_family,
                "role_counts_json": json.dumps(dict(role_counts), ensure_ascii=False),
                "family_counts_json": json.dumps(dict(family_counts), ensure_ascii=False),
            }
        )

    out_roles_csv = Path(args.out_roles_csv)
    out_roles_csv.parent.mkdir(parents=True, exist_ok=True)
    role_fields = list(chain_rows[0].keys()) + [
        "resolved_role",
        "resolved_role_family",
        "resolved_role_reasons_json",
    ] if chain_rows else []
    with out_roles_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=role_fields)
        writer.writeheader()
        for row in role_rows:
            writer.writerow({key: row.get(key, "") for key in role_fields})

    out_frame_csv = Path(args.out_frame_role_summary_csv)
    frame_fields = [
        "frame_index",
        "time_sec",
        "active_role_count",
        "distinct_roles",
        "dominant_role",
        "dominant_role_family",
        "role_counts_json",
        "family_counts_json",
    ]
    with out_frame_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=frame_fields)
        writer.writeheader()
        for row in frame_role_rows:
            writer.writerow({key: row.get(key, "") for key in frame_fields})

    summary_lines = [
        "BACKBONE VS LOCAL ROLE RESOLVER",
        "=" * 72,
        "source_mode               : MICRO_CHAIN_ROLE_LAYER",
        f"input_chain_rows          : {len(chain_rows)}",
        f"input_chain_frame_rows    : {len(frame_rows)}",
        f"frame_role_rows           : {len(frame_role_rows)}",
        "",
        "resolved_role_counts:",
    ]
    for key, value in role_counter.most_common():
        summary_lines.append(f"  {key}: {value}")
    summary_lines.extend(["", "resolved_role_family_counts:"])
    for key, value in family_counter.most_common():
        summary_lines.append(f"  {key}: {value}")
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "stage": "backbone_vs_local_role_resolver",
                "source_mode": "MICRO_CHAIN_ROLE_LAYER",
                "inputs": {
                    "refined_chains_csv": args.refined_chains_csv,
                    "chain_frames_csv": args.chain_frames_csv,
                },
                "result": {
                    "input_chain_rows": len(chain_rows),
                    "input_chain_frame_rows": len(frame_rows),
                    "frame_role_rows": len(frame_role_rows),
                    "resolved_role_counts": dict(role_counter),
                    "resolved_role_family_counts": dict(family_counter),
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
