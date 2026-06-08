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


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _owner_from_role_and_lineage(
    resolved_role: str,
    backbone_lineage_class: str,
) -> tuple[str, str, list[str]]:
    reasons: list[str] = []
    if backbone_lineage_class == "MAIN_HARMONIC_BACKBONE":
        reasons.append("main_backbone_lineage")
        return "MAIN_BACKBONE_OWNER", "PIANOISH_OWNER", reasons
    if backbone_lineage_class == "POSSIBLE_SECOND_SUSTAINED_LINEAGE":
        reasons.append("possible_second_sustained_lineage")
        return "SECOND_SUSTAIN_OWNER", "SECOND_LAYER_OWNER", reasons
    if backbone_lineage_class == "BODY_CONTINUATION_LINEAGE":
        reasons.append("body_continuation_lineage")
        return "BODY_CONTINUATION_OWNER", "PIANOISH_OWNER", reasons

    if resolved_role == "SHORT_LOCAL_TRANSIENT_ROLE":
        reasons.append("short_local_transient_role")
        return "LOCAL_TRANSIENT_OWNER", "PIANOISH_OWNER", reasons
    if resolved_role == "MEDIUM_LOCAL_EVENT_ROLE":
        reasons.append("medium_local_event_role")
        return "LOCAL_EVENT_OWNER", "PIANOISH_OWNER", reasons
    if resolved_role in {
        "LONG_BACKBONE_ROLE",
        "VERY_LONG_BACKBONE_ROLE",
        "PROBABLE_SUSTAIN_BACKBONE_ROLE",
        "PROBABLE_BACKBONE_ROLE",
        "BACKBONE_ROLE",
    }:
        reasons.append("backbone_role_without_specific_lineage")
        return "UNRESOLVED_BACKBONE_OWNER", "UNRESOLVED_BACKBONE", reasons

    reasons.append("no_lineage_no_role_match")
    return "UNRESOLVED_OWNER", "UNRESOLVED", reasons


def _collision_kind(owner_labels: set[str], owner_families: set[str]) -> str:
    if not owner_labels:
        return "EMPTY"
    if owner_labels == {"SECOND_SUSTAIN_OWNER"}:
        return "SECOND_ONLY"
    if "SECOND_SUSTAIN_OWNER" in owner_labels and "PIANOISH_OWNER" in owner_families:
        return "SECOND_WITH_PIANOISH"
    if "SECOND_SUSTAIN_OWNER" in owner_labels:
        return "SECOND_WITH_OTHER"
    if owner_labels == {"MAIN_BACKBONE_OWNER"}:
        return "MAIN_ONLY"
    if owner_labels == {"BODY_CONTINUATION_OWNER"}:
        return "BODY_ONLY"
    if owner_labels <= {"LOCAL_TRANSIENT_OWNER", "LOCAL_EVENT_OWNER"}:
        return "LOCAL_ONLY"
    if owner_labels <= {"UNRESOLVED_BACKBONE_OWNER"}:
        return "BACKBONE_ONLY_UNRESOLVED"
    if len(owner_labels) > 1:
        return "MIXED_NON_SECOND"
    return "OTHER_SINGLE_OWNER"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build probe-domain ownership for a local window by joining micro notechain frame "
            "observations with role and backbone-lineage decisions."
        )
    )
    ap.add_argument("--notechain-frames-csv", required=True)
    ap.add_argument("--roles-csv", required=True)
    ap.add_argument("--backbone-lineages-csv", required=True)
    ap.add_argument("--window-start-sec", type=float, required=True)
    ap.add_argument("--window-end-sec", type=float, required=True)
    ap.add_argument("--out-observations-csv", required=True)
    ap.add_argument("--out-probe-cells-csv", required=True)
    ap.add_argument("--out-frame-summary-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--progress-json", default="")
    args = ap.parse_args()

    start_frame = int(args.window_start_sec * 60.0)
    end_frame = int(args.window_end_sec * 60.0 + 0.999999)

    _write_progress(
        args.progress_json,
        {
            "status": "running",
            "phase": "loading_inputs",
            "window_start_sec": args.window_start_sec,
            "window_end_sec": args.window_end_sec,
            "window_start_frame": start_frame,
            "window_end_frame": end_frame,
        },
    )

    role_rows = _load_csv(Path(args.roles_csv))
    lineage_rows = _load_csv(Path(args.backbone_lineages_csv))
    frame_rows = _load_csv(Path(args.notechain_frames_csv))

    role_map: dict[int, dict[str, Any]] = {}
    for row in role_rows:
        role_map[_safe_int(row.get("chain_id"), 0)] = row

    lineage_map: dict[int, dict[str, Any]] = {}
    for row in lineage_rows:
        lineage_map[_safe_int(row.get("chain_id"), 0)] = row

    selected_frames = [
        row for row in frame_rows
        if start_frame <= _safe_int(row.get("frame_index"), 0) <= end_frame
    ]

    _write_progress(
        args.progress_json,
        {
            "status": "running",
            "phase": "building_observations",
            "selected_frame_rows": len(selected_frames),
        },
    )

    observation_rows: list[dict[str, Any]] = []
    cell_buckets: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    owner_counter: Counter[str] = Counter()
    owner_family_counter: Counter[str] = Counter()

    for row in selected_frames:
        chain_id = _safe_int(row.get("chain_id"), 0)
        role_info = role_map.get(chain_id, {})
        lineage_info = lineage_map.get(chain_id, {})
        resolved_role = str(role_info.get("resolved_role", "")).strip()
        resolved_role_family = str(role_info.get("resolved_role_family", "")).strip()
        backbone_lineage_class = str(lineage_info.get("backbone_lineage_class", "")).strip()
        owner_label, owner_family, owner_reasons = _owner_from_role_and_lineage(
            resolved_role=resolved_role,
            backbone_lineage_class=backbone_lineage_class,
        )
        obs = {
            "chain_id": chain_id,
            "frame_index": _safe_int(row.get("frame_index"), 0),
            "time_sec": row.get("time_sec", ""),
            "probe_index": _safe_int(row.get("probe_index"), 0),
            "trajectory_id": _safe_int(row.get("trajectory_id"), 0),
            "slot_index": _safe_int(row.get("slot_index"), 0),
            "observed_micro_symbol": row.get("observed_micro_symbol", ""),
            "observed_coarse_symbol": row.get("observed_coarse_symbol", ""),
            "micro_suffix": row.get("micro_suffix", ""),
            "frequency_hz": row.get("frequency_hz", ""),
            "energy": row.get("energy", ""),
            "rise": row.get("rise", ""),
            "continuation": row.get("continuation", ""),
            "coarse_group_rank": row.get("coarse_group_rank", ""),
            "coarse_group_size": row.get("coarse_group_size", ""),
            "pitchclass_group_rank": row.get("pitchclass_group_rank", ""),
            "pitchclass_group_size": row.get("pitchclass_group_size", ""),
            "observation_kind": row.get("observation_kind", ""),
            "resolved_role": resolved_role,
            "resolved_role_family": resolved_role_family,
            "backbone_lineage_class": backbone_lineage_class,
            "owner_label": owner_label,
            "owner_family": owner_family,
            "owner_reasons_json": _json_dumps(owner_reasons),
        }
        observation_rows.append(obs)
        cell_buckets[(obs["frame_index"], obs["probe_index"])].append(obs)
        owner_counter[owner_label] += 1
        owner_family_counter[owner_family] += 1

    observation_fields = [
        "chain_id",
        "frame_index",
        "time_sec",
        "probe_index",
        "trajectory_id",
        "slot_index",
        "observed_micro_symbol",
        "observed_coarse_symbol",
        "micro_suffix",
        "frequency_hz",
        "energy",
        "rise",
        "continuation",
        "coarse_group_rank",
        "coarse_group_size",
        "pitchclass_group_rank",
        "pitchclass_group_size",
        "observation_kind",
        "resolved_role",
        "resolved_role_family",
        "backbone_lineage_class",
        "owner_label",
        "owner_family",
        "owner_reasons_json",
    ]
    out_obs = Path(args.out_observations_csv)
    out_obs.parent.mkdir(parents=True, exist_ok=True)
    with out_obs.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=observation_fields)
        writer.writeheader()
        for row in observation_rows:
            writer.writerow(row)

    _write_progress(
        args.progress_json,
        {
            "status": "running",
            "phase": "aggregating_probe_cells",
            "observation_rows": len(observation_rows),
            "probe_cells": len(cell_buckets),
        },
    )

    probe_cell_rows: list[dict[str, Any]] = []
    frame_summary_buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    collision_counter: Counter[str] = Counter()
    second_contamination_counter: Counter[str] = Counter()

    for (frame_index, probe_index), rows in sorted(cell_buckets.items()):
        owner_labels = {str(r["owner_label"]) for r in rows}
        owner_families = {str(r["owner_family"]) for r in rows}
        energy_by_owner: dict[str, float] = defaultdict(float)
        coarse_counter: Counter[str] = Counter()
        micro_counter: Counter[str] = Counter()
        for row in rows:
            energy_by_owner[str(row["owner_label"])] += _safe_float(row.get("energy"), 0.0)
            coarse_counter[str(row.get("observed_coarse_symbol", "")).strip()] += 1
            micro_counter[str(row.get("observed_micro_symbol", "")).strip()] += 1
        total_energy = sum(energy_by_owner.values())
        dominant_owner = max(
            energy_by_owner.items(),
            key=lambda item: item[1],
        )[0] if energy_by_owner else "UNRESOLVED_OWNER"
        dominant_energy_share = (energy_by_owner.get(dominant_owner, 0.0) / total_energy) if total_energy > 0.0 else 0.0
        collision_kind = _collision_kind(owner_labels=owner_labels, owner_families=owner_families)
        collision_counter[collision_kind] += 1
        if "SECOND_SUSTAIN_OWNER" in owner_labels:
            second_contamination_counter[collision_kind] += 1
        time_sec = _safe_float(rows[0].get("time_sec"), 0.0)
        mean_frequency_hz = sum(_safe_float(r.get("frequency_hz"), 0.0) for r in rows) / len(rows)
        probe_cell = {
            "frame_index": frame_index,
            "time_sec": f"{time_sec:.9f}",
            "probe_index": probe_index,
            "observation_count": len(rows),
            "owner_count": len(owner_labels),
            "dominant_owner": dominant_owner,
            "dominant_energy_share": f"{dominant_energy_share:.6f}",
            "total_energy": f"{total_energy:.9f}",
            "mean_frequency_hz": f"{mean_frequency_hz:.9f}",
            "collision_kind": collision_kind,
            "has_main_backbone_owner": int("MAIN_BACKBONE_OWNER" in owner_labels),
            "has_second_sustain_owner": int("SECOND_SUSTAIN_OWNER" in owner_labels),
            "has_body_continuation_owner": int("BODY_CONTINUATION_OWNER" in owner_labels),
            "has_local_transient_owner": int("LOCAL_TRANSIENT_OWNER" in owner_labels),
            "has_local_event_owner": int("LOCAL_EVENT_OWNER" in owner_labels),
            "has_unresolved_backbone_owner": int("UNRESOLVED_BACKBONE_OWNER" in owner_labels),
            "owner_labels_json": _json_dumps(sorted(owner_labels)),
            "owner_families_json": _json_dumps(sorted(owner_families)),
            "owner_energy_json": _json_dumps({k: round(v, 9) for k, v in sorted(energy_by_owner.items())}),
            "top_coarse_symbols_json": _json_dumps(coarse_counter.most_common(8)),
            "top_micro_symbols_json": _json_dumps(micro_counter.most_common(8)),
        }
        probe_cell_rows.append(probe_cell)
        frame_summary_buckets[frame_index].append(probe_cell)

    out_cells = Path(args.out_probe_cells_csv)
    cell_fields = [
        "frame_index",
        "time_sec",
        "probe_index",
        "observation_count",
        "owner_count",
        "dominant_owner",
        "dominant_energy_share",
        "total_energy",
        "mean_frequency_hz",
        "collision_kind",
        "has_main_backbone_owner",
        "has_second_sustain_owner",
        "has_body_continuation_owner",
        "has_local_transient_owner",
        "has_local_event_owner",
        "has_unresolved_backbone_owner",
        "owner_labels_json",
        "owner_families_json",
        "owner_energy_json",
        "top_coarse_symbols_json",
        "top_micro_symbols_json",
    ]
    with out_cells.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cell_fields)
        writer.writeheader()
        for row in probe_cell_rows:
            writer.writerow(row)

    frame_summary_rows: list[dict[str, Any]] = []
    frame_coexistence_counter: Counter[str] = Counter()
    for frame_index, rows in sorted(frame_summary_buckets.items()):
        probe_cells = len(rows)
        second_only = sum(1 for r in rows if str(r["collision_kind"]) == "SECOND_ONLY")
        second_with_pianoish = sum(1 for r in rows if str(r["collision_kind"]) == "SECOND_WITH_PIANOISH")
        main_only = sum(1 for r in rows if str(r["collision_kind"]) == "MAIN_ONLY")
        body_only = sum(1 for r in rows if str(r["collision_kind"]) == "BODY_ONLY")
        local_only = sum(1 for r in rows if str(r["collision_kind"]) == "LOCAL_ONLY")
        mixed_non_second = sum(1 for r in rows if str(r["collision_kind"]) == "MIXED_NON_SECOND")
        dominant_counter = Counter(str(r["dominant_owner"]) for r in rows)
        has_second_frame = any(_safe_int(r["has_second_sustain_owner"], 0) > 0 for r in rows)
        has_pianoish_frame = any("PIANOISH_OWNER" in json.loads(str(r["owner_families_json"])) for r in rows)
        has_unresolved_backbone_frame = any(_safe_int(r["has_unresolved_backbone_owner"], 0) > 0 for r in rows)
        if has_second_frame and has_pianoish_frame:
            frame_coexistence_kind = "SECOND_WITH_PIANOISH_FRAME"
        elif has_second_frame and not has_pianoish_frame and not has_unresolved_backbone_frame:
            frame_coexistence_kind = "SECOND_ONLY_FRAME"
        elif has_second_frame:
            frame_coexistence_kind = "SECOND_WITH_OTHER_FRAME"
        elif has_pianoish_frame:
            frame_coexistence_kind = "PIANOISH_ONLY_FRAME"
        else:
            frame_coexistence_kind = "OTHER_FRAME"
        frame_coexistence_counter[frame_coexistence_kind] += 1
        frame_summary_rows.append(
            {
                "frame_index": frame_index,
                "time_sec": rows[0]["time_sec"],
                "probe_cell_count": probe_cells,
                "second_only_cells": second_only,
                "second_with_pianoish_cells": second_with_pianoish,
                "main_only_cells": main_only,
                "body_only_cells": body_only,
                "local_only_cells": local_only,
                "mixed_non_second_cells": mixed_non_second,
                "has_second_sustain_frame": int(has_second_frame),
                "has_pianoish_frame": int(has_pianoish_frame),
                "has_unresolved_backbone_frame": int(has_unresolved_backbone_frame),
                "frame_coexistence_kind": frame_coexistence_kind,
                "dominant_owner_counts_json": _json_dumps(dict(dominant_counter)),
            }
        )

    out_frame_summary = Path(args.out_frame_summary_csv)
    frame_fields = [
        "frame_index",
        "time_sec",
        "probe_cell_count",
        "second_only_cells",
        "second_with_pianoish_cells",
        "main_only_cells",
        "body_only_cells",
        "local_only_cells",
        "mixed_non_second_cells",
        "has_second_sustain_frame",
        "has_pianoish_frame",
        "has_unresolved_backbone_frame",
        "frame_coexistence_kind",
        "dominant_owner_counts_json",
    ]
    with out_frame_summary.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=frame_fields)
        writer.writeheader()
        for row in frame_summary_rows:
            writer.writerow(row)

    second_probe_cells = sum(1 for row in probe_cell_rows if _safe_int(row["has_second_sustain_owner"], 0) > 0)
    second_only_cells = collision_counter.get("SECOND_ONLY", 0)
    second_with_pianoish_cells = collision_counter.get("SECOND_WITH_PIANOISH", 0)
    second_with_other_cells = collision_counter.get("SECOND_WITH_OTHER", 0)
    second_clean_ratio = (float(second_only_cells) / float(second_probe_cells)) if second_probe_cells else 0.0
    second_pianoish_collision_ratio = (
        float(second_with_pianoish_cells) / float(second_probe_cells)
    ) if second_probe_cells else 0.0
    second_frame_count = sum(1 for row in frame_summary_rows if _safe_int(row["has_second_sustain_frame"], 0) > 0)
    second_with_pianoish_frames = frame_coexistence_counter.get("SECOND_WITH_PIANOISH_FRAME", 0)
    second_only_frames = frame_coexistence_counter.get("SECOND_ONLY_FRAME", 0)
    second_frame_pianoish_ratio = (
        float(second_with_pianoish_frames) / float(second_frame_count)
    ) if second_frame_count else 0.0

    summary_lines = [
        "WINDOW PROBE OWNERSHIP",
        "=" * 72,
        f"window_start_sec                 : {args.window_start_sec:.6f}",
        f"window_end_sec                   : {args.window_end_sec:.6f}",
        f"window_start_frame               : {start_frame}",
        f"window_end_frame                 : {end_frame}",
        f"observation_rows                 : {len(observation_rows)}",
        f"probe_cell_rows                  : {len(probe_cell_rows)}",
        f"frame_summary_rows               : {len(frame_summary_rows)}",
        "",
        "owner_label_counts:",
    ]
    for key, value in owner_counter.most_common():
        summary_lines.append(f"  {key}: {value}")
    summary_lines.extend(["", "collision_kind_counts:"])
    for key, value in collision_counter.most_common():
        summary_lines.append(f"  {key}: {value}")
    summary_lines.extend(
        [
            "",
            f"second_probe_cells               : {second_probe_cells}",
            f"second_only_cells                : {second_only_cells}",
            f"second_with_pianoish_cells       : {second_with_pianoish_cells}",
            f"second_with_other_cells          : {second_with_other_cells}",
            f"second_clean_ratio               : {second_clean_ratio:.6f}",
            f"second_pianoish_collision_ratio  : {second_pianoish_collision_ratio:.6f}",
            "",
            f"second_frame_count               : {second_frame_count}",
            f"second_only_frames               : {second_only_frames}",
            f"second_with_pianoish_frames      : {second_with_pianoish_frames}",
            f"second_frame_pianoish_ratio      : {second_frame_pianoish_ratio:.6f}",
            "",
            "second_collision_breakdown:",
        ]
    )
    for key, value in second_contamination_counter.most_common():
        summary_lines.append(f"  {key}: {value}")
    summary_lines.extend(["", "frame_coexistence_counts:"])
    for key, value in frame_coexistence_counter.most_common():
        summary_lines.append(f"  {key}: {value}")
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "stage": "window_probe_ownership_builder",
                "inputs": {
                    "notechain_frames_csv": args.notechain_frames_csv,
                    "roles_csv": args.roles_csv,
                    "backbone_lineages_csv": args.backbone_lineages_csv,
                },
                "window": {
                    "start_sec": args.window_start_sec,
                    "end_sec": args.window_end_sec,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                },
                "result": {
                    "observation_rows": len(observation_rows),
                    "probe_cell_rows": len(probe_cell_rows),
                    "frame_summary_rows": len(frame_summary_rows),
                    "owner_label_counts": dict(owner_counter),
                    "owner_family_counts": dict(owner_family_counter),
                    "collision_kind_counts": dict(collision_counter),
                    "second_collision_breakdown": dict(second_contamination_counter),
                    "second_probe_cells": second_probe_cells,
                    "second_only_cells": second_only_cells,
                    "second_with_pianoish_cells": second_with_pianoish_cells,
                    "second_with_other_cells": second_with_other_cells,
                    "second_clean_ratio": second_clean_ratio,
                    "second_pianoish_collision_ratio": second_pianoish_collision_ratio,
                    "second_frame_count": second_frame_count,
                    "second_only_frames": second_only_frames,
                    "second_with_pianoish_frames": second_with_pianoish_frames,
                    "second_frame_pianoish_ratio": second_frame_pianoish_ratio,
                    "frame_coexistence_counts": dict(frame_coexistence_counter),
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
            "observation_rows": len(observation_rows),
            "probe_cell_rows": len(probe_cell_rows),
            "frame_summary_rows": len(frame_summary_rows),
        },
    )


if __name__ == "__main__":
    main()
