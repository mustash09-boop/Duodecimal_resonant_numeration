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


def _normalize_note(note: str) -> str:
    s = str(note or "").strip()
    if not s:
        return ""
    if "'" in s:
        return s.split("'", 1)[0] + "'-"
    if s.endswith("-"):
        return s[:-1] + "'-"
    return s + "'-"


def _pitchclass(note: str) -> str:
    token = _normalize_note(note)
    if "." not in token:
        return ""
    return token.split(".", 1)[1].split("'", 1)[0].strip()


def _parse_json_list(value: str) -> list[Any]:
    try:
        loaded = json.loads(str(value or "").strip() or "[]")
        return loaded if isinstance(loaded, list) else []
    except Exception:
        return []


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _overlap_count(left: set[Any], right: set[Any]) -> int:
    if not left or not right:
        return 0
    return len(left & right)


def _trajectory_link_score(
    *,
    prev_row: dict[str, Any],
    row: dict[str, Any],
    max_gap_frames: int,
) -> tuple[float, str]:
    prev_start = _safe_int(prev_row.get("start_frame"), 0)
    prev_end = _safe_int(prev_row.get("end_frame"), 0)
    start = _safe_int(row.get("start_frame"), 0)
    if start <= prev_start:
        return -1.0, "non_forward_time"
    gap = start - prev_end
    if gap <= 0 or gap > max_gap_frames:
        return -1.0, "gap_out_of_range"

    prev_micro = str(prev_row.get("anchor_micro_note_token", "")).strip()
    micro = str(row.get("anchor_micro_note_token", "")).strip()
    prev_coarse = str(prev_row.get("anchor_coarse_note", "")).strip()
    coarse = str(row.get("anchor_coarse_note", "")).strip()
    prev_pc = _pitchclass(prev_micro)
    pc = _pitchclass(micro)

    score = 0.0
    reasons: list[str] = []

    if prev_micro and micro and prev_micro == micro:
        score += 3.1
        reasons.append("same_anchor_micro")
    elif prev_coarse and coarse and prev_coarse == coarse:
        score += 2.0
        reasons.append("same_anchor_coarse")
    elif prev_pc and pc and prev_pc == pc:
        score += 0.9
        reasons.append("same_pitchclass")

    prev_micro_set = {str(item[0]) for item in _parse_json_list(prev_row.get("top_micro_note_hypotheses_json", "")) if isinstance(item, list) and item}
    micro_set = {str(item[0]) for item in _parse_json_list(row.get("top_micro_note_hypotheses_json", "")) if isinstance(item, list) and item}
    micro_overlap = _overlap_count(prev_micro_set, micro_set)
    if micro_overlap > 0:
        score += min(1.8, micro_overlap * 0.45)
        reasons.append(f"micro_overlap_{micro_overlap}")

    prev_probe_set = {int(item[0]) for item in _parse_json_list(prev_row.get("top_probes_json", "")) if isinstance(item, list) and item}
    probe_set = {int(item[0]) for item in _parse_json_list(row.get("top_probes_json", "")) if isinstance(item, list) and item}
    probe_overlap = _overlap_count(prev_probe_set, probe_set)
    if probe_overlap > 0:
        score += min(1.3, probe_overlap * 0.28)
        reasons.append(f"probe_overlap_{probe_overlap}")

    prev_rank_set = {int(item[0]) for item in _parse_json_list(prev_row.get("top_coarse_ranks_json", "")) if isinstance(item, list) and item}
    rank_set = {int(item[0]) for item in _parse_json_list(row.get("top_coarse_ranks_json", "")) if isinstance(item, list) and item}
    rank_overlap = _overlap_count(prev_rank_set, rank_set)
    if rank_overlap > 0:
        score += min(1.5, rank_overlap * 0.55)
        reasons.append(f"rank_overlap_{rank_overlap}")

    prev_pc_rank_set = {int(item[0]) for item in _parse_json_list(prev_row.get("top_pitchclass_ranks_json", "")) if isinstance(item, list) and item}
    pc_rank_set = {int(item[0]) for item in _parse_json_list(row.get("top_pitchclass_ranks_json", "")) if isinstance(item, list) and item}
    pc_rank_overlap = _overlap_count(prev_pc_rank_set, pc_rank_set)
    if pc_rank_overlap > 0:
        score += min(0.7, pc_rank_overlap * 0.25)
        reasons.append(f"pc_rank_overlap_{pc_rank_overlap}")

    prev_freq = _safe_float(prev_row.get("mean_frequency_hz"), 0.0)
    freq = _safe_float(row.get("mean_frequency_hz"), 0.0)
    if prev_freq > 0.0 and freq > 0.0:
        rel_freq = abs(freq - prev_freq) / max(prev_freq, 1e-9)
        if rel_freq <= 0.0025:
            score += 1.2
            reasons.append("tight_freq")
        elif rel_freq <= 0.0060:
            score += 0.6
            reasons.append("near_freq")

    prev_primary_share = _safe_float(prev_row.get("primary_micro_share"), 0.0)
    primary_share = _safe_float(row.get("primary_micro_share"), 0.0)
    score += min(prev_primary_share, primary_share) * 0.40

    prev_probe_share = _safe_float(prev_row.get("primary_probe_share"), 0.0)
    probe_share = _safe_float(row.get("primary_probe_share"), 0.0)
    score += min(prev_probe_share, probe_share) * 0.30

    prev_rank_share = _safe_float(prev_row.get("primary_coarse_rank_share"), 0.0)
    rank_share = _safe_float(row.get("primary_coarse_rank_share"), 0.0)
    score += min(prev_rank_share, rank_share) * 0.80

    prev_status = str(prev_row.get("trajectory_status", "")).strip()
    status = str(row.get("trajectory_status", "")).strip()
    if prev_status == "CONFIRMED_OBSERVATION_TRAJECTORY":
        score += 0.45
    if status == "CONFIRMED_OBSERVATION_TRAJECTORY":
        score += 0.45
    if prev_status == "PROBABLE_OBSERVATION_TRAJECTORY":
        score += 0.15
    if status == "PROBABLE_OBSERVATION_TRAJECTORY":
        score += 0.15

    score += max(0.0, 0.70 - 0.15 * gap)
    return score, "|".join(reasons) if reasons else "weak"


def _classify_chain(
    *,
    micro_counter: Counter[str],
    coarse_counter: Counter[str],
    suffix_counter: Counter[str],
    trajectory_class_counter: Counter[str],
    trajectory_status_counter: Counter[str],
    trajectory_count: int,
    observation_frame_count: int,
) -> tuple[str, str]:
    top_micro_count = micro_counter.most_common(1)[0][1] if micro_counter else 0
    top_coarse_count = coarse_counter.most_common(1)[0][1] if coarse_counter else 0
    micro_share = top_micro_count / max(sum(micro_counter.values()), 1)
    coarse_share = top_coarse_count / max(sum(coarse_counter.values()), 1)
    suffix_div = len(suffix_counter)

    if trajectory_count >= 3 and observation_frame_count >= 12 and micro_share >= 0.72 and suffix_div <= 4:
        chain_kind = "EXACT_COHORT_TRAJECTORY_NOTECHAIN"
    elif trajectory_class_counter.get("COHORT_MICRO_DRIFT_TRAJECTORY", 0) >= max(2, trajectory_count // 2):
        chain_kind = "COHORT_DRIFT_BACKBONE_CHAIN"
    elif trajectory_class_counter.get("EXACT_MICRO_COHORT_TRAJECTORY", 0) >= max(2, trajectory_count // 2):
        chain_kind = "EXACT_MICRO_COHORT_CHAIN"
    elif trajectory_class_counter.get("EXACT_PROBE_COHORT_TRAJECTORY", 0) >= max(2, trajectory_count // 2) and coarse_share >= 0.70:
        chain_kind = "EXACT_PROBE_COHORT_BACKBONE_CHAIN"
    else:
        chain_kind = "LOCAL_COHORT_TRAJECTORY_CHAIN"

    if observation_frame_count >= 16 or trajectory_status_counter.get("CONFIRMED_OBSERVATION_TRAJECTORY", 0) >= 2:
        confirmation = "CONFIRMED_CHAIN_V3"
    elif observation_frame_count >= 8 or trajectory_status_counter.get("PROBABLE_OBSERVATION_TRAJECTORY", 0) >= 2:
        confirmation = "PROBABLE_CHAIN_V3"
    else:
        confirmation = "WEAK_CHAIN_V3"
    return chain_kind, confirmation


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build micro notechains v3 from cohort-aware observation trajectories v2."
    )
    ap.add_argument("--trajectory-csv", required=True)
    ap.add_argument("--trajectory-frames-csv", required=True)
    ap.add_argument("--out-chains-csv", required=True)
    ap.add_argument("--out-chain-frames-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--progress-json", default="")
    ap.add_argument("--max-gap-frames", type=int, default=4)
    ap.add_argument("--min-link-score", type=float, default=5.10)
    args = ap.parse_args()

    _write_progress(
        args.progress_json,
        {
            "status": "running",
            "phase": "loading_inputs",
            "processed_trajectories": 0,
            "total_trajectories": 0,
            "built_chain_count": 0,
        },
    )

    trajectory_rows = _load_csv(Path(args.trajectory_csv))
    trajectory_frame_rows = _load_csv(Path(args.trajectory_frames_csv))
    trajectory_rows.sort(
        key=lambda row: (
            _safe_int(row.get("start_frame"), 0),
            _safe_int(row.get("end_frame"), 0),
            _safe_int(row.get("trajectory_id"), 0),
        )
    )
    total_trajectories = len(trajectory_rows)

    _write_progress(
        args.progress_json,
        {
            "status": "running",
            "phase": "linking_trajectories",
            "processed_trajectories": 0,
            "total_trajectories": total_trajectories,
            "built_chain_count": 0,
        },
    )

    active_chain_ids: list[int] = []
    chain_tail_by_id: dict[int, dict[str, Any]] = {}
    chain_trajectory_ids: dict[int, list[int]] = defaultdict(list)
    trajectory_link_reason: dict[int, str] = {}
    next_chain_id = 1
    trajectory_by_id: dict[int, dict[str, Any]] = {}

    for idx, row in enumerate(trajectory_rows, start=1):
        trajectory_id = _safe_int(row.get("trajectory_id"), 0)
        start_frame = _safe_int(row.get("start_frame"), 0)
        trajectory_by_id[trajectory_id] = row

        keep_chain_ids: list[int] = []
        for chain_id in active_chain_ids:
            tail = chain_tail_by_id[chain_id]
            if start_frame - _safe_int(tail.get("end_frame"), 0) <= int(args.max_gap_frames):
                keep_chain_ids.append(chain_id)
        active_chain_ids = keep_chain_ids

        best_chain_id = 0
        best_score = -1.0
        best_reason = ""
        for chain_id in active_chain_ids:
            tail = chain_tail_by_id[chain_id]
            score, reason = _trajectory_link_score(
                prev_row=tail,
                row=row,
                max_gap_frames=int(args.max_gap_frames),
            )
            if score > best_score:
                best_chain_id = chain_id
                best_score = score
                best_reason = reason

        if best_chain_id and best_score >= float(args.min_link_score):
            chain_id = best_chain_id
            chain_tail_by_id[chain_id] = row
            chain_trajectory_ids[chain_id].append(trajectory_id)
            trajectory_link_reason[trajectory_id] = best_reason
        else:
            chain_id = next_chain_id
            next_chain_id += 1
            chain_tail_by_id[chain_id] = row
            chain_trajectory_ids[chain_id].append(trajectory_id)
            trajectory_link_reason[trajectory_id] = "new_chain"
            active_chain_ids.append(chain_id)

        if idx % 4000 == 0 or idx == total_trajectories:
            _write_progress(
                args.progress_json,
                {
                    "status": "running",
                    "phase": "linking_trajectories",
                    "processed_trajectories": idx,
                    "total_trajectories": total_trajectories,
                    "built_chain_count": len(chain_trajectory_ids),
                },
            )

    frame_rows_by_trajectory: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in trajectory_frame_rows:
        frame_rows_by_trajectory[_safe_int(row.get("trajectory_id"), 0)].append(row)

    _write_progress(
        args.progress_json,
        {
            "status": "running",
            "phase": "writing_outputs",
            "processed_trajectories": total_trajectories,
            "total_trajectories": total_trajectories,
            "built_chain_count": len(chain_trajectory_ids),
        },
    )

    chain_rows: list[dict[str, Any]] = []
    chain_frame_rows: list[dict[str, Any]] = []
    chain_kind_counter: Counter[str] = Counter()
    confirmation_counter: Counter[str] = Counter()

    for chain_id in sorted(chain_trajectory_ids):
        trajectory_ids = chain_trajectory_ids[chain_id]
        rows = [trajectory_by_id[trajectory_id] for trajectory_id in trajectory_ids if trajectory_id in trajectory_by_id]
        if not rows:
            continue
        rows.sort(
            key=lambda row: (
                _safe_int(row.get("start_frame"), 0),
                _safe_int(row.get("end_frame"), 0),
                _safe_int(row.get("trajectory_id"), 0),
            )
        )

        micro_counter: Counter[str] = Counter()
        coarse_counter: Counter[str] = Counter()
        suffix_counter: Counter[str] = Counter()
        probe_counter: Counter[str] = Counter()
        coarse_rank_counter: Counter[str] = Counter()
        pc_rank_counter: Counter[str] = Counter()
        trajectory_class_counter: Counter[str] = Counter()
        trajectory_status_counter: Counter[str] = Counter()
        freq_values: list[float] = []
        energy_values: list[float] = []
        observation_path: list[str] = []
        accepted_links: list[str] = []
        observation_frame_count = 0
        start_frame = min(_safe_int(row.get("start_frame"), 0) for row in rows)
        end_frame = max(_safe_int(row.get("end_frame"), 0) for row in rows)
        start_sec = min(_safe_float(row.get("start_time_sec"), 0.0) for row in rows)
        end_sec = max(_safe_float(row.get("end_time_sec"), 0.0) for row in rows)

        for row in rows:
            trajectory_id = _safe_int(row.get("trajectory_id"), 0)
            accepted_links.append(trajectory_link_reason.get(trajectory_id, ""))
            trajectory_class_counter[str(row.get("trajectory_class", "")).strip()] += 1
            trajectory_status_counter[str(row.get("trajectory_status", "")).strip()] += 1
            micro_counter[str(row.get("anchor_micro_note_token", "")).strip()] += _safe_int(row.get("frame_count"), 0)
            coarse_counter[str(row.get("anchor_coarse_note", "")).strip()] += _safe_int(row.get("frame_count"), 0)
            for item in _parse_json_list(row.get("top_micro_suffixes_json", "")):
                if isinstance(item, list) and len(item) >= 2:
                    suffix_counter[str(item[0])] += _safe_int(item[1], 0)
            for item in _parse_json_list(row.get("top_probes_json", "")):
                if isinstance(item, list) and len(item) >= 2:
                    probe_counter[str(item[0])] += _safe_int(item[1], 0)
            for item in _parse_json_list(row.get("top_coarse_ranks_json", "")):
                if isinstance(item, list) and len(item) >= 2:
                    coarse_rank_counter[str(item[0])] += _safe_int(item[1], 0)
            for item in _parse_json_list(row.get("top_pitchclass_ranks_json", "")):
                if isinstance(item, list) and len(item) >= 2:
                    pc_rank_counter[str(item[0])] += _safe_int(item[1], 0)
            freq_values.append(_safe_float(row.get("mean_frequency_hz"), 0.0))
            energy_values.append(_safe_float(row.get("mean_energy"), 0.0))

            for frow in frame_rows_by_trajectory.get(trajectory_id, []):
                observation_frame_count += 1
                observation_path.append(str(frow.get("observed_micro_symbol", "")).strip())
                chain_frame_rows.append(
                    {
                        "chain_id": chain_id,
                        "frame_index": _safe_int(frow.get("frame_index"), 0),
                        "time_sec": frow.get("time_sec", ""),
                        "trajectory_id": trajectory_id,
                        "event_id": _safe_int(frow.get("event_id"), 0),
                        "slot_index": _safe_int(frow.get("slot_index"), 0),
                        "probe_index": _safe_int(frow.get("probe_index"), 0),
                        "observed_micro_symbol": str(frow.get("observed_micro_symbol", "")).strip(),
                        "observed_coarse_symbol": str(frow.get("observed_coarse_symbol", "")).strip(),
                        "micro_suffix": str(frow.get("micro_suffix", "")).strip(),
                        "frequency_hz": frow.get("frequency_hz", ""),
                        "energy": frow.get("energy", ""),
                        "rise": frow.get("rise", ""),
                        "continuation": frow.get("continuation", ""),
                        "coarse_group_rank": _safe_int(frow.get("coarse_group_rank"), 0),
                        "coarse_group_size": _safe_int(frow.get("coarse_group_size"), 0),
                        "pitchclass_group_rank": _safe_int(frow.get("pitchclass_group_rank"), 0),
                        "pitchclass_group_size": _safe_int(frow.get("pitchclass_group_size"), 0),
                        "observation_kind": "cohort_trajectory_chain_observation",
                    }
                )

        anchor_micro = micro_counter.most_common(1)[0][0] if micro_counter else ""
        anchor_coarse = coarse_counter.most_common(1)[0][0] if coarse_counter else ""
        chain_kind, confirmation_level = _classify_chain(
            micro_counter=micro_counter,
            coarse_counter=coarse_counter,
            suffix_counter=suffix_counter,
            trajectory_class_counter=trajectory_class_counter,
            trajectory_status_counter=trajectory_status_counter,
            trajectory_count=len(rows),
            observation_frame_count=observation_frame_count,
        )
        chain_kind_counter[chain_kind] += 1
        confirmation_counter[confirmation_level] += 1

        chain_rows.append(
            {
                "chain_id": chain_id,
                "source_mode": "MICRO_COHORT_TRAJECTORY_TO_CHAIN_V3",
                "start_frame": start_frame,
                "end_frame": end_frame,
                "duration_frames": end_frame - start_frame + 1,
                "start_time_sec": f"{start_sec:.9f}",
                "end_time_sec": f"{end_sec:.9f}",
                "trajectory_count": len(rows),
                "observation_frame_count": observation_frame_count,
                "anchor_micro_note_token": anchor_micro,
                "anchor_coarse_note": anchor_coarse,
                "micro_note_hypothesis_count": len(micro_counter),
                "coarse_note_hypothesis_count": len(coarse_counter),
                "micro_suffix_diversity": len(suffix_counter),
                "probe_diversity": len(probe_counter),
                "coarse_rank_diversity": len(coarse_rank_counter),
                "pitchclass_rank_diversity": len(pc_rank_counter),
                "mean_frequency_hz": f"{_mean(freq_values):.9f}",
                "micro_frequency_span_hz": f"{(max(freq_values) - min(freq_values)) if freq_values else 0.0:.9f}",
                "mean_energy": f"{_mean(energy_values):.9f}",
                "chain_structure_class": chain_kind,
                "confirmation_level": confirmation_level,
                "top_micro_note_hypotheses_json": json.dumps(micro_counter.most_common(16), ensure_ascii=False),
                "top_coarse_note_hypotheses_json": json.dumps(coarse_counter.most_common(12), ensure_ascii=False),
                "dominant_micro_suffixes_json": json.dumps(suffix_counter.most_common(16), ensure_ascii=False),
                "dominant_probes_json": json.dumps(probe_counter.most_common(16), ensure_ascii=False),
                "dominant_coarse_ranks_json": json.dumps(coarse_rank_counter.most_common(12), ensure_ascii=False),
                "dominant_pitchclass_ranks_json": json.dumps(pc_rank_counter.most_common(12), ensure_ascii=False),
                "accepted_link_kinds_json": json.dumps(accepted_links, ensure_ascii=False),
                "trajectory_class_counts_json": json.dumps(dict(trajectory_class_counter), ensure_ascii=False),
                "trajectory_status_counts_json": json.dumps(dict(trajectory_status_counter), ensure_ascii=False),
                "source_trajectory_ids_json": json.dumps(trajectory_ids, ensure_ascii=False),
                "observed_micro_path_json": json.dumps(observation_path, ensure_ascii=False),
            }
        )

    chain_fields = [
        "chain_id",
        "source_mode",
        "start_frame",
        "end_frame",
        "duration_frames",
        "start_time_sec",
        "end_time_sec",
        "trajectory_count",
        "observation_frame_count",
        "anchor_micro_note_token",
        "anchor_coarse_note",
        "micro_note_hypothesis_count",
        "coarse_note_hypothesis_count",
        "micro_suffix_diversity",
        "probe_diversity",
        "coarse_rank_diversity",
        "pitchclass_rank_diversity",
        "mean_frequency_hz",
        "micro_frequency_span_hz",
        "mean_energy",
        "chain_structure_class",
        "confirmation_level",
        "top_micro_note_hypotheses_json",
        "top_coarse_note_hypotheses_json",
        "dominant_micro_suffixes_json",
        "dominant_probes_json",
        "dominant_coarse_ranks_json",
        "dominant_pitchclass_ranks_json",
        "accepted_link_kinds_json",
        "trajectory_class_counts_json",
        "trajectory_status_counts_json",
        "source_trajectory_ids_json",
        "observed_micro_path_json",
    ]
    frame_fields = [
        "chain_id",
        "frame_index",
        "time_sec",
        "trajectory_id",
        "event_id",
        "slot_index",
        "probe_index",
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
    ]

    out_chain_csv = Path(args.out_chains_csv)
    out_chain_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_chain_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=chain_fields)
        writer.writeheader()
        for row in chain_rows:
            writer.writerow({key: row.get(key, "") for key in chain_fields})

    out_frame_csv = Path(args.out_chain_frames_csv)
    with out_frame_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=frame_fields)
        writer.writeheader()
        for row in chain_frame_rows:
            writer.writerow({key: row.get(key, "") for key in frame_fields})

    summary_lines = [
        "MICRO NOTECHAIN BUILDER V3",
        "=" * 72,
        "source_mode               : MICRO_COHORT_TRAJECTORY_TO_CHAIN_V3",
        f"input_trajectory_rows     : {len(trajectory_rows)}",
        f"input_trajectory_frames   : {len(trajectory_frame_rows)}",
        f"built_chain_count         : {len(chain_rows)}",
        f"built_chain_frame_rows    : {len(chain_frame_rows)}",
        f"max_gap_frames            : {int(args.max_gap_frames)}",
        f"min_link_score            : {float(args.min_link_score):.3f}",
        "",
        "chain_structure_counts:",
    ]
    for key, value in chain_kind_counter.most_common():
        summary_lines.append(f"  {key}: {value}")
    summary_lines.extend(["", "confirmation_counts:"])
    for key, value in confirmation_counter.most_common():
        summary_lines.append(f"  {key}: {value}")
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "stage": "micro_notechain_builder_v3",
                "source_mode": "MICRO_COHORT_TRAJECTORY_TO_CHAIN_V3",
                "inputs": {
                    "trajectory_csv": args.trajectory_csv,
                    "trajectory_frames_csv": args.trajectory_frames_csv,
                },
                "parameters": {
                    "max_gap_frames": int(args.max_gap_frames),
                    "min_link_score": float(args.min_link_score),
                },
                "result": {
                    "input_trajectory_rows": len(trajectory_rows),
                    "input_trajectory_frames": len(trajectory_frame_rows),
                    "built_chain_count": len(chain_rows),
                    "built_chain_frame_rows": len(chain_frame_rows),
                    "chain_structure_counts": dict(chain_kind_counter),
                    "confirmation_counts": dict(confirmation_counter),
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
            "processed_trajectories": total_trajectories,
            "total_trajectories": total_trajectories,
            "built_chain_count": len(chain_rows),
        },
    )


if __name__ == "__main__":
    main()
