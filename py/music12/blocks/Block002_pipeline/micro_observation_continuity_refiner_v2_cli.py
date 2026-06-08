# -*- coding: ascii -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
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


def _micro_suffix(note_token: str) -> str:
    s = str(note_token or "").strip()
    if "'" not in s:
        return "-"
    return s.split("'", 1)[1] or "-"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass
class ObservationTrajectory:
    trajectory_id: int
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float
    last_seen_frame: int
    last_probe_index: int
    last_note_token: str
    last_coarse_note: str
    last_pitchclass: str
    last_frequency_hz: float
    last_slot_index: int
    last_coarse_rank: int
    last_coarse_group_size: int
    last_pitchclass_rank: int
    last_pitchclass_group_size: int
    frame_count: int = 0
    note_counter: Counter[str] = field(default_factory=Counter)
    coarse_counter: Counter[str] = field(default_factory=Counter)
    suffix_counter: Counter[str] = field(default_factory=Counter)
    probe_counter: Counter[int] = field(default_factory=Counter)
    slot_counter: Counter[int] = field(default_factory=Counter)
    coarse_rank_counter: Counter[int] = field(default_factory=Counter)
    pitchclass_rank_counter: Counter[int] = field(default_factory=Counter)
    frequency_values: list[float] = field(default_factory=list)
    energy_values: list[float] = field(default_factory=list)
    continuation_values: list[float] = field(default_factory=list)
    rise_values: list[float] = field(default_factory=list)
    micro_path: list[str] = field(default_factory=list)

    def add(self, row: dict[str, Any]) -> None:
        frame_index = _safe_int(row.get("frame_index"), 0)
        time_sec = _safe_float(row.get("time_sec"), 0.0)
        probe_index = _safe_int(row.get("probe_index"), 0)
        note_token = str(row.get("note_token", "")).strip()
        coarse_note = str(row.get("coarse_note_overlay", "")).strip()
        slot_index = _safe_int(row.get("slot_index"), 0)
        frequency_hz = _safe_float(row.get("frequency_hz"), 0.0)
        self.end_frame = frame_index
        self.end_time_sec = time_sec
        self.last_seen_frame = frame_index
        self.last_probe_index = probe_index
        self.last_note_token = note_token
        self.last_coarse_note = coarse_note
        self.last_pitchclass = _pitchclass(note_token)
        self.last_slot_index = slot_index
        self.last_frequency_hz = frequency_hz
        self.last_coarse_rank = _safe_int(row.get("coarse_group_rank"), 0)
        self.last_coarse_group_size = _safe_int(row.get("coarse_group_size"), 0)
        self.last_pitchclass_rank = _safe_int(row.get("pitchclass_group_rank"), 0)
        self.last_pitchclass_group_size = _safe_int(row.get("pitchclass_group_size"), 0)
        self.frame_count += 1
        self.note_counter[note_token] += 1
        self.coarse_counter[coarse_note] += 1
        self.suffix_counter[_micro_suffix(note_token)] += 1
        self.probe_counter[probe_index] += 1
        self.slot_counter[slot_index] += 1
        self.coarse_rank_counter[self.last_coarse_rank] += 1
        self.pitchclass_rank_counter[self.last_pitchclass_rank] += 1
        self.frequency_values.append(frequency_hz)
        self.energy_values.append(_safe_float(row.get("energy"), 0.0))
        self.continuation_values.append(_safe_float(row.get("continuation"), 0.0))
        self.rise_values.append(_safe_float(row.get("rise"), 0.0))
        self.micro_path.append(note_token)


def _annotate_frame_rows(frame_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated = [dict(row) for row in frame_rows]

    coarse_groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    pitchclass_groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for idx, row in enumerate(annotated):
        coarse_groups[str(row.get("coarse_note_overlay", "")).strip()].append((idx, row))
        pitchclass_groups[_pitchclass(str(row.get("note_token", "")).strip())].append((idx, row))

    for _, items in coarse_groups.items():
        items.sort(key=lambda item: (_safe_float(item[1].get("frequency_hz"), 0.0), _safe_int(item[1].get("probe_index"), 0)))
        group_size = len(items)
        for rank, (_, row) in enumerate(items, start=1):
            row["coarse_group_rank"] = rank
            row["coarse_group_size"] = group_size
            row["coarse_group_reverse_rank"] = group_size - rank + 1

    for _, items in pitchclass_groups.items():
        items.sort(key=lambda item: (_safe_float(item[1].get("frequency_hz"), 0.0), _safe_int(item[1].get("probe_index"), 0)))
        group_size = len(items)
        for rank, (_, row) in enumerate(items, start=1):
            row["pitchclass_group_rank"] = rank
            row["pitchclass_group_size"] = group_size
            row["pitchclass_group_reverse_rank"] = group_size - rank + 1

    return annotated


def _match_score(
    traj: ObservationTrajectory,
    row: dict[str, Any],
    max_gap_frames: int,
    rel_freq_tolerance: float,
) -> tuple[float, str]:
    frame_index = _safe_int(row.get("frame_index"), 0)
    gap = frame_index - traj.last_seen_frame
    if gap <= 0 or gap > max_gap_frames:
        return -1.0, "gap_out_of_range"

    note_token = str(row.get("note_token", "")).strip()
    coarse_note = str(row.get("coarse_note_overlay", "")).strip()
    pitchclass = _pitchclass(note_token)
    slot_index = _safe_int(row.get("slot_index"), 0)
    probe_index = _safe_int(row.get("probe_index"), 0)
    frequency_hz = _safe_float(row.get("frequency_hz"), 0.0)
    continuation = _safe_float(row.get("continuation"), 0.0)
    rise = _safe_float(row.get("rise"), 0.0)
    coarse_rank = _safe_int(row.get("coarse_group_rank"), 0)
    coarse_group_size = _safe_int(row.get("coarse_group_size"), 0)
    pitchclass_rank = _safe_int(row.get("pitchclass_group_rank"), 0)
    pitchclass_group_size = _safe_int(row.get("pitchclass_group_size"), 0)

    score = 0.0
    reasons: list[str] = []

    if probe_index == traj.last_probe_index:
        score += 5.0
        reasons.append("same_probe")
    if note_token == traj.last_note_token:
        score += 3.0
        reasons.append("same_note")
    elif coarse_note == traj.last_coarse_note:
        score += 1.6
        reasons.append("same_coarse")
    elif pitchclass and pitchclass == traj.last_pitchclass:
        score += 0.6
        reasons.append("same_pitchclass")

    if slot_index == traj.last_slot_index:
        score += 0.25
        reasons.append("same_slot")

    if coarse_note and coarse_note == traj.last_coarse_note and coarse_rank and traj.last_coarse_rank:
        rank_gap = abs(coarse_rank - traj.last_coarse_rank)
        if rank_gap == 0:
            score += 1.8
            reasons.append("same_coarse_rank")
        elif rank_gap == 1:
            score += 0.9
            reasons.append("near_coarse_rank")
        if coarse_group_size and coarse_group_size == traj.last_coarse_group_size:
            score += 0.35
            reasons.append("same_coarse_group_size")

    if pitchclass and pitchclass == traj.last_pitchclass and pitchclass_rank and traj.last_pitchclass_rank:
        rank_gap = abs(pitchclass_rank - traj.last_pitchclass_rank)
        if rank_gap == 0:
            score += 0.75
            reasons.append("same_pitchclass_rank")
        elif rank_gap == 1:
            score += 0.30
            reasons.append("near_pitchclass_rank")
        if pitchclass_group_size and pitchclass_group_size == traj.last_pitchclass_group_size:
            score += 0.20
            reasons.append("same_pitchclass_group_size")

    if traj.last_frequency_hz > 0.0 and frequency_hz > 0.0:
        rel_freq = abs(frequency_hz - traj.last_frequency_hz) / max(traj.last_frequency_hz, 1e-9)
        if rel_freq <= rel_freq_tolerance:
            score += 1.5
            reasons.append("tight_freq")
        elif rel_freq <= rel_freq_tolerance * 2.5:
            score += 0.6
            reasons.append("near_freq")

    score += max(0.0, 0.75 - 0.15 * gap)
    score += continuation * 0.35
    score += min(rise, 0.10)
    return score, "|".join(reasons) if reasons else "weak"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build early micro-observation continuity trajectories v2 with cohort-aware one-to-one matching inside simultaneous coarse-note families."
    )
    ap.add_argument("--frame-slots-csv", required=True)
    ap.add_argument("--out-trajectories-csv", required=True)
    ap.add_argument("--out-trajectory-frames-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--progress-json", default="")
    ap.add_argument("--max-gap-frames", type=int, default=3)
    ap.add_argument("--relative-frequency-tolerance", type=float, default=0.0035)
    ap.add_argument("--min-match-score", type=float, default=3.7)
    args = ap.parse_args()

    rows = _load_csv(Path(args.frame_slots_csv))
    rows.sort(
        key=lambda row: (
            _safe_int(row.get("frame_index"), 0),
            _safe_int(row.get("rank_in_frame"), 0),
            _safe_int(row.get("probe_index"), 0),
        )
    )
    total_rows = len(rows)

    _write_progress(
        args.progress_json,
        {
            "status": "running",
            "phase": "building_trajectories_v2",
            "processed_rows": 0,
            "total_rows": total_rows,
            "trajectory_count": 0,
        },
    )

    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_frame[_safe_int(row.get("frame_index"), 0)].append(row)

    active: list[ObservationTrajectory] = []
    finalized: list[ObservationTrajectory] = []
    next_id = 1
    frame_rows_out: list[dict[str, Any]] = []
    processed_rows = 0

    def _finalize_stale(current_frame: int) -> None:
        stale: list[ObservationTrajectory] = []
        keep: list[ObservationTrajectory] = []
        for traj in active:
            if current_frame - traj.last_seen_frame > int(args.max_gap_frames):
                stale.append(traj)
            else:
                keep.append(traj)
        active[:] = keep
        finalized.extend(stale)

    frame_keys = sorted(by_frame)
    for frame_idx, frame_index in enumerate(frame_keys, start=1):
        _finalize_stale(frame_index)
        annotated_rows = _annotate_frame_rows(by_frame[frame_index])

        candidates: list[tuple[float, int, int, str]] = []
        for row_pos, row in enumerate(annotated_rows):
            for traj in active:
                score, reason = _match_score(
                    traj,
                    row,
                    max_gap_frames=int(args.max_gap_frames),
                    rel_freq_tolerance=float(args.relative_frequency_tolerance),
                )
                if score >= float(args.min_match_score):
                    candidates.append((score, row_pos, traj.trajectory_id, reason))

        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        assigned_rows: set[int] = set()
        assigned_trajs: set[int] = set()
        row_to_assignment: dict[int, tuple[int, str]] = {}

        for _, row_pos, traj_id, reason in candidates:
            if row_pos in assigned_rows or traj_id in assigned_trajs:
                continue
            assigned_rows.add(row_pos)
            assigned_trajs.add(traj_id)
            row_to_assignment[row_pos] = (traj_id, reason)

        active_by_id = {traj.trajectory_id: traj for traj in active}
        for row_pos, row in enumerate(annotated_rows):
            assignment = row_to_assignment.get(row_pos)
            if assignment:
                traj = active_by_id[assignment[0]]
                link_reason = assignment[1]
            else:
                traj = ObservationTrajectory(
                    trajectory_id=next_id,
                    start_frame=frame_index,
                    end_frame=frame_index,
                    start_time_sec=_safe_float(row.get("time_sec"), 0.0),
                    end_time_sec=_safe_float(row.get("time_sec"), 0.0),
                    last_seen_frame=frame_index,
                    last_probe_index=_safe_int(row.get("probe_index"), 0),
                    last_note_token=str(row.get("note_token", "")).strip(),
                    last_coarse_note=str(row.get("coarse_note_overlay", "")).strip(),
                    last_pitchclass=_pitchclass(str(row.get("note_token", "")).strip()),
                    last_frequency_hz=_safe_float(row.get("frequency_hz"), 0.0),
                    last_slot_index=_safe_int(row.get("slot_index"), 0),
                    last_coarse_rank=_safe_int(row.get("coarse_group_rank"), 0),
                    last_coarse_group_size=_safe_int(row.get("coarse_group_size"), 0),
                    last_pitchclass_rank=_safe_int(row.get("pitchclass_group_rank"), 0),
                    last_pitchclass_group_size=_safe_int(row.get("pitchclass_group_size"), 0),
                )
                next_id += 1
                active.append(traj)
                link_reason = "new_trajectory"

            traj.add(row)
            frame_rows_out.append(
                {
                    "trajectory_id": traj.trajectory_id,
                    "frame_index": frame_index,
                    "time_sec": row.get("time_sec", ""),
                    "event_id": _safe_int(row.get("event_id"), 0),
                    "slot_index": _safe_int(row.get("slot_index"), 0),
                    "probe_index": _safe_int(row.get("probe_index"), 0),
                    "observed_micro_symbol": str(row.get("note_token", "")).strip(),
                    "observed_coarse_symbol": str(row.get("coarse_note_overlay", "")).strip(),
                    "micro_suffix": str(row.get("micro_suffix", "")).strip(),
                    "frequency_hz": row.get("frequency_hz", ""),
                    "energy": row.get("energy", ""),
                    "rise": row.get("rise", ""),
                    "continuation": row.get("continuation", ""),
                    "coarse_group_rank": _safe_int(row.get("coarse_group_rank"), 0),
                    "coarse_group_size": _safe_int(row.get("coarse_group_size"), 0),
                    "pitchclass_group_rank": _safe_int(row.get("pitchclass_group_rank"), 0),
                    "pitchclass_group_size": _safe_int(row.get("pitchclass_group_size"), 0),
                    "link_reason": link_reason,
                }
            )

        processed_rows += len(annotated_rows)
        if frame_idx % 256 == 0 or frame_idx == len(frame_keys):
            _write_progress(
                args.progress_json,
                {
                    "status": "running",
                    "phase": "building_trajectories_v2",
                    "processed_rows": processed_rows,
                    "total_rows": total_rows,
                    "trajectory_count": next_id - 1,
                },
            )

    finalized.extend(active)
    active.clear()

    trajectory_rows: list[dict[str, Any]] = []
    class_counter: Counter[str] = Counter()
    status_counter: Counter[str] = Counter()

    for traj in sorted(finalized, key=lambda item: (item.start_frame, item.end_frame, item.trajectory_id)):
        anchor_micro = traj.note_counter.most_common(1)[0][0] if traj.note_counter else ""
        anchor_coarse = traj.coarse_counter.most_common(1)[0][0] if traj.coarse_counter else ""
        anchor_probe = traj.probe_counter.most_common(1)[0][0] if traj.probe_counter else 0
        micro_share = (traj.note_counter.most_common(1)[0][1] / max(traj.frame_count, 1)) if traj.note_counter else 0.0
        probe_share = (traj.probe_counter.most_common(1)[0][1] / max(traj.frame_count, 1)) if traj.probe_counter else 0.0
        rank_share = (traj.coarse_rank_counter.most_common(1)[0][1] / max(traj.frame_count, 1)) if traj.coarse_rank_counter else 0.0

        if probe_share >= 0.75 and micro_share >= 0.55:
            traj_class = "EXACT_PROBE_COHORT_TRAJECTORY"
        elif micro_share >= 0.65 and rank_share >= 0.55:
            traj_class = "EXACT_MICRO_COHORT_TRAJECTORY"
        elif anchor_coarse and len(traj.coarse_counter) == 1 and len(traj.suffix_counter) >= 2:
            traj_class = "COHORT_MICRO_DRIFT_TRAJECTORY"
        else:
            traj_class = "LOCAL_COHORT_OBSERVATION_TRAJECTORY"

        if traj.frame_count >= 8:
            status = "CONFIRMED_OBSERVATION_TRAJECTORY"
        elif traj.frame_count >= 4:
            status = "PROBABLE_OBSERVATION_TRAJECTORY"
        else:
            status = "WEAK_OBSERVATION_TRAJECTORY"

        class_counter[traj_class] += 1
        status_counter[status] += 1

        trajectory_rows.append(
            {
                "trajectory_id": traj.trajectory_id,
                "source_mode": "MICRO_OBSERVATION_CONTINUITY_V2",
                "start_frame": traj.start_frame,
                "end_frame": traj.end_frame,
                "duration_frames": traj.end_frame - traj.start_frame + 1,
                "start_time_sec": f"{traj.start_time_sec:.9f}",
                "end_time_sec": f"{traj.end_time_sec:.9f}",
                "frame_count": traj.frame_count,
                "anchor_micro_note_token": anchor_micro,
                "anchor_coarse_note": anchor_coarse,
                "anchor_probe_index": anchor_probe,
                "micro_note_hypothesis_count": len(traj.note_counter),
                "coarse_note_hypothesis_count": len(traj.coarse_counter),
                "micro_suffix_diversity": len(traj.suffix_counter),
                "probe_diversity": len(traj.probe_counter),
                "slot_diversity": len(traj.slot_counter),
                "coarse_rank_diversity": len(traj.coarse_rank_counter),
                "pitchclass_rank_diversity": len(traj.pitchclass_rank_counter),
                "primary_micro_share": f"{micro_share:.9f}",
                "primary_probe_share": f"{probe_share:.9f}",
                "primary_coarse_rank_share": f"{rank_share:.9f}",
                "mean_frequency_hz": f"{_mean(traj.frequency_values):.9f}",
                "mean_energy": f"{_mean(traj.energy_values):.9f}",
                "mean_continuation": f"{_mean(traj.continuation_values):.9f}",
                "mean_rise": f"{_mean(traj.rise_values):.9f}",
                "trajectory_class": traj_class,
                "trajectory_status": status,
                "top_micro_note_hypotheses_json": json.dumps(traj.note_counter.most_common(12), ensure_ascii=False),
                "top_coarse_note_hypotheses_json": json.dumps(traj.coarse_counter.most_common(8), ensure_ascii=False),
                "top_micro_suffixes_json": json.dumps(traj.suffix_counter.most_common(12), ensure_ascii=False),
                "top_probes_json": json.dumps(traj.probe_counter.most_common(12), ensure_ascii=False),
                "top_coarse_ranks_json": json.dumps(traj.coarse_rank_counter.most_common(8), ensure_ascii=False),
                "top_pitchclass_ranks_json": json.dumps(traj.pitchclass_rank_counter.most_common(8), ensure_ascii=False),
                "micro_path_json": json.dumps(traj.micro_path, ensure_ascii=False),
            }
        )

    traj_fields = [
        "trajectory_id",
        "source_mode",
        "start_frame",
        "end_frame",
        "duration_frames",
        "start_time_sec",
        "end_time_sec",
        "frame_count",
        "anchor_micro_note_token",
        "anchor_coarse_note",
        "anchor_probe_index",
        "micro_note_hypothesis_count",
        "coarse_note_hypothesis_count",
        "micro_suffix_diversity",
        "probe_diversity",
        "slot_diversity",
        "coarse_rank_diversity",
        "pitchclass_rank_diversity",
        "primary_micro_share",
        "primary_probe_share",
        "primary_coarse_rank_share",
        "mean_frequency_hz",
        "mean_energy",
        "mean_continuation",
        "mean_rise",
        "trajectory_class",
        "trajectory_status",
        "top_micro_note_hypotheses_json",
        "top_coarse_note_hypotheses_json",
        "top_micro_suffixes_json",
        "top_probes_json",
        "top_coarse_ranks_json",
        "top_pitchclass_ranks_json",
        "micro_path_json",
    ]
    frame_fields = [
        "trajectory_id",
        "frame_index",
        "time_sec",
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
        "link_reason",
    ]

    out_traj_csv = Path(args.out_trajectories_csv)
    out_traj_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_traj_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=traj_fields)
        writer.writeheader()
        for row in trajectory_rows:
            writer.writerow({key: row.get(key, "") for key in traj_fields})

    out_frames_csv = Path(args.out_trajectory_frames_csv)
    with out_frames_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=frame_fields)
        writer.writeheader()
        for row in frame_rows_out:
            writer.writerow({key: row.get(key, "") for key in frame_fields})

    summary_lines = [
        "MICRO OBSERVATION CONTINUITY REFINER V2",
        "=" * 72,
        "source_mode               : MICRO_OBSERVATION_CONTINUITY_V2",
        f"input_frame_slot_rows     : {len(rows)}",
        f"trajectory_count          : {len(trajectory_rows)}",
        f"trajectory_frame_rows     : {len(frame_rows_out)}",
        f"max_gap_frames            : {int(args.max_gap_frames)}",
        f"min_match_score           : {float(args.min_match_score):.3f}",
        "",
        "trajectory_class_counts:",
    ]
    for key, value in class_counter.most_common():
        summary_lines.append(f"  {key}: {value}")
    summary_lines.extend(["", "trajectory_status_counts:"])
    for key, value in status_counter.most_common():
        summary_lines.append(f"  {key}: {value}")
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "stage": "micro_observation_continuity_refiner_v2",
                "source_mode": "MICRO_OBSERVATION_CONTINUITY_V2",
                "inputs": {
                    "frame_slots_csv": args.frame_slots_csv,
                },
                "parameters": {
                    "max_gap_frames": int(args.max_gap_frames),
                    "relative_frequency_tolerance": float(args.relative_frequency_tolerance),
                    "min_match_score": float(args.min_match_score),
                },
                "result": {
                    "input_frame_slot_rows": len(rows),
                    "trajectory_count": len(trajectory_rows),
                    "trajectory_frame_rows": len(frame_rows_out),
                    "trajectory_class_counts": dict(class_counter),
                    "trajectory_status_counts": dict(status_counter),
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
            "trajectory_count": len(trajectory_rows),
        },
    )


if __name__ == "__main__":
    main()
