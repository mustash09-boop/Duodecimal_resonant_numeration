# -*- coding: ascii -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
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
    last_frequency_hz: float
    last_slot_index: int
    frame_count: int = 0
    note_counter: Counter[str] = field(default_factory=Counter)
    coarse_counter: Counter[str] = field(default_factory=Counter)
    suffix_counter: Counter[str] = field(default_factory=Counter)
    probe_counter: Counter[int] = field(default_factory=Counter)
    slot_counter: Counter[int] = field(default_factory=Counter)
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
        self.last_slot_index = slot_index
        self.last_frequency_hz = frequency_hz
        self.frame_count += 1
        self.note_counter[note_token] += 1
        self.coarse_counter[coarse_note] += 1
        self.suffix_counter[_micro_suffix(note_token)] += 1
        self.probe_counter[probe_index] += 1
        self.slot_counter[slot_index] += 1
        self.frequency_values.append(frequency_hz)
        self.energy_values.append(_safe_float(row.get("energy"), 0.0))
        self.continuation_values.append(_safe_float(row.get("continuation"), 0.0))
        self.rise_values.append(_safe_float(row.get("rise"), 0.0))
        self.micro_path.append(note_token)


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
    slot_index = _safe_int(row.get("slot_index"), 0)
    probe_index = _safe_int(row.get("probe_index"), 0)
    frequency_hz = _safe_float(row.get("frequency_hz"), 0.0)
    continuation = _safe_float(row.get("continuation"), 0.0)
    rise = _safe_float(row.get("rise"), 0.0)

    score = 0.0
    reasons: list[str] = []

    if probe_index == traj.last_probe_index:
        score += 5.0
        reasons.append("same_probe")
    if note_token == traj.last_note_token:
        score += 3.0
        reasons.append("same_note")
    elif coarse_note == traj.last_coarse_note:
        score += 1.2
        reasons.append("same_coarse")
    elif _pitchclass(note_token) and _pitchclass(note_token) == _pitchclass(traj.last_note_token):
        score += 0.5
        reasons.append("same_pc")

    if slot_index == traj.last_slot_index:
        score += 0.35
        reasons.append("same_slot")

    if traj.last_frequency_hz > 0.0 and frequency_hz > 0.0:
        rel_freq = abs(frequency_hz - traj.last_frequency_hz) / max(traj.last_frequency_hz, 1e-9)
        if rel_freq <= rel_freq_tolerance:
            score += 1.5
            reasons.append("tight_freq")
        elif rel_freq <= rel_freq_tolerance * 2.5:
            score += 0.6
            reasons.append("near_freq")

    score += max(0.0, 0.65 - 0.18 * gap)
    score += continuation * 0.30
    score += min(rise, 0.10)
    return score, "|".join(reasons) if reasons else "weak"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build early micro-observation continuity trajectories directly from frame-slot observations before notechain inference."
    )
    ap.add_argument("--frame-slots-csv", required=True)
    ap.add_argument("--out-trajectories-csv", required=True)
    ap.add_argument("--out-trajectory-frames-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--progress-json", default="")
    ap.add_argument("--max-gap-frames", type=int, default=3)
    ap.add_argument("--relative-frequency-tolerance", type=float, default=0.0035)
    ap.add_argument("--min-match-score", type=float, default=3.4)
    args = ap.parse_args()

    rows = _load_csv(Path(args.frame_slots_csv))
    rows.sort(key=lambda row: (_safe_int(row.get("frame_index"), 0), _safe_int(row.get("rank_in_frame"), 0), _safe_int(row.get("probe_index"), 0)))
    total_rows = len(rows)

    _write_progress(
        args.progress_json,
        {
            "status": "running",
            "phase": "building_trajectories",
            "processed_rows": 0,
            "total_rows": total_rows,
            "trajectory_count": 0,
        },
    )

    active: list[ObservationTrajectory] = []
    finalized: list[ObservationTrajectory] = []
    next_id = 1
    frame_rows_out: list[dict[str, Any]] = []

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

    for idx, row in enumerate(rows, start=1):
        frame_index = _safe_int(row.get("frame_index"), 0)
        _finalize_stale(frame_index)

        best_traj: ObservationTrajectory | None = None
        best_score = -1.0
        best_reason = ""
        for traj in active:
            score, reason = _match_score(
                traj,
                row,
                max_gap_frames=int(args.max_gap_frames),
                rel_freq_tolerance=float(args.relative_frequency_tolerance),
            )
            if score > best_score:
                best_score = score
                best_traj = traj
                best_reason = reason

        if best_traj is not None and best_score >= float(args.min_match_score):
            traj = best_traj
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
                last_frequency_hz=_safe_float(row.get("frequency_hz"), 0.0),
                last_slot_index=_safe_int(row.get("slot_index"), 0),
            )
            next_id += 1
            active.append(traj)
            best_reason = "new_trajectory"

        traj.add(row)
        frame_rows_out.append(
            {
                "trajectory_id": traj.trajectory_id,
                "frame_index": frame_index,
                "time_sec": row.get("time_sec", ""),
                "event_id": _safe_int(row.get("event_id"), 0),
                "slot_index": _safe_int(row.get("slot_index"), 0),
                "probe_index": _safe_int(row.get("probe_index"), 0),
                "note_token": str(row.get("note_token", "")).strip(),
                "coarse_note_overlay": str(row.get("coarse_note_overlay", "")).strip(),
                "micro_suffix": str(row.get("micro_suffix", "")).strip(),
                "frequency_hz": row.get("frequency_hz", ""),
                "energy": row.get("energy", ""),
                "rise": row.get("rise", ""),
                "continuation": row.get("continuation", ""),
                "link_reason": best_reason,
            }
        )

        if idx % 2048 == 0 or idx == total_rows:
            _write_progress(
                args.progress_json,
                {
                    "status": "running",
                    "phase": "building_trajectories",
                    "processed_rows": idx,
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

        if probe_share >= 0.75 and micro_share >= 0.55:
            traj_class = "EXACT_PROBE_TRAJECTORY"
        elif micro_share >= 0.65:
            traj_class = "EXACT_MICRO_TRAJECTORY"
        elif anchor_coarse and len(traj.coarse_counter) == 1 and len(traj.suffix_counter) >= 2:
            traj_class = "COARSE_MICRO_DRIFT_TRAJECTORY"
        else:
            traj_class = "LOCAL_OBSERVATION_TRAJECTORY"

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
                "source_mode": "MICRO_OBSERVATION_CONTINUITY",
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
                "primary_micro_share": f"{micro_share:.9f}",
                "primary_probe_share": f"{probe_share:.9f}",
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
        "primary_micro_share",
        "primary_probe_share",
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
        "micro_path_json",
    ]
    frame_fields = [
        "trajectory_id",
        "frame_index",
        "time_sec",
        "event_id",
        "slot_index",
        "probe_index",
        "note_token",
        "coarse_note_overlay",
        "micro_suffix",
        "frequency_hz",
        "energy",
        "rise",
        "continuation",
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
        "MICRO OBSERVATION CONTINUITY REFINER",
        "=" * 72,
        "source_mode               : MICRO_OBSERVATION_CONTINUITY",
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
                "stage": "micro_observation_continuity_refiner",
                "source_mode": "MICRO_OBSERVATION_CONTINUITY",
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
