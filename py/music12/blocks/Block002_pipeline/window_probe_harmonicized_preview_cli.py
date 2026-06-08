# -*- coding: ascii -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import wave
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


FPS60 = 60.0
FRAME_STEP_SEC = 1.0 / FPS60
WINDOW_SEC = 0.05
MIN_TAIL_DECAY_RATIO = 0.015625


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


def _write_wav(path: Path, mono: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.rint(mono * 32767.0), -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(clipped.tobytes())


def _build_window_envelope(sample_count: int) -> np.ndarray:
    if sample_count <= 1:
        return np.ones(max(1, sample_count), dtype=np.float32)
    return np.hanning(sample_count).astype(np.float32)


def _owner_match(owner_label: str, target_owner: str) -> bool:
    return str(owner_label).strip() == str(target_owner).strip()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render a harmonicized owner preview with short-gap bridging between active frames."
    )
    ap.add_argument("--ownership-observations-csv", required=True)
    ap.add_argument("--window-start-sec", type=float, required=True)
    ap.add_argument("--window-end-sec", type=float, required=True)
    ap.add_argument("--target-owner", required=True)
    ap.add_argument("--sample-rate", type=int, default=44100)
    ap.add_argument("--amplitude-scale", type=float, default=0.18)
    ap.add_argument("--max-gap-frames", type=int, default=2)
    ap.add_argument("--max-hold-gap-frames", type=int, default=0)
    ap.add_argument("--tail-extend-to-window-end", action="store_true")
    ap.add_argument("--include-suboctave-root", action="store_true")
    ap.add_argument("--suboctave-root-weight", type=float, default=0.0)
    ap.add_argument("--out-wav", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--progress-json", default="")
    args = ap.parse_args()

    harmonic_multipliers = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0]
    harmonic_weights = [1.0, 0.48, 0.31, 0.21, 0.24, 0.20]

    _write_progress(
        args.progress_json,
        {
            "status": "running",
            "phase": "loading_inputs",
            "target_owner": args.target_owner,
        },
    )

    rows = _load_csv(Path(args.ownership_observations_csv))
    selected = [
        row for row in rows
        if _owner_match(row.get("owner_label", ""), args.target_owner)
        and args.window_start_sec <= _safe_float(row.get("time_sec"), -1.0) <= args.window_end_sec
    ]

    frame_data: dict[int, dict[str, Any]] = {}
    coarse_counter: Counter[str] = Counter()
    for row in selected:
        frame_index = _safe_int(row.get("frame_index"), 0)
        energy = _safe_float(row.get("energy"), 0.0)
        freq = _safe_float(row.get("frequency_hz"), 0.0)
        coarse = str(row.get("observed_coarse_symbol", "")).strip()
        if frame_index not in frame_data:
            frame_data[frame_index] = {
                "energy_sum": 0.0,
                "freq_weighted_sum": 0.0,
                "obs_count": 0,
                "time_sec": _safe_float(row.get("time_sec"), 0.0),
            }
        frame_data[frame_index]["energy_sum"] += energy
        frame_data[frame_index]["freq_weighted_sum"] += energy * freq
        frame_data[frame_index]["obs_count"] += 1
        coarse_counter[coarse] += 1

    active_frames = sorted(frame_data.keys())
    bridged_frames = 0
    held_gap_frames = 0
    for left, right in zip(active_frames, active_frames[1:]):
        gap = right - left - 1
        if gap <= 0 or gap > args.max_gap_frames:
            continue
        left_data = frame_data[left]
        right_data = frame_data[right]
        left_freq = left_data["freq_weighted_sum"] / max(left_data["energy_sum"], 1e-9)
        right_freq = right_data["freq_weighted_sum"] / max(right_data["energy_sum"], 1e-9)
        left_energy = left_data["energy_sum"]
        right_energy = right_data["energy_sum"]
        left_time = left_data["time_sec"]
        right_time = right_data["time_sec"]
        for offset in range(1, gap + 1):
            alpha = float(offset) / float(gap + 1)
            frame_index = left + offset
            frame_data[frame_index] = {
                "energy_sum": (1.0 - alpha) * left_energy + alpha * right_energy,
                "freq_weighted_sum": ((1.0 - alpha) * left_freq + alpha * right_freq),
                "obs_count": 0,
                "time_sec": (1.0 - alpha) * left_time + alpha * right_time,
                "bridged": True,
            }
            bridged_frames += 1

    active_frames = sorted(frame_data.keys())
    held_segment_count = 0
    if args.max_hold_gap_frames > max(args.max_gap_frames, 0):
        for left, right in zip(active_frames, active_frames[1:]):
            gap = right - left - 1
            if gap <= max(args.max_gap_frames, 0) or gap > args.max_hold_gap_frames:
                continue
            left_data = frame_data[left]
            right_data = frame_data[right]
            left_freq = left_data["freq_weighted_sum"] / max(left_data["energy_sum"], 1e-9)
            right_freq = right_data["freq_weighted_sum"] / max(right_data["energy_sum"], 1e-9)
            left_energy = left_data["energy_sum"]
            right_energy = right_data["energy_sum"]
            left_time = left_data["time_sec"]
            right_time = right_data["time_sec"]
            for offset in range(1, gap + 1):
                alpha = float(offset) / float(gap + 1)
                decay_shape = 1.0 - 0.22 * math.sin(math.pi * alpha)
                frame_index = left + offset
                interp_energy = ((1.0 - alpha) * left_energy + alpha * right_energy) * decay_shape
                interp_freq = (1.0 - alpha) * left_freq + alpha * right_freq
                frame_data[frame_index] = {
                    "energy_sum": interp_energy,
                    "freq_weighted_sum": interp_freq,
                    "obs_count": 0,
                    "time_sec": (1.0 - alpha) * left_time + alpha * right_time,
                    "held": True,
                }
                held_gap_frames += 1
            held_segment_count += 1

    active_frames = sorted(frame_data.keys())
    tail_extended_frames = 0
    if args.tail_extend_to_window_end and active_frames:
        last_frame = active_frames[-1]
        last_data = frame_data[last_frame]
        last_freq = last_data["freq_weighted_sum"] / max(last_data["energy_sum"], 1e-9)
        last_energy = last_data["energy_sum"]
        last_time = last_data["time_sec"]
        window_end_frame = int(math.ceil(args.window_end_sec * FPS60))
        for frame_index in range(last_frame + 1, window_end_frame + 1):
            delta = frame_index - last_frame
            decay = math.exp(-0.18 * delta)
            if decay < MIN_TAIL_DECAY_RATIO:
                break
            frame_data[frame_index] = {
                "energy_sum": last_energy * decay,
                "freq_weighted_sum": last_freq,
                "obs_count": 0,
                "time_sec": last_time + delta * FRAME_STEP_SEC,
                "tail": True,
            }
            tail_extended_frames += 1

    duration_sec = max(0.0, float(args.window_end_sec) - float(args.window_start_sec))
    sample_count = max(1, int(round(duration_sec * args.sample_rate)))
    audio = np.zeros(sample_count, dtype=np.float32)
    window_samples = max(2, int(round(WINDOW_SEC * args.sample_rate)))
    envelope = _build_window_envelope(window_samples)
    phase_state = {(harm, idx): 0.0 for harm in range(len(harmonic_multipliers)) for idx in range(1)}

    _write_progress(
        args.progress_json,
        {
            "status": "running",
            "phase": "rendering_harmonicized_preview",
            "selected_observations": len(selected),
            "active_frames_original": len(active_frames),
            "active_frames_with_bridges": len(frame_data),
        },
    )

    for frame_index in sorted(frame_data.keys()):
        info = frame_data[frame_index]
        time_sec = float(info["time_sec"])
        energy_sum = float(info["energy_sum"])
        if energy_sum <= 0.0:
            continue
        if "tail" in info:
            centroid_hz = float(info["freq_weighted_sum"])
            amp_scale = 0.52
        elif "held" in info:
            centroid_hz = float(info["freq_weighted_sum"])
            amp_scale = 0.68
        elif "bridged" in info:
            centroid_hz = float(info["freq_weighted_sum"])
            amp_scale = 0.72
        else:
            centroid_hz = float(info["freq_weighted_sum"]) / max(energy_sum, 1e-9)
            amp_scale = 1.0
        start_index = int(round((time_sec - args.window_start_sec) * args.sample_rate))
        if start_index >= sample_count or start_index + window_samples <= 0:
            continue
        left = max(0, start_index)
        right = min(sample_count, start_index + window_samples)
        if right <= left:
            continue
        local_left = left - start_index
        local_right = local_left + (right - left)
        t = np.arange(right - left, dtype=np.float32) / float(args.sample_rate)
        amp = float(args.amplitude_scale) * math.sqrt(max(0.0, energy_sum / 10.0)) * amp_scale
        frame_chunk = np.zeros(right - left, dtype=np.float32)
        if args.include_suboctave_root and args.suboctave_root_weight > 0.0:
            root_hz = centroid_hz * 0.5
            phase_key = ("suboctave", 0)
            phase0 = phase_state.get(phase_key, 0.0)
            frame_chunk += (amp * float(args.suboctave_root_weight)) * np.sin(
                (2.0 * math.pi * root_hz * t) + phase0
            )
            phase_advance = 2.0 * math.pi * root_hz * (right - left) / float(args.sample_rate)
            phase_state[phase_key] = float((phase0 + phase_advance) % (2.0 * math.pi))
        for harm_idx, (mult, weight) in enumerate(zip(harmonic_multipliers, harmonic_weights)):
            freq_hz = centroid_hz * mult
            phase_key = (harm_idx, 0)
            phase0 = phase_state[phase_key]
            frame_chunk += (amp * weight) * np.sin((2.0 * math.pi * freq_hz * t) + phase0)
            phase_advance = 2.0 * math.pi * freq_hz * (right - left) / float(args.sample_rate)
            phase_state[phase_key] = float((phase0 + phase_advance) % (2.0 * math.pi))
        frame_chunk *= envelope[local_left:local_right]
        audio[left:right] += frame_chunk

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.999:
        audio = audio / peak * 0.98
        normalization = "peak_limited"
    else:
        normalization = "not_needed"

    _write_wav(Path(args.out_wav), audio, args.sample_rate)

    summary_lines = [
        "WINDOW PROBE HARMONICIZED PREVIEW",
        "=" * 72,
        f"target_owner                : {args.target_owner}",
        f"window_start_sec            : {args.window_start_sec:.6f}",
        f"window_end_sec              : {args.window_end_sec:.6f}",
        f"selected_observations       : {len(selected)}",
        f"active_frames_original      : {len(active_frames)}",
        f"bridged_frames              : {bridged_frames}",
        f"held_gap_frames             : {held_gap_frames}",
        f"held_segment_count          : {held_segment_count}",
        f"tail_extended_frames        : {tail_extended_frames}",
        f"active_frames_with_bridges  : {len(frame_data)}",
        f"sample_rate                 : {args.sample_rate}",
        f"window_samples              : {window_samples}",
        f"amplitude_scale             : {args.amplitude_scale:.6f}",
        f"max_gap_frames              : {args.max_gap_frames}",
        f"max_hold_gap_frames         : {args.max_hold_gap_frames}",
        f"tail_extend_to_window_end   : {args.tail_extend_to_window_end}",
        f"include_suboctave_root      : {args.include_suboctave_root}",
        f"suboctave_root_weight       : {args.suboctave_root_weight:.6f}",
        f"peak_abs_before_limit       : {peak:.6f}",
        f"normalization               : {normalization}",
        "",
        "harmonic_stack:",
    ]
    for mult, weight in zip(harmonic_multipliers, harmonic_weights):
        summary_lines.append(f"  x{mult:.1f}: {weight:.3f}")
    summary_lines.extend(["", "top_coarse_symbols:"])
    for token, count in coarse_counter.most_common(16):
        summary_lines.append(f"  {token}: {count}")
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "stage": "window_probe_harmonicized_preview",
                "inputs": {
                    "ownership_observations_csv": args.ownership_observations_csv,
                },
                "window": {
                    "start_sec": args.window_start_sec,
                    "end_sec": args.window_end_sec,
                },
                "render": {
                    "target_owner": args.target_owner,
                    "sample_rate": args.sample_rate,
                    "window_sec": WINDOW_SEC,
                    "harmonic_multipliers": harmonic_multipliers,
                    "harmonic_weights": harmonic_weights,
                    "max_gap_frames": args.max_gap_frames,
                    "max_hold_gap_frames": args.max_hold_gap_frames,
                    "tail_extend_to_window_end": bool(args.tail_extend_to_window_end),
                    "include_suboctave_root": bool(args.include_suboctave_root),
                    "suboctave_root_weight": float(args.suboctave_root_weight),
                    "selected_observations": len(selected),
                    "active_frames_original": len(active_frames),
                    "bridged_frames": bridged_frames,
                    "held_gap_frames": held_gap_frames,
                    "held_segment_count": held_segment_count,
                    "tail_extended_frames": tail_extended_frames,
                    "active_frames_with_bridges": len(frame_data),
                    "peak_abs_before_limit": peak,
                    "normalization": normalization,
                    "top_coarse_symbols": coarse_counter.most_common(16),
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
            "selected_observations": len(selected),
            "active_frames_original": len(active_frames),
            "bridged_frames": bridged_frames,
            "held_gap_frames": held_gap_frames,
            "held_segment_count": held_segment_count,
            "tail_extended_frames": tail_extended_frames,
            "active_frames_with_bridges": len(frame_data),
        },
    )


if __name__ == "__main__":
    main()
