# -*- coding: ascii -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


FPS60 = 60.0
FRAME_STEP_SEC = 1.0 / FPS60
WINDOW_SEC = 0.05


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


def _owner_match(owner_label: str, target_owner: str) -> bool:
    return str(owner_label).strip() == str(target_owner).strip()


def _build_window_envelope(sample_count: int) -> np.ndarray:
    if sample_count <= 1:
        return np.ones(max(1, sample_count), dtype=np.float32)
    return np.hanning(sample_count).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render a probe-owned sinebank preview for a local window from ownership-tagged observations."
    )
    ap.add_argument("--ownership-observations-csv", required=True)
    ap.add_argument("--window-start-sec", type=float, required=True)
    ap.add_argument("--window-end-sec", type=float, required=True)
    ap.add_argument("--target-owner", required=True)
    ap.add_argument("--sample-rate", type=int, default=44100)
    ap.add_argument("--amplitude-scale", type=float, default=0.2)
    ap.add_argument("--out-wav", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--progress-json", default="")
    args = ap.parse_args()

    _write_progress(
        args.progress_json,
        {
            "status": "running",
            "phase": "loading_inputs",
            "target_owner": args.target_owner,
            "window_start_sec": args.window_start_sec,
            "window_end_sec": args.window_end_sec,
        },
    )

    rows = _load_csv(Path(args.ownership_observations_csv))
    selected = [
        row for row in rows
        if _owner_match(row.get("owner_label", ""), args.target_owner)
        and args.window_start_sec <= _safe_float(row.get("time_sec"), -1.0) <= args.window_end_sec
    ]

    duration_sec = max(0.0, float(args.window_end_sec) - float(args.window_start_sec))
    sample_count = max(1, int(round(duration_sec * args.sample_rate)))
    audio = np.zeros(sample_count, dtype=np.float32)

    window_samples = max(2, int(round(WINDOW_SEC * args.sample_rate)))
    envelope = _build_window_envelope(window_samples)
    phase_state: dict[int, float] = defaultdict(float)
    frame_counter: Counter[int] = Counter()
    coarse_counter: Counter[str] = Counter()

    _write_progress(
        args.progress_json,
        {
            "status": "running",
            "phase": "rendering_sinebank",
            "selected_observations": len(selected),
            "sample_count": sample_count,
        },
    )

    for row in selected:
        time_sec = _safe_float(row.get("time_sec"), 0.0)
        probe_index = _safe_int(row.get("probe_index"), 0)
        freq_hz = _safe_float(row.get("frequency_hz"), 0.0)
        energy = _safe_float(row.get("energy"), 0.0)
        if freq_hz <= 0.0 or energy <= 0.0:
            continue
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
        phase0 = phase_state[probe_index]
        amp = float(args.amplitude_scale) * math.sqrt(max(0.0, energy))
        chunk = amp * np.sin((2.0 * math.pi * freq_hz * t) + phase0) * envelope[local_left:local_right]
        audio[left:right] += chunk.astype(np.float32)
        phase_advance = (2.0 * math.pi * freq_hz * (right - left) / float(args.sample_rate))
        phase_state[probe_index] = float((phase0 + phase_advance) % (2.0 * math.pi))
        frame_counter[_safe_int(row.get("frame_index"), 0)] += 1
        coarse_counter[str(row.get("observed_coarse_symbol", "")).strip()] += 1

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.999:
        audio = audio / peak * 0.98
        normalization = "peak_limited"
    else:
        normalization = "not_needed"

    _write_wav(Path(args.out_wav), audio, args.sample_rate)

    summary_lines = [
        "WINDOW PROBE SINEBANK PREVIEW",
        "=" * 72,
        f"target_owner                : {args.target_owner}",
        f"window_start_sec            : {args.window_start_sec:.6f}",
        f"window_end_sec              : {args.window_end_sec:.6f}",
        f"selected_observations       : {len(selected)}",
        f"active_frame_count          : {len(frame_counter)}",
        f"sample_rate                 : {args.sample_rate}",
        f"window_samples              : {window_samples}",
        f"amplitude_scale             : {args.amplitude_scale:.6f}",
        f"peak_abs_before_limit       : {peak:.6f}",
        f"normalization               : {normalization}",
        "",
        "top_coarse_symbols:",
    ]
    for token, count in coarse_counter.most_common(16):
        summary_lines.append(f"  {token}: {count}")
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "stage": "window_probe_sinebank_preview",
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
                    "frame_step_sec": FRAME_STEP_SEC,
                    "amplitude_scale": args.amplitude_scale,
                    "selected_observations": len(selected),
                    "active_frame_count": len(frame_counter),
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
            "active_frame_count": len(frame_counter),
        },
    )


if __name__ == "__main__":
    main()
