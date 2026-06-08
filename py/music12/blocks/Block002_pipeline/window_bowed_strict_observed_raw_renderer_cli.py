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
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_wav(path: Path, mono: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.rint(mono * 32767.0), -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(clipped.tobytes())


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render a strict bowed preview using only observed rows from the continuity layer, with raw per-frame energies and no inferred gains."
    )
    ap.add_argument("--resolved-cloud-csv", required=True)
    ap.add_argument("--window-start-sec", type=float, required=True)
    ap.add_argument("--window-end-sec", type=float, required=True)
    ap.add_argument("--sample-rate", type=int, default=44100)
    ap.add_argument("--fft-size", type=int, default=4096)
    ap.add_argument("--hop-size", type=int, default=735)
    ap.add_argument("--gaussian-bin-sigma", type=float, default=1.35)
    ap.add_argument("--target-peak", type=float, default=0.30)
    ap.add_argument("--out-wav", required=True)
    ap.add_argument("--out-frame-cloud-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    args = ap.parse_args()

    rows = _load_csv(Path(args.resolved_cloud_csv))
    observed_rows = [
        row for row in rows
        if str(row.get("continuity_source", "")).strip() == "observed"
        and float(args.window_start_sec) <= _safe_float(row.get("time_sec"), -1.0) <= float(args.window_end_sec)
    ]
    if not observed_rows:
        raise SystemExit("No observed rows found in resolved cloud.")

    frame_indices = sorted({_safe_int(row.get("frame_index"), -1) for row in observed_rows if _safe_int(row.get("frame_index"), -1) >= 0})
    rows_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in observed_rows:
        rows_by_frame[_safe_int(row.get("frame_index"), -1)].append(row)

    duration_sec = max(0.0, float(args.window_end_sec) - float(args.window_start_sec))
    sample_count = max(1, int(round(duration_sec * args.sample_rate)))
    output = np.zeros(sample_count + int(args.fft_size), dtype=np.float32)
    window = np.hanning(int(args.fft_size)).astype(np.float32)
    bin_count = int(args.fft_size) // 2 + 1
    bin_freqs = np.fft.rfftfreq(int(args.fft_size), d=1.0 / float(args.sample_rate))
    phase_acc = np.random.default_rng(0).uniform(0.0, 2.0 * math.pi, size=bin_count).astype(np.float64)

    min_frame = min(frame_indices)
    max_frame = max(frame_indices)
    render_frame_indices = list(range(min_frame, max_frame + 1))

    frame_cloud_rows: list[dict[str, Any]] = []
    harmonic_counts: Counter[int] = Counter()
    for frame_pos, frame_index in enumerate(render_frame_indices):
        frame_rows = rows_by_frame.get(frame_index, [])
        if not frame_rows:
            phase_acc += 2.0 * math.pi * bin_freqs * float(args.hop_size) / float(args.sample_rate)
            continue
        mags = np.zeros(bin_count, dtype=np.float32)
        for row in frame_rows:
            harmonic_index = _safe_int(row.get("harmonic_index"), 0)
            harmonic_counts[harmonic_index] += 1
            freq_hz = _safe_float(row.get("frequency_hz"), 0.0)
            raw_energy = max(0.0, _safe_float(row.get("raw_energy"), 0.0))
            center_bin = freq_hz * float(args.fft_size) / float(args.sample_rate)
            sigma = max(float(args.gaussian_bin_sigma), 0.75)
            left = max(0, int(math.floor(center_bin - 3.0 * sigma)))
            right = min(bin_count - 1, int(math.ceil(center_bin + 3.0 * sigma)))
            if right < left:
                continue
            bin_ids = np.arange(left, right + 1, dtype=np.float32)
            kernel = np.exp(-0.5 * ((bin_ids - center_bin) / sigma) ** 2).astype(np.float32)
            kernel_sum = float(np.sum(kernel))
            if kernel_sum <= 0.0:
                continue
            kernel /= kernel_sum
            mags[left:right + 1] += raw_energy * kernel
            frame_cloud_rows.append(
                {
                    "frame_index": frame_index,
                    "time_sec": _safe_float(row.get("time_sec"), frame_index / FPS60),
                    "harmonic_index": harmonic_index,
                    "probe_index": _safe_int(row.get("probe_index"), -1),
                    "observed_note_token": str(row.get("observed_note_token", "")).strip(),
                    "frequency_hz": freq_hz,
                    "raw_energy": raw_energy,
                }
            )
        phase_acc += 2.0 * math.pi * bin_freqs * float(args.hop_size) / float(args.sample_rate)
        spec = mags.astype(np.complex128) * np.exp(1j * phase_acc)
        frame_signal = np.fft.irfft(spec, n=int(args.fft_size)).astype(np.float32) * window
        start = frame_pos * int(args.hop_size)
        end = start + int(args.fft_size)
        if end > output.size:
            break
        output[start:end] += frame_signal

    audio = output[:sample_count]
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.0 and peak < float(args.target_peak):
        audio = audio / peak * float(args.target_peak)
        peak = float(np.max(np.abs(audio))) if audio.size else peak
    elif peak > 0.95:
        audio = audio / peak * 0.95
        peak = float(np.max(np.abs(audio))) if audio.size else peak

    _write_wav(Path(args.out_wav), audio, int(args.sample_rate))

    with Path(args.out_frame_cloud_csv).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "frame_index",
                "time_sec",
                "harmonic_index",
                "probe_index",
                "observed_note_token",
                "frequency_hz",
                "raw_energy",
            ],
        )
        writer.writeheader()
        writer.writerows(frame_cloud_rows)

    segments = []
    if frame_indices:
        start = prev = frame_indices[0]
        for frame in frame_indices[1:]:
            if frame == prev + 1:
                prev = frame
            else:
                segments.append((start, prev))
                start = prev = frame
        segments.append((start, prev))

    summary_lines = [
        "WINDOW BOWED STRICT OBSERVED RAW RENDERER",
        "=" * 72,
        f"observed_row_count                 : {len(observed_rows)}",
        f"active_frame_count                 : {len(frame_indices)}",
        f"window_duration_sec                : {duration_sec:.6f}",
        f"sample_rate                        : {args.sample_rate}",
        f"fft_size                           : {args.fft_size}",
        f"hop_size                           : {args.hop_size}",
        f"peak_after_render                  : {peak:.6f}",
        "",
        "active_segments:",
    ]
    for start, end in segments:
        summary_lines.append(f"  {start}..{end} len={end - start + 1}")
    summary_lines.extend(["", "harmonic_row_counts:"])
    for harmonic_index in sorted(harmonic_counts):
        summary_lines.append(f"  h{harmonic_index}: {harmonic_counts[harmonic_index]}")
    summary_lines.extend(
        [
            "",
            "interpretation:",
            "  This is a strict observed-only render. It uses only rows marked",
            "  continuity_source=observed, with their real raw_energy and real frequency_hz.",
            "  No held rows, harmonic gains, passport priors or probability upweights are used.",
        ]
    )
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "stage": "window_bowed_strict_observed_raw_renderer",
                "observed_row_count": len(observed_rows),
                "active_frame_count": len(frame_indices),
                "window_duration_sec": duration_sec,
                "sample_rate": args.sample_rate,
                "fft_size": args.fft_size,
                "hop_size": args.hop_size,
                "peak_after_render": peak,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
