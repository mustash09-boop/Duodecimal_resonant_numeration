# -*- coding: ascii -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from resonance_candidate_inference_core import load_coords_csv, load_matrix_csv_memmap


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


def _load_wav_mono_float32(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sampwidth = wf.getsampwidth()
        frame_count = wf.getnframes()
        raw = wf.readframes(frame_count)
    if sampwidth != 2:
        raise ValueError(f"Only 16-bit PCM wav is supported: {path}")
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return pcm.astype(np.float32), int(sample_rate)


def _build_average_spectrum(samples: np.ndarray, fft_size: int, hop: int) -> np.ndarray:
    if samples.size == 0:
        return np.zeros(fft_size // 2 + 1, dtype=np.float32)
    if samples.size < fft_size:
        padded = np.zeros(fft_size, dtype=np.float32)
        padded[:samples.size] = samples
        samples = padded
    window = np.hanning(fft_size).astype(np.float32)
    acc = np.zeros(fft_size // 2 + 1, dtype=np.float64)
    count = 0
    for start in range(0, samples.size - fft_size + 1, max(1, hop)):
        chunk = samples[start:start + fft_size]
        spec = np.fft.rfft(chunk * window)
        acc += np.abs(spec)
        count += 1
    if count == 0:
        spec = np.fft.rfft(samples[:fft_size] * window)
        acc += np.abs(spec)
        count = 1
    return (acc / float(count)).astype(np.float32)


def _sample_spectrum_magnitude(mean_mag: np.ndarray, sample_rate: int, fft_size: int, frequency_hz: float) -> float:
    if mean_mag.size == 0:
        return 0.0
    bin_freqs = np.fft.rfftfreq(fft_size, d=1.0 / float(sample_rate))
    target = min(max(0.0, float(frequency_hz)), 0.5 * float(sample_rate))
    return _safe_float(np.interp(target, bin_freqs, mean_mag), 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render a bowed micro-cloud from selected harmonic families, scanning the full window directly in the raw matrix to recover missing continuity."
    )
    ap.add_argument("--selected-families-csv", required=True)
    ap.add_argument("--probe-coords-csv", required=True)
    ap.add_argument("--probe-matrix-csv", required=True)
    ap.add_argument("--probe-times-csv", required=True)
    ap.add_argument("--source-wav", required=True)
    ap.add_argument("--window-start-sec", type=float, required=True)
    ap.add_argument("--window-end-sec", type=float, required=True)
    ap.add_argument("--sample-rate", type=int, default=44100)
    ap.add_argument("--fft-size", type=int, default=4096)
    ap.add_argument("--hop-size", type=int, default=735)
    ap.add_argument("--gaussian-bin-sigma", type=float, default=1.35)
    ap.add_argument("--amplitude-scale", type=float, default=0.07)
    ap.add_argument("--target-peak", type=float, default=0.30)
    ap.add_argument("--continuation-ratio", type=float, default=0.18)
    ap.add_argument("--frame-p95-activation-ratio", type=float, default=0.045)
    ap.add_argument("--out-wav", required=True)
    ap.add_argument("--out-frame-clouds-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--progress-json", default="")
    args = ap.parse_args()

    _write_progress(args.progress_json, {"status": "running", "phase": "loading_inputs"})

    selected_rows = _load_csv(Path(args.selected_families_csv))
    coords = load_coords_csv(args.probe_coords_csv)
    coord_by_probe = {int(coord.probe_index): coord for coord in coords}
    matrix, _info = load_matrix_csv_memmap(args.probe_matrix_csv)
    time_rows = _load_csv(Path(args.probe_times_csv))
    frame_time_map = {
        _safe_int(row.get("frame_index"), -1): _safe_float(row.get("time_seconds"), 0.0)
        for row in time_rows
    }
    source_samples, source_sr = _load_wav_mono_float32(Path(args.source_wav))
    source_spec = _build_average_spectrum(source_samples, int(args.fft_size), max(1, int(args.fft_size // 4)))
    source_peak = max(_safe_float(np.max(source_spec), 0.0), 1e-9)

    families_by_h: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        h = _safe_int(row.get("harmonic_index"), 0)
        p = _safe_int(row.get("probe_index"), -1)
        if h <= 0 or p < 0 or p not in coord_by_probe:
            continue
        families_by_h[h].append(row)

    start_frame = int(math.floor(float(args.window_start_sec) * FPS60))
    end_frame = int(math.ceil(float(args.window_end_sec) * FPS60))
    frame_indices = list(range(start_frame, end_frame + 1))
    frame_times = np.asarray([frame_time_map.get(idx, idx / FPS60) for idx in frame_indices], dtype=np.float64)
    frame_p95 = np.quantile(np.asarray(matrix[:, start_frame:end_frame + 1], dtype=np.float32), 0.95, axis=0).astype(np.float32)

    _write_progress(
        args.progress_json,
        {
            "status": "running",
            "phase": "scanning_full_window",
            "selected_harmonics": len(families_by_h),
            "frame_count": len(frame_indices),
        },
    )

    cloud_rows: list[dict[str, Any]] = []
    harmonic_scale: dict[int, float] = {}
    mean_family_energy_by_h: dict[int, float] = {}

    for harmonic_index in sorted(families_by_h):
        rows = families_by_h[harmonic_index]
        probe_indices = [_safe_int(row.get("probe_index"), -1) for row in rows]
        probe_freqs = np.asarray([float(coord_by_probe[probe_idx].frequency_hz) for probe_idx in probe_indices], dtype=np.float32)
        family_mean_energies = np.asarray([_safe_float(row.get("mean_energy_over_frame_p95"), 0.0) for row in rows], dtype=np.float32)
        family_thresholds = family_mean_energies * float(args.continuation_ratio)
        band = np.asarray(matrix[probe_indices, start_frame:end_frame + 1], dtype=np.float32)

        mean_freq = float(np.mean(probe_freqs)) if probe_freqs.size else 0.0
        mean_family_energy = max(1e-9, float(np.mean(family_mean_energies)))
        mean_family_energy_by_h[harmonic_index] = mean_family_energy
        spectral_ref = _sample_spectrum_magnitude(source_spec, source_sr, int(args.fft_size), mean_freq)
        harmonic_scale[harmonic_index] = float(args.amplitude_scale) * spectral_ref / source_peak / mean_family_energy

        for local_frame_pos, frame_index in enumerate(frame_indices):
            local_rows = []
            local_p95 = max(_safe_float(frame_p95[local_frame_pos], 0.0), 1e-9)
            for rel_idx, probe_index in enumerate(probe_indices):
                energy = float(band[rel_idx, local_frame_pos])
                if energy < float(family_thresholds[rel_idx]):
                    continue
                if energy < local_p95 * float(args.frame_p95_activation_ration if False else args.frame_p95_activation_ratio):
                    continue
                local_rows.append(
                    {
                        "frame_index": frame_index,
                        "time_sec": frame_time_map.get(frame_index, frame_index / FPS60),
                        "harmonic_index": harmonic_index,
                        "probe_index": probe_index,
                        "observed_note_token": str(rows[rel_idx].get("observed_note_token", "")).strip(),
                        "frequency_hz": float(probe_freqs[rel_idx]),
                        "energy_over_frame_p95": energy / local_p95,
                        "raw_energy": energy,
                    }
                )
            cloud_rows.extend(local_rows)

    cloud_rows.sort(key=lambda row: (_safe_int(row.get("frame_index"), 0), _safe_int(row.get("harmonic_index"), 0), _safe_float(row.get("frequency_hz"), 0.0)))

    duration_sec = max(0.0, float(args.window_end_sec) - float(args.window_start_sec))
    sample_count = max(1, int(round(duration_sec * args.sample_rate)))
    output = np.zeros(sample_count + int(args.fft_size), dtype=np.float32)
    window = np.hanning(int(args.fft_size)).astype(np.float32)
    bin_count = int(args.fft_size) // 2 + 1
    bin_freqs = np.fft.rfftfreq(int(args.fft_size), d=1.0 / float(args.sample_rate))
    phase_acc = np.random.default_rng(0).uniform(0.0, 2.0 * math.pi, size=bin_count).astype(np.float64)

    rows_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in cloud_rows:
        rows_by_frame[_safe_int(row.get("frame_index"), -1)].append(row)

    for frame_pos, frame_index in enumerate(frame_indices):
        rows = rows_by_frame.get(frame_index, [])
        if not rows:
            continue
        mags = np.zeros(bin_count, dtype=np.float32)
        for row in rows:
            harmonic_index = _safe_int(row.get("harmonic_index"), 0)
            freq_hz = _safe_float(row.get("frequency_hz"), 0.0)
            energy = _safe_float(row.get("energy_over_frame_p95"), 0.0)
            row_mag = energy * harmonic_scale.get(harmonic_index, 0.0)
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
            mags[left:right + 1] += row_mag * kernel

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

    with Path(args.out_frame_clouds_csv).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "frame_index",
                "time_sec",
                "harmonic_index",
                "probe_index",
                "observed_note_token",
                "frequency_hz",
                "energy_over_frame_p95",
                "raw_energy",
            ],
        )
        writer.writeheader()
        writer.writerows(cloud_rows)

    frame_counts = defaultdict(int)
    for row in cloud_rows:
        frame_counts[_safe_int(row.get("frame_index"), -1)] += 1
    active_frames = sorted(fi for fi, count in frame_counts.items() if count > 0)
    gaps = []
    for left, right in zip(active_frames[:-1], active_frames[1:]):
        if right > left + 1:
            gaps.append({"gap_start_frame": left + 1, "gap_end_frame": right - 1, "gap_len": right - left - 1})

    summary_lines = [
        "WINDOW BOWED BUNDLE MICROCLOUD FULLWINDOW PREVIEW",
        "=" * 72,
        f"selected_harmonic_count            : {len(families_by_h)}",
        f"cloud_row_count                    : {len(cloud_rows)}",
        f"active_frame_count                 : {len(active_frames)}",
        f"window_duration_sec                : {duration_sec:.6f}",
        f"sample_rate                        : {args.sample_rate}",
        f"fft_size                           : {args.fft_size}",
        f"hop_size                           : {args.hop_size}",
        f"peak_after_render                  : {peak:.6f}",
        f"gap_count                          : {len(gaps)}",
        "",
        "harmonic_scales:",
    ]
    for harmonic_index in sorted(families_by_h):
        summary_lines.append(
            "  "
            f"h{harmonic_index} families={len(families_by_h[harmonic_index])} "
            f"mean_energy={mean_family_energy_by_h[harmonic_index]:.6f} "
            f"scale={harmonic_scale[harmonic_index]:.9f}"
        )
    if gaps:
        summary_lines.extend(["", "remaining_gaps:"])
        for gap in gaps[:20]:
            summary_lines.append(
                "  "
                f"{gap['gap_start_frame']}..{gap['gap_end_frame']} "
                f"len={gap['gap_len']}"
            )
    summary_lines.extend(
        [
            "",
            "interpretation:",
            "  This preview scans the full raw matrix across the whole window for the",
            "  already selected bowed probe families, so missing continuity is recovered",
            "  from real framewise energy rather than from sparse preselected owner rows.",
        ]
    )
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "stage": "window_bowed_bundle_microcloud_fullwindow_preview",
                "selected_harmonic_count": len(families_by_h),
                "cloud_row_count": len(cloud_rows),
                "active_frame_count": len(active_frames),
                "window_duration_sec": duration_sec,
                "sample_rate": args.sample_rate,
                "fft_size": args.fft_size,
                "hop_size": args.hop_size,
                "gap_count": len(gaps),
                "peak_after_render": peak,
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
            "cloud_row_count": len(cloud_rows),
            "active_frame_count": len(active_frames),
            "gap_count": len(gaps),
        },
    )


if __name__ == "__main__":
    main()
