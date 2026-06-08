# -*- coding: ascii -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import wave
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


def _build_average_spectrum(samples: np.ndarray, fft_size: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    if samples.size == 0:
        return np.zeros(fft_size // 2 + 1, dtype=np.float32), np.linspace(0.0, 0.5, fft_size // 2 + 1, dtype=np.float32)
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
    mean_mag = (acc / float(count)).astype(np.float32)
    freq_axis = np.fft.rfftfreq(fft_size, d=1.0).astype(np.float32)
    return mean_mag, freq_axis


def _sample_spectrum_magnitude(mean_mag: np.ndarray, freq_axis: np.ndarray, sample_rate: int, frequency_hz: float) -> float:
    if mean_mag.size == 0 or freq_axis.size == 0:
        return 0.0
    nyquist = 0.5 * float(sample_rate)
    target = min(max(0.0, float(frequency_hz)), nyquist)
    bin_positions = freq_axis * float(sample_rate)
    return _safe_float(np.interp(target, bin_positions, mean_mag), 0.0)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _moving_average(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or values.size == 0:
        return values.astype(np.float32, copy=True)
    kernel = np.ones(radius * 2 + 1, dtype=np.float32)
    kernel /= float(kernel.size)
    padded = np.pad(values.astype(np.float32), (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _bridge_small_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    if max_gap <= 0 or mask.size == 0:
        return mask
    out = mask.copy()
    active = np.flatnonzero(mask)
    if active.size < 2:
        return out
    for left, right in zip(active[:-1], active[1:]):
        gap = int(right) - int(left) - 1
        if 0 < gap <= max_gap:
            out[left + 1:right] = True
    return out


def _harmonic_gain(harmonic_index: int) -> float:
    harmonic_gain_map = {
        1: 1.00,
        2: 0.92,
        3: 0.66,
        4: 0.40,
        5: 0.52,
        6: 0.31,
        7: 0.18,
    }
    if harmonic_index in harmonic_gain_map:
        return harmonic_gain_map[harmonic_index]
    return 0.18 * (7.0 / float(harmonic_index)) ** 0.72


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render a bowed bundle preview from one connected trajectory per harmonic, preserving per-harmonic microshifts."
    )
    ap.add_argument("--selected-families-csv", required=True)
    ap.add_argument("--probe-coords-csv", required=True)
    ap.add_argument("--probe-matrix-csv", required=True)
    ap.add_argument("--probe-times-csv", required=True)
    ap.add_argument("--window-start-sec", type=float, required=True)
    ap.add_argument("--window-end-sec", type=float, required=True)
    ap.add_argument("--sample-rate", type=int, default=44100)
    ap.add_argument("--amplitude-scale", type=float, default=0.085)
    ap.add_argument(
        "--amplitude-mode",
        choices=("legacy", "real", "spectral"),
        default="real",
        help="Use legacy leveled harmonic gains, probe-real amplitudes, or amplitudes calibrated to the real WAV spectrum.",
    )
    ap.add_argument("--source-wav", default="")
    ap.add_argument("--spectrum-fft-size", type=int, default=16384)
    ap.add_argument("--spectrum-hop", type=int, default=4096)
    ap.add_argument("--base-threshold-ratio", type=float, default=0.12)
    ap.add_argument("--gap-bridge-frames", type=int, default=2)
    ap.add_argument("--frame-smooth-radius", type=int, default=2)
    ap.add_argument("--out-wav", required=True)
    ap.add_argument("--out-harmonic-summary-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--progress-json", default="")
    args = ap.parse_args()

    _write_progress(args.progress_json, {"status": "running", "phase": "loading_inputs"})

    selected_rows = _load_csv(Path(args.selected_families_csv))
    coords = load_coords_csv(args.probe_coords_csv)
    matrix, _info = load_matrix_csv_memmap(args.probe_matrix_csv)
    time_rows = _load_csv(Path(args.probe_times_csv))
    spectral_mag = np.zeros(0, dtype=np.float32)
    spectral_freq_axis = np.zeros(0, dtype=np.float32)
    source_wav_peak = 1.0
    source_wav_sr = args.sample_rate
    if args.amplitude_mode == "spectral":
        if not str(args.source_wav).strip():
            raise ValueError("--source-wav is required for --amplitude-mode spectral")
        source_wav_samples, source_wav_sr = _load_wav_mono_float32(Path(args.source_wav))
        spectral_mag, spectral_freq_axis = _build_average_spectrum(
            source_wav_samples,
            int(args.spectrum_fft_size),
            int(args.spectrum_hop),
        )
        source_wav_peak = max(_safe_float(np.max(spectral_mag), 0.0), 1e-9)

    coord_by_probe = {int(coord.probe_index): coord for coord in coords}
    time_by_frame = {
        _safe_int(row.get("frame_index"), -1): _safe_float(row.get("time_seconds"), 0.0)
        for row in time_rows
    }
    start_frame = int(math.floor(args.window_start_sec * FPS60))
    end_frame = int(math.ceil(args.window_end_sec * FPS60))
    frame_indices = list(range(start_frame, end_frame + 1))
    frame_times = np.asarray([time_by_frame.get(idx, idx / FPS60) for idx in frame_indices], dtype=np.float64)

    selected_by_h: dict[int, list[dict[str, Any]]] = {}
    for row in selected_rows:
        harmonic_index = _safe_int(row.get("harmonic_index"), 0)
        if harmonic_index <= 0:
            continue
        selected_by_h.setdefault(harmonic_index, []).append(row)

    _write_progress(
        args.progress_json,
        {
            "status": "running",
            "phase": "building_harmonic_trajectories",
            "selected_harmonics": len(selected_by_h),
            "frame_count": len(frame_indices),
        },
    )

    duration_sec = max(0.0, float(args.window_end_sec) - float(args.window_start_sec))
    sample_count = max(1, int(round(duration_sec * args.sample_rate)))
    time_axis = np.arange(sample_count, dtype=np.float32) / float(args.sample_rate) + float(args.window_start_sec)
    audio = np.zeros(sample_count, dtype=np.float32)

    threshold_scale_map = {
        1: 1.00,
        2: 1.00,
        3: 0.70,
        4: 0.65,
        5: 0.60,
        6: 0.55,
        7: 0.45,
    }

    harmonic_summary_rows: list[dict[str, Any]] = []
    for harmonic_index in sorted(selected_by_h):
        family_rows = selected_by_h[harmonic_index]
        probe_indices = [_safe_int(row.get("probe_index"), -1) for row in family_rows if _safe_int(row.get("probe_index"), -1) >= 0]
        if not probe_indices:
            continue
        freq_weights = np.asarray(
            [
                max(0.1, _safe_float(row.get("extraction_score"), 0.0))
                for row in family_rows
            ],
            dtype=np.float32,
        )
        freq_weights = freq_weights / np.maximum(freq_weights.sum(), 1e-9)
        probe_freqs = np.asarray(
            [float(coord_by_probe[probe_idx].frequency_hz) for probe_idx in probe_indices],
            dtype=np.float32,
        )

        probe_matrix = np.asarray(matrix[probe_indices, start_frame:end_frame + 1], dtype=np.float32)
        family_energy_curve = np.average(probe_matrix, axis=0, weights=freq_weights)
        family_energy_curve = _moving_average(family_energy_curve, args.frame_smooth_radius)
        peak_energy = float(np.max(family_energy_curve)) if family_energy_curve.size else 0.0
        threshold = peak_energy * float(args.base_threshold_ratio) * threshold_scale_map.get(harmonic_index, 0.5)
        active_mask = family_energy_curve >= threshold
        active_mask = _bridge_small_gaps(active_mask, args.gap_bridge_frames)

        if not np.any(active_mask):
            continue

        weighted_freq_numer = np.sum(probe_matrix * probe_freqs[:, None], axis=0)
        weighted_freq_denom = np.sum(probe_matrix, axis=0)
        family_freq_curve = np.divide(
            weighted_freq_numer,
            np.maximum(weighted_freq_denom, 1e-9),
            out=np.zeros_like(weighted_freq_numer, dtype=np.float32),
            where=weighted_freq_denom > 0.0,
        ).astype(np.float32)
        family_freq_curve = _moving_average(family_freq_curve, args.frame_smooth_radius)

        active_positions = np.flatnonzero(active_mask)
        first_frame = frame_indices[int(active_positions[0])]
        last_frame = frame_indices[int(active_positions[-1])]
        active_frame_count = int(active_positions.size)

        frame_amp = np.where(active_mask, np.sqrt(np.maximum(0.0, family_energy_curve)), 0.0).astype(np.float32)
        max_amp = float(np.max(frame_amp)) if frame_amp.size else 0.0
        legacy_normalized = False
        if args.amplitude_mode == "legacy" and max_amp > 0.0:
            frame_amp /= max_amp
            legacy_normalized = True

        frame_amp_interp = np.interp(time_axis, frame_times, frame_amp, left=0.0, right=0.0).astype(np.float32)
        frame_freq_interp = np.interp(time_axis, frame_times, family_freq_curve, left=0.0, right=0.0).astype(np.float32)

        if args.amplitude_mode == "legacy":
            gain = (
                float(args.amplitude_scale)
                * _harmonic_gain(harmonic_index)
                * max(0.25, min(1.0, _mean([_safe_float(row.get("extraction_score"), 0.0) for row in family_rows])))
            )
            spectral_reference = 0.0
        else:
            mean_active_amp = _safe_float(np.mean(frame_amp[active_mask]), 0.0)
            if args.amplitude_mode == "spectral":
                mean_active_freq = _safe_float(np.mean(family_freq_curve[active_mask]), 0.0)
                spectral_reference = _sample_spectrum_magnitude(
                    spectral_mag,
                    spectral_freq_axis,
                    int(source_wav_sr),
                    mean_active_freq,
                )
                gain = (
                    float(args.amplitude_scale)
                    * (spectral_reference / source_wav_peak)
                    / max(mean_active_amp, 1e-9)
                )
            else:
                spectral_reference = 0.0
                gain = float(args.amplitude_scale)

        phase = 2.0 * math.pi * np.cumsum(frame_freq_interp, dtype=np.float64) / float(args.sample_rate)
        partial = (frame_amp_interp * gain * np.sin(phase)).astype(np.float32)
        audio += partial

        dominant_token = str(family_rows[0].get("observed_note_token", "")).strip()
        harmonic_summary_rows.append(
            {
                "harmonic_index": harmonic_index,
                "probe_count": len(probe_indices),
                "dominant_token": dominant_token,
                "first_frame_index": first_frame,
                "first_time_sec": time_by_frame.get(first_frame, first_frame / FPS60),
                "last_frame_index": last_frame,
                "last_time_sec": time_by_frame.get(last_frame, last_frame / FPS60),
                "active_frame_count": active_frame_count,
                "max_consecutive_frames": active_frame_count,
                "peak_energy": peak_energy,
                "threshold_energy": threshold,
                "mean_frequency_hz": _safe_float(np.mean(family_freq_curve[active_mask]), 0.0),
                "mean_energy": _safe_float(np.mean(family_energy_curve[active_mask]), 0.0),
                "mean_amplitude": _safe_float(np.mean(frame_amp[active_mask]), 0.0),
                "peak_amplitude": max_amp,
                "render_gain": gain,
                "amplitude_mode": args.amplitude_mode,
                "legacy_normalized": int(legacy_normalized),
                "spectral_reference": spectral_reference,
            }
        )

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.95:
        audio = audio / peak * 0.95

    _write_wav(Path(args.out_wav), audio, args.sample_rate)

    with Path(args.out_harmonic_summary_csv).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "harmonic_index",
                "probe_count",
                "dominant_token",
                "first_frame_index",
                "first_time_sec",
                "last_frame_index",
                "last_time_sec",
                "active_frame_count",
                "max_consecutive_frames",
                "peak_energy",
                "threshold_energy",
                "mean_frequency_hz",
                "mean_energy",
                "mean_amplitude",
                "peak_amplitude",
                "render_gain",
                "amplitude_mode",
                "legacy_normalized",
                "spectral_reference",
            ],
        )
        writer.writeheader()
        writer.writerows(harmonic_summary_rows)

    summary_lines = [
        "WINDOW BOWED BUNDLE PREVIEW",
        "=" * 72,
        f"harmonic_count                      : {len(harmonic_summary_rows)}",
        f"window_duration_sec                 : {duration_sec:.6f}",
        f"sample_rate                         : {args.sample_rate}",
        f"amplitude_mode                      : {args.amplitude_mode}",
        f"peak_after_render                   : {peak:.6f}",
        "",
        "harmonic_trajectories:",
    ]
    for row in harmonic_summary_rows:
        summary_lines.append(
            "  "
            f"h{row['harmonic_index']} token={row['dominant_token']} "
            f"frames={row['active_frame_count']} "
            f"first={_safe_float(row['first_time_sec'], 0.0):.6f} "
            f"last={_safe_float(row['last_time_sec'], 0.0):.6f} "
            f"mean_hz={_safe_float(row['mean_frequency_hz'], 0.0):.6f}"
        )
    summary_lines.extend(
        [
            "",
            "interpretation:",
            "  This version renders one connected trajectory per harmonic, using raw probe",
            "  energies and per-harmonic frequency drift across the full window.",
        ]
    )
    if args.amplitude_mode == "real":
        summary_lines.extend(
            [
                "  amplitude_mode=real preserves inter-harmonic amplitude hierarchy from the",
                "  observed probe energies instead of leveling each harmonic independently.",
            ]
        )
    elif args.amplitude_mode == "spectral":
        summary_lines.extend(
            [
                "  amplitude_mode=spectral calibrates each harmonic against the real FFT",
                "  spectrum of the source window, while keeping probe-derived continuity and",
                "  microshift drift for the harmonic trajectory itself.",
            ]
        )
    else:
        summary_lines.extend(
            [
                "  amplitude_mode=legacy keeps the older leveled harmonic-gain rendering for",
                "  comparison purposes.",
            ]
        )
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "stage": "window_bowed_bundle_preview_v2",
                "renderer": "window_bowed_bundle_preview",
                "harmonic_count": len(harmonic_summary_rows),
                "window_duration_sec": duration_sec,
                "sample_rate": args.sample_rate,
                "amplitude_mode": args.amplitude_mode,
                "source_wav_peak": source_wav_peak if args.amplitude_mode == "spectral" else 0.0,
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
            "harmonic_count": len(harmonic_summary_rows),
        },
    )


if __name__ == "__main__":
    main()
