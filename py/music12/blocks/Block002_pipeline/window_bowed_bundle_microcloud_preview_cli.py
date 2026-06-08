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
        return np.zeros(fft_size // 2 + 1, dtype=np.float32), np.fft.rfftfreq(fft_size, d=1.0).astype(np.float32)
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
    return (acc / float(count)).astype(np.float32), np.fft.rfftfreq(fft_size, d=1.0 / 1.0).astype(np.float32)


def _sample_spectrum_magnitude(mean_mag: np.ndarray, sample_rate: int, fft_size: int, frequency_hz: float) -> float:
    if mean_mag.size == 0:
        return 0.0
    bin_freqs = np.fft.rfftfreq(fft_size, d=1.0 / float(sample_rate))
    target = min(max(0.0, float(frequency_hz)), 0.5 * float(sample_rate))
    return _safe_float(np.interp(target, bin_freqs, mean_mag), 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render a bowed bundle from framewise micro-clouds of adjacent real probe families instead of one ideal line per harmonic."
    )
    ap.add_argument("--families-csv", required=True)
    ap.add_argument("--owner-rows-csv", required=True)
    ap.add_argument("--source-wav", required=True)
    ap.add_argument("--window-start-sec", type=float, required=True)
    ap.add_argument("--window-end-sec", type=float, required=True)
    ap.add_argument("--sample-rate", type=int, default=44100)
    ap.add_argument("--fft-size", type=int, default=4096)
    ap.add_argument("--hop-size", type=int, default=735)
    ap.add_argument("--max-harmonic", type=int, default=25)
    ap.add_argument("--max-piano-overlap", type=float, default=0.0)
    ap.add_argument("--family-score-ratio", type=float, default=0.97)
    ap.add_argument("--max-families-per-harmonic", type=int, default=8)
    ap.add_argument("--gaussian-bin-sigma", type=float, default=1.35)
    ap.add_argument("--amplitude-scale", type=float, default=0.07)
    ap.add_argument("--target-peak", type=float, default=0.30)
    ap.add_argument("--out-wav", required=True)
    ap.add_argument("--out-selected-families-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--progress-json", default="")
    args = ap.parse_args()

    _write_progress(args.progress_json, {"status": "running", "phase": "loading_inputs"})

    family_rows = _load_csv(Path(args.families_csv))
    owner_rows = _load_csv(Path(args.owner_rows_csv))
    source_samples, source_sr = _load_wav_mono_float32(Path(args.source_wav))
    source_spec, _ = _build_average_spectrum(source_samples, int(args.fft_size), max(1, int(args.fft_size // 4)))
    source_peak = max(_safe_float(np.max(source_spec), 0.0), 1e-9)

    selected_family_rows: list[dict[str, Any]] = []
    families_by_h: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in family_rows:
        harmonic_index = _safe_int(row.get("harmonic_index"), 0)
        if 1 <= harmonic_index <= int(args.max_harmonic):
            families_by_h[harmonic_index].append(row)

    for harmonic_index in sorted(families_by_h):
        rows = [
            row
            for row in families_by_h[harmonic_index]
            if _safe_float(row.get("pianoish_overlap_ratio"), 1.0) <= float(args.max_piano_overlap)
        ]
        if not rows:
            continue
        rows.sort(key=lambda row: _safe_float(row.get("extraction_score"), 0.0), reverse=True)
        best_score = _safe_float(rows[0].get("extraction_score"), 0.0)
        keep = [
            row
            for row in rows
            if _safe_float(row.get("extraction_score"), 0.0) >= best_score * float(args.family_score_ratio)
        ]
        selected_family_rows.extend(keep[: max(1, int(args.max_families_per_harmonic))])

    selected_keys = {
        (_safe_int(row.get("harmonic_index"), 0), _safe_int(row.get("probe_index"), -1))
        for row in selected_family_rows
    }
    selected_owner_rows = [
        row
        for row in owner_rows
        if (_safe_int(row.get("harmonic_index"), 0), _safe_int(row.get("probe_index"), -1)) in selected_keys
        and float(args.window_start_sec) <= _safe_float(row.get("time_sec"), -1.0) <= float(args.window_end_sec)
    ]
    if not selected_owner_rows:
        raise SystemExit("No selected bowed micro-cloud rows for preview.")

    _write_progress(
        args.progress_json,
        {
            "status": "running",
            "phase": "building_microcloud",
            "selected_family_count": len(selected_family_rows),
            "selected_owner_rows": len(selected_owner_rows),
        },
    )

    frame_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    family_rows_by_h: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_family_rows:
        family_rows_by_h[_safe_int(row.get("harmonic_index"), 0)].append(row)
    for row in selected_owner_rows:
        frame_rows[_safe_int(row.get("frame_index"), -1)].append(row)

    harmonic_energy_mean: dict[int, float] = {}
    harmonic_freq_mean: dict[int, float] = {}
    harmonic_family_count: dict[int, int] = {}
    harmonic_selected_probe_indices: dict[int, list[int]] = {}
    for harmonic_index, rows in family_rows_by_h.items():
        harmonic_energy_mean[harmonic_index] = max(
            1e-9,
            float(np.mean([_safe_float(row.get("mean_energy_over_frame_p95"), 0.0) for row in rows])),
        )
        harmonic_freq_mean[harmonic_index] = float(
            np.mean([_safe_float(row.get("frequency_hz"), 0.0) for row in rows])
        )
        harmonic_family_count[harmonic_index] = len(rows)
        harmonic_selected_probe_indices[harmonic_index] = [_safe_int(row.get("probe_index"), -1) for row in rows]

    harmonic_scale: dict[int, float] = {}
    for harmonic_index, mean_freq in harmonic_freq_mean.items():
        spectral_ref = _sample_spectrum_magnitude(source_spec, source_sr, int(args.fft_size), mean_freq)
        harmonic_scale[harmonic_index] = (
            float(args.amplitude_scale) * spectral_ref / source_peak / harmonic_energy_mean[harmonic_index]
        )

    duration_sec = max(0.0, float(args.window_end_sec) - float(args.window_end_sec if False else args.window_start_sec))
    sample_count = max(1, int(round(duration_sec * args.sample_rate)))
    output = np.zeros(sample_count + int(args.fft_size), dtype=np.float32)
    window = np.hanning(int(args.fft_size)).astype(np.float32)
    bin_freqs = np.fft.rfftfreq(int(args.fft_size), d=1.0 / float(args.sample_rate))
    phase_acc = np.random.default_rng(0).uniform(0.0, 2.0 * math.pi, size=bin_freqs.size).astype(np.float64)

    frame_indices = list(range(int(math.floor(args.window_start_sec * FPS60)), int(math.ceil(args.window_end_sec * FPS60)) + 1))
    frame_time_map = {idx: idx / FPS60 for idx in frame_indices}
    for row in selected_owner_rows:
        frame_time_map[_safe_int(row.get("frame_index"), -1)] = _safe_float(row.get("time_sec"), 0.0)

    for frame_pos, frame_index in enumerate(frame_indices):
        rows = frame_rows.get(frame_index, [])
        if not rows:
            continue
        mags = np.zeros(bin_freqs.size, dtype=np.float32)
        for row in rows:
            harmonic_index = _safe_int(row.get("harmonic_index"), 0)
            if harmonic_index <= 0:
                continue
            freq_hz = _safe_float(row.get("frequency_hz"), 0.0)
            energy = _safe_float(row.get("energy_over_frame_p95"), 0.0)
            row_mag = energy * harmonic_scale.get(harmonic_index, 0.0)
            center_bin = freq_hz * float(args.fft_size) / float(args.sample_rate)
            sigma = max(float(args.gaussian_bin_sigma), 0.75)
            left = max(0, int(math.floor(center_bin - 3.0 * sigma)))
            right = min(bin_freqs.size - 1, int(math.ceil(center_bin + 3.0 * sigma)))
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

    selected_family_rows_sorted = sorted(
        selected_family_rows,
        key=lambda row: (_safe_int(row.get("harmonic_index"), 0), -_safe_float(row.get("extraction_score"), 0.0)),
    )
    with Path(args.out_selected_families_csv).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "harmonic_index",
                "probe_index",
                "observed_note_token",
                "frequency_hz",
                "active_frame_count",
                "mean_energy_over_frame_p95",
                "pianoish_overlap_ratio",
                "extraction_score",
            ],
        )
        writer.writeheader()
        for row in selected_family_rows_sorted:
            writer.writerow(
                {
                    "harmonic_index": _safe_int(row.get("harmonic_index"), 0),
                    "probe_index": _safe_int(row.get("probe_index"), -1),
                    "observed_note_token": str(row.get("observed_note_token", "")).strip(),
                    "frequency_hz": _safe_float(row.get("frequency_hz"), 0.0),
                    "active_frame_count": _safe_int(row.get("active_frame_count"), 0),
                    "mean_energy_over_frame_p95": _safe_float(row.get("mean_energy_over_frame_p95"), 0.0),
                    "pianoish_overlap_ratio": _safe_float(row.get("pianoish_overlap_ratio"), 0.0),
                    "extraction_score": _safe_float(row.get("extraction_score"), 0.0),
                }
            )

    summary_lines = [
        "WINDOW BOWED BUNDLE MICROCLOUD PREVIEW",
        "=" * 72,
        f"selected_family_count              : {len(selected_family_rows)}",
        f"selected_owner_row_count           : {len(selected_owner_rows)}",
        f"window_duration_sec                : {duration_sec:.6f}",
        f"sample_rate                        : {args.sample_rate}",
        f"fft_size                           : {args.fft_size}",
        f"hop_size                           : {args.hop_size}",
        f"peak_after_render                  : {peak:.6f}",
        "",
        "harmonic_microclouds:",
    ]
    for harmonic_index in sorted(harmonic_family_count):
        summary_lines.append(
            "  "
            f"h{harmonic_index} families={harmonic_family_count[harmonic_index]} "
            f"mean_hz={harmonic_freq_mean[harmonic_index]:.6f} "
            f"scale={harmonic_scale[harmonic_index]:.9f}"
        )
    summary_lines.extend(
        [
            "",
            "interpretation:",
            "  This preview uses multiple adjacent real probe families per harmonic and",
            "  framewise spectral-cloud rendering, instead of one ideal line per harmonic.",
            "  It is intended to restore the observed micro-width and local peak drift",
            "  visible in the live spectrum while keeping piano-free bowed ownership.",
        ]
    )
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "stage": "window_bowed_bundle_microcloud_preview",
                "selected_family_count": len(selected_family_rows),
                "selected_owner_row_count": len(selected_owner_rows),
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

    _write_progress(
        args.progress_json,
        {
            "status": "done",
            "phase": "complete",
            "selected_family_count": len(selected_family_rows),
            "selected_owner_row_count": len(selected_owner_rows),
        },
    )


if __name__ == "__main__":
    main()
