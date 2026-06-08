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


def _write_wav(path: Path, mono: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.rint(mono * 32767.0), -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(clipped.tobytes())


def _stft(signal: np.ndarray, fft_size: int, hop_size: int) -> tuple[np.ndarray, np.ndarray, int]:
    window = np.hanning(fft_size).astype(np.float32)
    if signal.size < fft_size:
        padded = np.zeros(fft_size, dtype=np.float32)
        padded[:signal.size] = signal
        signal = padded
    if signal.size <= fft_size:
        frame_count = 1
    else:
        frame_count = int(math.ceil((signal.size - fft_size) / float(hop_size))) + 1
    padded_len = (frame_count - 1) * hop_size + fft_size
    if padded_len > signal.size:
        signal = np.pad(signal, (0, padded_len - signal.size), mode="constant")
    frame_starts = [frame_idx * hop_size for frame_idx in range(frame_count)]
    spectra = []
    for start in frame_starts:
        frame = signal[start:start + fft_size]
        spectra.append(np.fft.rfft(frame * window))
    return np.asarray(spectra, dtype=np.complex128).T, window, padded_len


def _istft(spec: np.ndarray, window: np.ndarray, hop_size: int, out_len: int) -> np.ndarray:
    fft_size = (spec.shape[0] - 1) * 2
    frame_count = spec.shape[1]
    out = np.zeros(max(out_len, (frame_count - 1) * hop_size + fft_size), dtype=np.float32)
    norm = np.zeros_like(out)
    for frame_idx in range(frame_count):
        start = frame_idx * hop_size
        frame = np.fft.irfft(spec[:, frame_idx], n=fft_size).astype(np.float32)
        out[start:start + fft_size] += frame * window
        norm[start:start + fft_size] += window * window
    valid = norm > 1e-9
    out[valid] /= norm[valid]
    return out[:out_len]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Reconstruct the bowed layer directly from the source window STFT, preserving original complex phase and keeping only source-locked bins supported by observed bowed rows."
    )
    ap.add_argument("--source-wav", required=True)
    ap.add_argument("--resolved-cloud-csv", required=True)
    ap.add_argument("--window-start-sec", type=float, required=True)
    ap.add_argument("--sample-rate", type=int, default=44100)
    ap.add_argument("--fft-size", type=int, default=4096)
    ap.add_argument("--hop-size", type=int, default=735)
    ap.add_argument("--gaussian-bin-sigma", type=float, default=2.5)
    ap.add_argument("--skirt-bin-sigma", type=float, default=6.0)
    ap.add_argument("--skirt_gain", type=float, default=0.28)
    ap.add_argument("--target-peak", type=float, default=0.30)
    ap.add_argument("--normalization-percentile", type=float, default=99.9)
    ap.add_argument("--out-wav", required=True)
    ap.add_argument("--out-mask-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    args = ap.parse_args()

    source, source_sr = _load_wav_mono_float32(Path(args.source_wav))
    if int(source_sr) != int(args.sample_rate):
        raise ValueError(f"Expected sample_rate {args.sample_rate}, got {source_sr}")
    rows = _load_csv(Path(args.resolved_cloud_csv))
    observed_rows = [
        row for row in rows
        if str(row.get("continuity_source", "")).strip() == "observed"
    ]
    if not observed_rows:
        raise SystemExit("No observed rows found in resolved cloud CSV.")

    spec, window, padded_len = _stft(source, int(args.fft_size), int(args.hop_size))
    bin_count, frame_count = spec.shape
    mask = np.zeros((bin_count, frame_count), dtype=np.float32)
    bin_freqs = np.fft.rfftfreq(int(args.fft_size), d=1.0 / float(args.sample_rate))

    rows_by_stft_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in observed_rows:
        local_time = _safe_float(row.get("time_sec"), 0.0) - float(args.window_start_sec)
        stft_frame = int(round(local_time * float(args.sample_rate) / float(args.hop_size)))
        if 0 <= stft_frame < frame_count:
            rows_by_stft_frame[stft_frame].append(row)

    mask_rows: list[dict[str, Any]] = []
    harmonic_counts: Counter[int] = Counter()
    for stft_frame, frame_rows in rows_by_stft_frame.items():
        max_raw = max(max(0.0, _safe_float(row.get("raw_energy"), 0.0)) for row in frame_rows)
        if max_raw <= 0.0:
            continue
        for row in frame_rows:
            harmonic_index = _safe_int(row.get("harmonic_index"), 0)
            harmonic_counts[harmonic_index] += 1
            freq_hz = _safe_float(row.get("frequency_hz"), 0.0)
            raw_energy = max(0.0, _safe_float(row.get("raw_energy"), 0.0))
            weight = raw_energy / max_raw
            center_bin = freq_hz * float(args.fft_size) / float(args.sample_rate)
            core_sigma = max(float(args.gaussian_bin_sigma), 0.75)
            skirt_sigma = max(float(args.skirt_bin_sigma), core_sigma + 0.5)
            left = max(0, int(math.floor(center_bin - 3.0 * skirt_sigma)))
            right = min(bin_count - 1, int(math.ceil(center_bin + 3.0 * skirt_sigma)))
            bin_ids = np.arange(left, right + 1, dtype=np.float32)
            core = np.exp(-0.5 * ((bin_ids - center_bin) / core_sigma) ** 2).astype(np.float32)
            skirt = np.exp(-0.5 * ((bin_ids - center_bin) / skirt_sigma) ** 2).astype(np.float32)
            shape = np.maximum(core, skirt * float(args.skirt_gain))
            mask[left:right + 1, stft_frame] = np.maximum(mask[left:right + 1, stft_frame], shape * weight)
            mask_rows.append(
                {
                    "stft_frame_index": stft_frame,
                    "harmonic_index": harmonic_index,
                    "probe_index": _safe_int(row.get("probe_index"), -1),
                    "frequency_hz": freq_hz,
                    "raw_energy": raw_energy,
                    "weight_in_frame": weight,
                    "center_bin": center_bin,
                }
            )

    filtered_spec = spec * mask.astype(np.complex128)
    recon = _istft(filtered_spec, window, int(args.hop_size), len(source))

    peak = float(np.max(np.abs(recon))) if recon.size else 0.0
    abs_recon = np.abs(recon)
    normalization_percentile = min(max(float(args.normalization_percentile), 50.0), 100.0)
    robust_peak = float(np.percentile(abs_recon, normalization_percentile)) if recon.size else 0.0
    if robust_peak > 1e-9:
        scale = float(args.target_peak) / robust_peak
        recon = recon * scale
        peak = float(np.max(np.abs(recon))) if recon.size else 0.0
        if peak > 0.95:
            recon = recon / peak * 0.95
            peak = float(np.max(np.abs(recon))) if recon.size else peak
    robust_peak_after = float(np.percentile(np.abs(recon), normalization_percentile)) if recon.size else 0.0

    _write_wav(Path(args.out_wav), recon, int(args.sample_rate))

    with Path(args.out_mask_csv).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "stft_frame_index",
                "harmonic_index",
                "probe_index",
                "frequency_hz",
                "raw_energy",
                "weight_in_frame",
                "center_bin",
            ],
        )
        writer.writeheader()
        writer.writerows(mask_rows)

    active_stft_frames = sorted(rows_by_stft_frame)
    segments = []
    if active_stft_frames:
        start = prev = active_stft_frames[0]
        for frame in active_stft_frames[1:]:
            if frame == prev + 1:
                prev = frame
            else:
                segments.append((start, prev))
                start = prev = frame
        segments.append((start, prev))

    summary_lines = [
        "WINDOW BOWED SOURCE-LOCKED STFT RECONSTRUCTOR",
        "=" * 72,
        f"observed_row_count                 : {len(observed_rows)}",
        f"active_stft_frame_count            : {len(active_stft_frames)}",
        f"fft_size                           : {args.fft_size}",
        f"hop_size                           : {args.hop_size}",
        f"padded_signal_len                  : {padded_len}",
        f"normalization_percentile           : {normalization_percentile}",
        f"robust_peak_after_render           : {robust_peak_after:.6f}",
        f"peak_after_render                  : {peak:.6f}",
        "",
        "active_stft_segments:",
    ]
    for start, end in segments:
        summary_lines.append(f"  {start}..{end} len={end - start + 1}")
    summary_lines.extend(["", "harmonic_observed_row_counts:"])
    for harmonic_index in sorted(harmonic_counts):
        summary_lines.append(f"  h{harmonic_index}: {harmonic_counts[harmonic_index]}")
    summary_lines.extend(
        [
            "",
            "interpretation:",
            "  This reconstruction keeps the original complex STFT phase of the source",
            "  window and filters it with a bowed-only mask derived from observed rows.",
            "  It does not synthesize new oscillators; it keeps source-locked spectral",
            "  detail wherever the observed bowed layer says the note is present.",
        ]
    )
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "stage": "window_bowed_source_locked_stft_reconstructor",
                "observed_row_count": len(observed_rows),
                "active_stft_frame_count": len(active_stft_frames),
                "fft_size": args.fft_size,
                "hop_size": args.hop_size,
                "padded_signal_len": padded_len,
                "normalization_percentile": normalization_percentile,
                "robust_peak_after_render": robust_peak_after,
                "peak_after_render": peak,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
