from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf


# ------------------------------------------------------------
# Hamming window
# ------------------------------------------------------------

def hamming_window(n: int) -> np.ndarray:
    return np.hamming(n).astype(np.float32)


# ------------------------------------------------------------
# Parabolic interpolation (frequency refinement)
# ------------------------------------------------------------

def parabolic_interpolation(mag: np.ndarray, k: int):
    if k <= 0 or k >= len(mag) - 1:
        return k, 0.0

    alpha = mag[k - 1]
    beta = mag[k]
    gamma = mag[k + 1]

    denom = (alpha - 2 * beta + gamma)
    if denom == 0:
        return k, 0.0

    delta = 0.5 * (alpha - gamma) / denom
    return k + delta, delta


# ------------------------------------------------------------
# Peak detection
# ------------------------------------------------------------

def find_peaks(mag: np.ndarray, threshold: float):
    peaks = []
    for i in range(1, len(mag) - 1):
        if mag[i] > mag[i - 1] and mag[i] > mag[i + 1]:
            if mag[i] > threshold:
                peaks.append(i)
    return peaks


# ------------------------------------------------------------
# Main processing
# ------------------------------------------------------------

def process_wav(
    wav_path: Path,
    out_csv: Path,
    window_sec: float,
    step_sec: float,
    peak_threshold: float,
):
    data, sr = sf.read(str(wav_path))
    if data.ndim > 1:
        data = data.mean(axis=1)

    data = data.astype(np.float32)

    win_size = int(window_sec * sr)
    step_size = int(step_sec * sr)

    window = hamming_window(win_size)

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "time_sec",
            "freq_hz",
            "amplitude",
            "phase_rad",
            "frame_index",
            "peak_index",
        ])

        frame_idx = 0

        for start in range(0, len(data) - win_size, step_size):
            segment = data[start:start + win_size]
            segment = segment * window

            spectrum = np.fft.rfft(segment)
            mag = np.abs(spectrum)

            threshold = peak_threshold * np.max(mag)

            peaks = find_peaks(mag, threshold)

            for rank, k in enumerate(peaks):
                k_refined, delta = parabolic_interpolation(mag, k)

                freq = k_refined * sr / win_size

                amp = mag[k]

                phase = np.angle(spectrum[k])

                time_sec = (start + win_size // 2) / sr

                writer.writerow([
                    time_sec,
                    freq,
                    amp,
                    phase,
                    frame_idx,
                    rank,
                ])

            frame_idx += 1


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Dense spectral observer (time, freq, amplitude, phase)"
    )

    parser.add_argument("--wav", required=True)
    parser.add_argument("--out_csv", required=True)

    parser.add_argument("--window_sec", type=float, default=0.05)
    parser.add_argument("--step_sec", type=float, default=1.0 / 60.0)
    parser.add_argument("--peak_threshold", type=float, default=0.05)

    args = parser.parse_args()

    process_wav(
        wav_path=Path(args.wav),
        out_csv=Path(args.out_csv),
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        peak_threshold=args.peak_threshold,
    )


if __name__ == "__main__":
    main()