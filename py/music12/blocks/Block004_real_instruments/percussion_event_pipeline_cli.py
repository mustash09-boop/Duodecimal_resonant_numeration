# -*- coding: utf-8 -*-
"""
PERCUSSION EVENT PIPELINE

Для ударных событий без обязательной ноты:
- читает percussion_manifest_events.csv
- анализирует WAV
- строит dense-пики по времени
- считает attack / decay / spectral centroid
- собирает resonance-кластеры
- создаёт PNG спектрального профиля
- создаёт PNG 12-ричной спирали резонансов
- пишет общий summary по всем событиям

Это НЕ note pipeline.
Здесь нет expected_note, root consensus и theory comparison.
"""

import os
import csv
import json
import math
import wave
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DIGITS12 = "123456789ABC"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_name(s):
    return str(s).replace(" ", "_").replace("/", "_").replace("\\", "_")


def read_wav_mono(path):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if sampwidth == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        data = data / 32768.0
    elif sampwidth == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        data = (
            b[:, 0].astype(np.int32)
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 2].astype(np.int32) << 16)
        )
        mask = data & 0x800000
        data = data - (mask << 1)
        data = data.astype(np.float32) / 8388608.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
        data = data / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)

    return sr, data


def hz_to_token_approx(freq_hz, a4_hz=440.0):
    """
    Приближённая 12-ричная токенизация для визуализации percussion.
    Это не SSOT-замена pitch12; только подпись частот.
    A4 = 9.A
    """
    if freq_hz <= 0:
        return ""

    midi_float = 69 + 12 * math.log2(freq_hz / a4_hz)
    semi = int(round(midi_float - 69))

    base_oct = 9
    base_deg_index = DIGITS12.index("A")

    total = base_deg_index + semi
    oct_shift, deg_index = divmod(total, 12)

    octave = base_oct + oct_shift
    degree = DIGITS12[deg_index]

    return f"{octave}.{degree}'-"


def token_to_spiral_xy(token):
    """
    Упрощённая 12-ричная спираль для percussion.
    token вида:
      9.A'-
      A.3'-
      11.C'-
    """
    if token is None:
        return None

    token = str(token).strip()
    clean = token.split("'")[0].replace("-", "")

    if "." not in clean:
        return None

    octave_s, degree_s = clean.split(".", 1)

    try:
        octave = int(octave_s)
        degree = DIGITS12.index(degree_s[0]) + 1
    except Exception:
        return None

    angle = (degree - 1) * (2.0 * math.pi / 12.0)
    radius = octave + degree / 12.0

    return radius * math.cos(angle), radius * math.sin(angle)


def stft_peak_scan(
    y,
    sr,
    frame_size=4096,
    hop_size=1024,
    max_freq=18000.0,
    top_n=24,
    min_rel_amp=0.02,
):
    if len(y) < frame_size:
        y = np.pad(y, (0, frame_size - len(y)))

    window = np.hanning(frame_size).astype(np.float32)
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sr)

    rows = []
    frame_index = 0

    for start in range(0, max(1, len(y) - frame_size + 1), hop_size):
        frame = y[start:start + frame_size]

        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))

        spectrum = np.fft.rfft(frame * window)
        mag = np.abs(spectrum)
        phase = np.angle(spectrum)

        valid = np.where((freqs > 20.0) & (freqs <= max_freq))[0]
        if len(valid) == 0:
            frame_index += 1
            continue

        mag_valid = mag[valid]
        max_amp = float(mag_valid.max()) if len(mag_valid) else 0.0

        if max_amp <= 0:
            frame_index += 1
            continue

        candidate_idx = valid[np.argsort(mag_valid)[-top_n:]]
        candidate_idx = sorted(candidate_idx, key=lambda i: mag[i], reverse=True)

        for rank, idx in enumerate(candidate_idx, start=1):
            amp = float(mag[idx])
            rel_amp = amp / max_amp

            if rel_amp < min_rel_amp:
                continue

            hz = float(freqs[idx])

            rows.append(
                {
                    "frame_index": frame_index,
                    "time_sec": start / sr,
                    "peak_rank": rank,
                    "freq_hz": hz,
                    "amplitude": amp,
                    "relative_amp": rel_amp,
                    "phase_rad": float(phase[idx]),
                    "note_token": hz_to_token_approx(hz),
                }
            )

        frame_index += 1

    return pd.DataFrame(rows)


def envelope_metrics(y, sr):
    abs_y = np.abs(y)

    if len(abs_y) == 0:
        return {}

    peak = float(abs_y.max())
    rms = float(np.sqrt(np.mean(y ** 2)))
    duration = len(y) / sr

    if peak <= 0:
        return {
            "duration_sec": duration,
            "peak_amp": 0.0,
            "rms_amp": rms,
            "attack_time_sec": 0.0,
            "decay_to_20pct_sec": duration,
        }

    peak_idx = int(np.argmax(abs_y))
    attack_time = peak_idx / sr

    after = abs_y[peak_idx:]
    threshold = peak * 0.2
    below = np.where(after <= threshold)[0]

    if len(below):
        decay_time = below[0] / sr
    else:
        decay_time = max(0.0, duration - attack_time)

    return {
        "duration_sec": duration,
        "peak_amp": peak,
        "rms_amp": rms,
        "attack_time_sec": attack_time,
        "decay_to_20pct_sec": decay_time,
    }


def spectral_metrics(dense_df):
    if dense_df is None or len(dense_df) == 0:
        return {
            "spectral_centroid_hz": 0.0,
            "spectral_spread_hz": 0.0,
            "dominant_freq_hz": 0.0,
            "dominant_token": "",
        }

    weights = dense_df["amplitude"].astype(float).to_numpy()
    freqs = dense_df["freq_hz"].astype(float).to_numpy()

    total = weights.sum()

    if total <= 0:
        return {
            "spectral_centroid_hz": 0.0,
            "spectral_spread_hz": 0.0,
            "dominant_freq_hz": 0.0,
            "dominant_token": "",
        }

    centroid = float((freqs * weights).sum() / total)
    spread = float(np.sqrt(((freqs - centroid) ** 2 * weights).sum() / total))

    top = dense_df.sort_values("amplitude", ascending=False).iloc[0]

    return {
        "spectral_centroid_hz": centroid,
        "spectral_spread_hz": spread,
        "dominant_freq_hz": float(top["freq_hz"]),
        "dominant_token": str(top["note_token"]),
    }


def frequency_clusters(dense_df, bin_hz=20.0, min_hits=2):
    if dense_df is None or len(dense_df) == 0:
        return pd.DataFrame()

    df = dense_df.copy()
    df["cluster_hz"] = (df["freq_hz"] / bin_hz).round() * bin_hz

    grouped = (
        df.groupby("cluster_hz")
        .agg(
            hit_count=("freq_hz", "count"),
            frame_count=("frame_index", "nunique"),
            mean_freq_hz=("freq_hz", "mean"),
            mean_amp=("amplitude", "mean"),
            max_amp=("amplitude", "max"),
            mean_rel_amp=("relative_amp", "mean"),
        )
        .reset_index()
    )

    grouped = grouped[grouped["hit_count"] >= min_hits]
    grouped["token"] = grouped["mean_freq_hz"].apply(hz_to_token_approx)

    grouped = grouped.sort_values(
        by=["frame_count", "mean_amp"],
        ascending=[False, False],
    )

    return grouped


def save_event_png(event_name, dense_df, clusters_df, out_png):
    if dense_df is None or len(dense_df) == 0:
        return

    plt.figure(figsize=(10, 5))

    plt.scatter(
        dense_df["time_sec"],
        dense_df["freq_hz"],
        s=np.maximum(8, dense_df["relative_amp"] * 80),
        alpha=0.55,
        label="dense peaks",
    )

    if clusters_df is not None and len(clusters_df) > 0:
        top = clusters_df.head(12)

        for _, r in top.iterrows():
            plt.axhline(
                float(r["mean_freq_hz"]),
                linewidth=0.8,
                alpha=0.35,
            )
            plt.text(
                0,
                float(r["mean_freq_hz"]),
                str(r["token"]),
                fontsize=7,
            )

    plt.title(f"Percussion event spectrum: {event_name}")
    plt.xlabel("time, sec")
    plt.ylabel("frequency, Hz")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def save_event_spiral_png(event_name, dense_df, clusters_df, out_png):
    points = []
    cluster_points = []

    if dense_df is not None and len(dense_df) > 0:
        for _, r in dense_df.iterrows():
            xy = token_to_spiral_xy(r.get("note_token", ""))
            if xy:
                points.append(
                    {
                        "x": xy[0],
                        "y": xy[1],
                        "amp": float(r.get("relative_amp", 0.0)),
                        "token": str(r.get("note_token", "")),
                    }
                )

    if clusters_df is not None and len(clusters_df) > 0:
        for _, r in clusters_df.head(40).iterrows():
            xy = token_to_spiral_xy(r.get("token", ""))
            if xy:
                cluster_points.append(
                    {
                        "x": xy[0],
                        "y": xy[1],
                        "amp": float(r.get("mean_rel_amp", 0.0)),
                        "token": str(r.get("token", "")),
                        "hz": float(r.get("mean_freq_hz", 0.0)),
                    }
                )

    if not points and not cluster_points:
        return

    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    for radius in range(1, 15):
        circle = plt.Circle(
            (0, 0),
            radius,
            fill=False,
            linewidth=0.4,
            alpha=0.25,
        )
        ax.add_patch(circle)

    for i in range(12):
        angle = i * (2.0 * math.pi / 12.0)
        plt.plot(
            [0, 15 * math.cos(angle)],
            [0, 15 * math.sin(angle)],
            linewidth=0.4,
            alpha=0.25,
        )

    if points:
        plt.scatter(
            [p["x"] for p in points],
            [p["y"] for p in points],
            s=[max(6, p["amp"] * 35) for p in points],
            alpha=0.25,
            label="dense peaks",
        )

    if cluster_points:
        plt.scatter(
            [p["x"] for p in cluster_points],
            [p["y"] for p in cluster_points],
            s=[max(40, p["amp"] * 160) for p in cluster_points],
            marker="x",
            label="resonance clusters",
        )

        for p in cluster_points[:25]:
            plt.text(
                p["x"],
                p["y"],
                f"{p['token']}\n{p['hz']:.0f}",
                fontsize=6,
            )

    plt.title(f"Percussion 12-spiral resonance map: {event_name}")
    plt.axis("equal")
    plt.grid(True, alpha=0.2)
    plt.legend()

    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()


def process_event(row, reports_root, args):
    wav_path = row["wav_path"]
    original_filename = row["original_filename"]

    stem = os.path.splitext(original_filename)[0]
    event_dir = os.path.join(reports_root, safe_name(stem))
    ensure_dir(event_dir)

    sr, y = read_wav_mono(wav_path)

    dense_df = stft_peak_scan(
        y=y,
        sr=sr,
        frame_size=args.frame_size,
        hop_size=args.hop_size,
        max_freq=args.max_freq,
        top_n=args.top_n,
        min_rel_amp=args.min_rel_amp,
    )

    dense_csv = os.path.join(event_dir, f"{stem}__percussion_dense.csv")
    dense_df.to_csv(dense_csv, index=False)

    clusters_df = frequency_clusters(
        dense_df,
        bin_hz=args.cluster_bin_hz,
        min_hits=args.cluster_min_hits,
    )

    clusters_csv = os.path.join(event_dir, f"{stem}__percussion_frequency_clusters.csv")
    clusters_df.to_csv(clusters_csv, index=False)

    env = envelope_metrics(y, sr)
    spec = spectral_metrics(dense_df)

    summary = {
        "original_filename": original_filename,
        "wav_path": wav_path,
        "instrument_family": row.get("instrument_family", "percussion"),
        "instrument_name": row.get("instrument_name", ""),
        "event_id": row.get("event_id", ""),
        "dynamic": row.get("dynamic", ""),
        "articulation": row.get("articulation", ""),
        "gesture_type": row.get("gesture_type", ""),
        "sample_rate": sr,
        "dense_peaks": int(len(dense_df)),
        "frequency_clusters": int(len(clusters_df)),
        **env,
        **spec,
    }

    summary_json = os.path.join(event_dir, f"{stem}__percussion_event_summary.json")
    summary_txt = os.path.join(event_dir, f"{stem}__percussion_event_summary.txt")
    png_path = os.path.join(event_dir, f"{stem}__percussion_spectrum.png")
    spiral_png_path = os.path.join(event_dir, f"{stem}__percussion_spiral.png")

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"PERCUSSION EVENT SUMMARY: {stem}\n")
        f.write("=" * 80 + "\n")

        for k, v in summary.items():
            f.write(f"{k:24}: {v}\n")

        f.write("\nTOP FREQUENCY CLUSTERS\n")
        f.write("-" * 80 + "\n")

        for _, r in clusters_df.head(20).iterrows():
            f.write(
                f"{r['token']:10} "
                f"hz={r['mean_freq_hz']:.2f} "
                f"frames={int(r['frame_count']):4d} "
                f"hits={int(r['hit_count']):4d} "
                f"mean_amp={r['mean_amp']:.6f}\n"
            )

    save_event_png(stem, dense_df, clusters_df, png_path)
    save_event_spiral_png(stem, dense_df, clusters_df, spiral_png_path)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run percussion event analysis without expected_note/root."
    )

    parser.add_argument("--manifest_csv", required=True)
    parser.add_argument("--reports_root", required=True)

    parser.add_argument("--frame_size", type=int, default=4096)
    parser.add_argument("--hop_size", type=int, default=1024)
    parser.add_argument("--max_freq", type=float, default=18000.0)
    parser.add_argument("--top_n", type=int, default=24)
    parser.add_argument("--min_rel_amp", type=float, default=0.02)

    parser.add_argument("--cluster_bin_hz", type=float, default=20.0)
    parser.add_argument("--cluster_min_hits", type=int, default=2)

    args = parser.parse_args()

    ensure_dir(args.reports_root)

    manifest = pd.read_csv(args.manifest_csv)

    summaries = []
    failed = []

    for _, row in manifest.iterrows():
        try:
            summaries.append(
                process_event(
                    row=row,
                    reports_root=args.reports_root,
                    args=args,
                )
            )
        except Exception as e:
            failed.append(
                {
                    "original_filename": row.get("original_filename", ""),
                    "wav_path": row.get("wav_path", ""),
                    "error": repr(e),
                }
            )

    summary_df = pd.DataFrame(summaries)

    out_summary_csv = os.path.join(args.reports_root, "percussion__event_pipeline_summary.csv")
    out_summary_json = os.path.join(args.reports_root, "percussion__event_pipeline_summary.json")
    out_failed_csv = os.path.join(args.reports_root, "percussion__event_pipeline_failed.csv")

    summary_df.to_csv(out_summary_csv, index=False)

    with open(out_summary_json, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    pd.DataFrame(failed).to_csv(out_failed_csv, index=False)

    print("PERCUSSION EVENT PIPELINE DONE")
    print(f"manifest_csv : {args.manifest_csv}")
    print(f"reports_root : {args.reports_root}")
    print(f"processed    : {len(summaries)}")
    print(f"failed       : {len(failed)}")


if __name__ == "__main__":
    main()