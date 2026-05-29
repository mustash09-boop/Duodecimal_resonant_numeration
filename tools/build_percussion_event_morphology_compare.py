from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(r"E:\Duodecimal_resonant_numeration")
PERC_ROOT = PROJECT_ROOT / "Block004_data" / "percussion"
SUMMARY_CSV = PERC_ROOT / "10_reports" / "percussion__event_pipeline_summary.csv"
OUT_DIR = PERC_ROOT / "45_morphology_compare"


NUM_COLS = [
    "duration_sec",
    "peak_amp",
    "rms_amp",
    "attack_time_sec",
    "decay_to_20pct_sec",
    "spectral_centroid_hz",
    "spectral_spread_hz",
    "dominant_freq_hz",
    "dense_peaks",
    "frequency_clusters",
]


def zsafe(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    std = float(s.std())
    if std <= 1e-12:
        return s * 0.0
    return (s - float(s.mean())) / std


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SUMMARY_CSV)

    raw_csv = OUT_DIR / "01_percussion_event_rows.csv"
    feature_csv = OUT_DIR / "02_percussion_instrument_features.csv"
    pairwise_csv = OUT_DIR / "03_percussion_pairwise_morphology.csv"
    meta_json = OUT_DIR / "percussion_morphology_compare_meta.json"

    df.to_csv(raw_csv, index=False, encoding="utf-8-sig")

    feat = (
        df.groupby("instrument_name", as_index=False)
        .agg(
            event_count=("original_filename", "count"),
            gesture_type_count=("gesture_type", "nunique"),
            dynamic_count=("dynamic", "nunique"),
            articulation_count=("articulation", "nunique"),
            duration_sec=("duration_sec", "mean"),
            peak_amp=("peak_amp", "mean"),
            rms_amp=("rms_amp", "mean"),
            attack_time_sec=("attack_time_sec", "mean"),
            decay_to_20pct_sec=("decay_to_20pct_sec", "mean"),
            spectral_centroid_hz=("spectral_centroid_hz", "mean"),
            spectral_spread_hz=("spectral_spread_hz", "mean"),
            dominant_freq_hz=("dominant_freq_hz", "mean"),
            dense_peaks=("dense_peaks", "mean"),
            frequency_clusters=("frequency_clusters", "mean"),
        )
        .sort_values("instrument_name")
    )

    for col in NUM_COLS:
        feat[f"z_{col}"] = zsafe(feat[col])

    feat.to_csv(feature_csv, index=False, encoding="utf-8-sig")

    rows = []
    for a, b in combinations(feat["instrument_name"].tolist(), 2):
        ra = feat[feat["instrument_name"] == a].iloc[0]
        rb = feat[feat["instrument_name"] == b].iloc[0]
        dz = []
        for col in NUM_COLS:
            dz.append(abs(float(ra[f"z_{col}"]) - float(rb[f"z_{col}"])))
        rows.append(
            {
                "instrument_a": a,
                "instrument_b": b,
                "mean_z_distance": float(np.mean(dz)),
                "max_z_distance": float(np.max(dz)),
                "duration_delta": abs(float(ra["duration_sec"]) - float(rb["duration_sec"])),
                "centroid_delta": abs(float(ra["spectral_centroid_hz"]) - float(rb["spectral_centroid_hz"])),
                "spread_delta": abs(float(ra["spectral_spread_hz"]) - float(rb["spectral_spread_hz"])),
                "attack_delta": abs(float(ra["attack_time_sec"]) - float(rb["attack_time_sec"])),
                "decay_delta": abs(float(ra["decay_to_20pct_sec"]) - float(rb["decay_to_20pct_sec"])),
                "morphology_distance_score": float(np.mean(dz)),
            }
        )

    pair_df = pd.DataFrame(rows).sort_values("morphology_distance_score")
    pair_df.to_csv(pairwise_csv, index=False, encoding="utf-8-sig")
    meta_json.write_text(
        json.dumps(
            {
                "event_rows": int(len(df)),
                "instrument_count": int(feat["instrument_name"].nunique()),
                "numeric_features": NUM_COLS,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("OK")
    print(f"event_rows      : {len(df)}")
    print(f"instrument_count: {feat['instrument_name'].nunique()}")
    print(f"out_dir         : {OUT_DIR}")


if __name__ == "__main__":
    main()
