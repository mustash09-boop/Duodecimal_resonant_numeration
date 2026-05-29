from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(r"E:\Duodecimal_resonant_numeration")
PERC_ROOT = PROJECT_ROOT / "Block004_data" / "percussion"
PASSPORT_DIR = PERC_ROOT / "40_passports"
FEATURE_CSV = PERC_ROOT / "45_morphology_compare" / "02_percussion_instrument_features.csv"
PAIRWISE_CSV = PERC_ROOT / "45_morphology_compare" / "03_percussion_pairwise_morphology.csv"


def update_passport(passport_path: Path, feat_df: pd.DataFrame, pair_df: pd.DataFrame) -> None:
    data = json.loads(passport_path.read_text(encoding="utf-8"))
    instrument = data.get("instrument_name", "")
    feat = feat_df[feat_df["instrument_name"] == instrument]
    if len(feat) == 0:
        return
    feat_row = feat.iloc[0]

    a = pair_df[pair_df["instrument_a"] == instrument].copy()
    a["other_instrument"] = a["instrument_b"]
    b = pair_df[pair_df["instrument_b"] == instrument].copy()
    b["other_instrument"] = b["instrument_a"]
    pair_sub = pd.concat([a, b], ignore_index=True)
    closest = pair_sub.sort_values("morphology_distance_score").head(6)
    furthest = pair_sub.sort_values("morphology_distance_score", ascending=False).head(6)

    block = {
        "event_count": int(feat_row["event_count"]),
        "gesture_type_count": int(feat_row["gesture_type_count"]),
        "dynamic_count": int(feat_row["dynamic_count"]),
        "articulation_count": int(feat_row["articulation_count"]),
        "mean_duration_sec": float(feat_row["duration_sec"]),
        "mean_attack_time_sec": float(feat_row["attack_time_sec"]),
        "mean_decay_to_20pct_sec": float(feat_row["decay_to_20pct_sec"]),
        "mean_spectral_centroid_hz": float(feat_row["spectral_centroid_hz"]),
        "mean_spectral_spread_hz": float(feat_row["spectral_spread_hz"]),
        "mean_dense_peaks": float(feat_row["dense_peaks"]),
        "mean_frequency_clusters": float(feat_row["frequency_clusters"]),
        "closest_instruments": [
            {
                "instrument": str(r["other_instrument"]),
                "morphology_distance_score": float(r["morphology_distance_score"]),
                "centroid_delta": float(r["centroid_delta"]),
                "spread_delta": float(r["spread_delta"]),
                "attack_delta": float(r["attack_delta"]),
                "decay_delta": float(r["decay_delta"]),
            }
            for _, r in closest.iterrows()
        ],
        "furthest_instruments": [
            {
                "instrument": str(r["other_instrument"]),
                "morphology_distance_score": float(r["morphology_distance_score"]),
                "centroid_delta": float(r["centroid_delta"]),
                "spread_delta": float(r["spread_delta"]),
                "attack_delta": float(r["attack_delta"]),
                "decay_delta": float(r["decay_delta"]),
            }
            for _, r in furthest.iterrows()
        ],
    }

    summary = dict(data.get("summary", {}))
    summary["event_morphology_event_count"] = block["event_count"]
    summary["event_morphology_mean_duration_sec"] = block["mean_duration_sec"]
    summary["event_morphology_mean_attack_time_sec"] = block["mean_attack_time_sec"]
    summary["event_morphology_mean_decay_to_20pct_sec"] = block["mean_decay_to_20pct_sec"]
    summary["event_morphology_mean_spectral_centroid_hz"] = block["mean_spectral_centroid_hz"]
    summary["event_morphology_mean_spectral_spread_hz"] = block["mean_spectral_spread_hz"]
    data["summary"] = summary

    meanings = dict(data.get("meaning", {}))
    meanings["event_morphology_compare"] = (
        "Cross-instrument comparison of percussion event morphology: duration, attack, decay, centroid, spread, and resonance density."
    )
    data["meaning"] = meanings
    data["event_morphology_compare"] = block
    passport_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = passport_path.with_suffix(".md")
    if md_path.exists():
        text = md_path.read_text(encoding="utf-8", errors="replace")
        marker = "\n## Event morphology compare\n"
        if marker in text:
            text = text.split(marker, 1)[0].rstrip() + "\n"
        lines = []
        lines.append("## Event morphology compare")
        lines.append("")
        lines.append(f"- Event count: {block['event_count']}")
        lines.append(f"- Gesture types: {block['gesture_type_count']}")
        lines.append(f"- Dynamics: {block['dynamic_count']}")
        lines.append(f"- Articulations: {block['articulation_count']}")
        lines.append(f"- Mean duration sec: {block['mean_duration_sec']:.4f}")
        lines.append(f"- Mean attack sec: {block['mean_attack_time_sec']:.4f}")
        lines.append(f"- Mean decay-to-20% sec: {block['mean_decay_to_20pct_sec']:.4f}")
        lines.append(f"- Mean centroid Hz: {block['mean_spectral_centroid_hz']:.2f}")
        lines.append(f"- Mean spread Hz: {block['mean_spectral_spread_hz']:.2f}")
        lines.append(f"- Mean dense peaks: {block['mean_dense_peaks']:.2f}")
        lines.append(f"- Mean frequency clusters: {block['mean_frequency_clusters']:.2f}")
        lines.append("")
        lines.append("### Closest percussion instruments")
        lines.append("")
        lines.append("| instrument | distance | centroid Δ | spread Δ | attack Δ | decay Δ |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in block["closest_instruments"]:
            lines.append(
                f"| {row['instrument']} | {row['morphology_distance_score']:.6f} | {row['centroid_delta']:.2f} | "
                f"{row['spread_delta']:.2f} | {row['attack_delta']:.4f} | {row['decay_delta']:.4f} |"
            )
        lines.append("")
        lines.append("### Furthest percussion instruments")
        lines.append("")
        lines.append("| instrument | distance | centroid Δ | spread Δ | attack Δ | decay Δ |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in block["furthest_instruments"]:
            lines.append(
                f"| {row['instrument']} | {row['morphology_distance_score']:.6f} | {row['centroid_delta']:.2f} | "
                f"{row['spread_delta']:.2f} | {row['attack_delta']:.4f} | {row['decay_delta']:.4f} |"
            )
        md_path.write_text(text.rstrip() + "\n\n" + "\n".join(lines), encoding="utf-8")


def main() -> None:
    feat_df = pd.read_csv(FEATURE_CSV)
    pair_df = pd.read_csv(PAIRWISE_CSV)
    for passport_path in sorted(PASSPORT_DIR.glob("*__percussion_passport.json")):
        update_passport(passport_path, feat_df, pair_df)
        print(f"UPDATED {passport_path.name}")


if __name__ == "__main__":
    main()
