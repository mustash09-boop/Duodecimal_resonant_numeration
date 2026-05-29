from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(r"E:\Duodecimal_resonant_numeration")
BLOCK004_ROOT = PROJECT_ROOT / "Block004_data"
MORPH_ROOT = BLOCK004_ROOT / "_multi_instrument_compare" / "91_harmonic_morphology_batch"
FEATURES_CSV = MORPH_ROOT / "all_morphology_features.csv"
PAIRWISE_SUMMARY_CSV = MORPH_ROOT / "all_pairwise_morphology_summary.csv"


def tonal_dataset_dirs() -> list[Path]:
    out: list[Path] = []
    for d in sorted(BLOCK004_ROOT.iterdir()):
        if not d.is_dir():
            continue
        if d.name.startswith("_") or d.name == "percussion":
            continue
        if (d / "20_range_research").exists():
            out.append(d)
    return out


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_morphology_block(instrument: str, features_df: pd.DataFrame, pair_df: pd.DataFrame) -> dict:
    sub = features_df[features_df["instrument"] == instrument].copy()
    if len(sub) == 0:
        return {
            "notes_compared": 0,
            "feature_rows": 0,
            "mean_peak_time": 0.0,
            "mean_attack_energy": 0.0,
            "mean_sustain_energy": 0.0,
            "mean_tail_energy": 0.0,
            "mean_active_ratio": 0.0,
            "mean_roughness": 0.0,
            "top_harmonics_by_attack_energy": [],
            "top_harmonics_by_sustain_energy": [],
            "closest_instruments": [],
            "furthest_instruments": [],
        }

    harm_agg = (
        sub.groupby("harmonic", as_index=False)
        .agg(
            mean_peak_amp=("peak_amp", "mean"),
            mean_peak_time=("peak_time", "mean"),
            mean_attack_energy=("attack_energy", "mean"),
            mean_sustain_energy=("sustain_energy", "mean"),
            mean_tail_energy=("tail_energy", "mean"),
            mean_active_ratio=("active_ratio", "mean"),
            mean_roughness=("roughness", "mean"),
            notes_seen=("note", "nunique"),
        )
        .sort_values("harmonic")
    )

    pair_rows = []
    if len(pair_df):
        a = pair_df[pair_df["instrument_a"] == instrument].copy()
        a["other_instrument"] = a["instrument_b"]
        b = pair_df[pair_df["instrument_b"] == instrument].copy()
        b["other_instrument"] = b["instrument_a"]
        pair_rows = [a, b]

    if pair_rows:
        pair_sub = pd.concat(pair_rows, ignore_index=True)
        pair_agg = (
            pair_sub.groupby("other_instrument", as_index=False)
            .agg(
                notes_compared=("note", "nunique"),
                mean_morphology_distance_score=("morphology_distance_score", "mean"),
                max_morphology_distance_score=("morphology_distance_score", "max"),
                mean_shape_l2=("mean_shape_l2", "mean"),
                mean_corr_distance=("mean_corr_distance", "mean"),
            )
            .sort_values("mean_morphology_distance_score", ascending=True)
        )
    else:
        pair_agg = pd.DataFrame()

    return {
        "notes_compared": int(sub["note"].nunique()),
        "feature_rows": int(len(sub)),
        "mean_peak_time": float(sub["peak_time"].mean()),
        "mean_attack_energy": float(sub["attack_energy"].mean()),
        "mean_sustain_energy": float(sub["sustain_energy"].mean()),
        "mean_tail_energy": float(sub["tail_energy"].mean()),
        "mean_active_ratio": float(sub["active_ratio"].mean()),
        "mean_roughness": float(sub["roughness"].mean()),
        "top_harmonics_by_attack_energy": [
            {
                "harmonic": int(r["harmonic"]),
                "mean_attack_energy": float(r["mean_attack_energy"]),
                "mean_peak_amp": float(r["mean_peak_amp"]),
                "notes_seen": int(r["notes_seen"]),
            }
            for _, r in harm_agg.sort_values("mean_attack_energy", ascending=False).head(8).iterrows()
        ],
        "top_harmonics_by_sustain_energy": [
            {
                "harmonic": int(r["harmonic"]),
                "mean_sustain_energy": float(r["mean_sustain_energy"]),
                "mean_peak_time": float(r["mean_peak_time"]),
                "notes_seen": int(r["notes_seen"]),
            }
            for _, r in harm_agg.sort_values("mean_sustain_energy", ascending=False).head(8).iterrows()
        ],
        "closest_instruments": [
            {
                "instrument": str(r["other_instrument"]),
                "notes_compared": int(r["notes_compared"]),
                "mean_morphology_distance_score": float(r["mean_morphology_distance_score"]),
                "mean_shape_l2": float(r["mean_shape_l2"]),
                "mean_corr_distance": float(r["mean_corr_distance"]),
            }
            for _, r in pair_agg.head(6).iterrows()
        ],
        "furthest_instruments": [
            {
                "instrument": str(r["other_instrument"]),
                "notes_compared": int(r["notes_compared"]),
                "mean_morphology_distance_score": float(r["mean_morphology_distance_score"]),
                "mean_shape_l2": float(r["mean_shape_l2"]),
                "mean_corr_distance": float(r["mean_corr_distance"]),
            }
            for _, r in pair_agg.sort_values("mean_morphology_distance_score", ascending=False).head(6).iterrows()
        ],
    }


def update_markdown(md_path: Path, morphology: dict) -> None:
    text = md_path.read_text(encoding="utf-8", errors="replace")
    marker_start = "\n## Harmonic morphology compare\n"
    if marker_start in text:
        text = text.split(marker_start, 1)[0].rstrip() + "\n"

    lines = []
    lines.append("## Harmonic morphology compare")
    lines.append("")
    lines.append("This section summarizes cross-instrument harmonic morphology for notes shared with other tonal instruments.")
    lines.append("")
    lines.append(f"- Shared notes compared: {morphology['notes_compared']}")
    lines.append(f"- Feature rows: {morphology['feature_rows']}")
    lines.append(f"- Mean peak time: {morphology['mean_peak_time']:.4f}")
    lines.append(f"- Mean attack energy: {morphology['mean_attack_energy']:.6f}")
    lines.append(f"- Mean sustain energy: {morphology['mean_sustain_energy']:.6f}")
    lines.append(f"- Mean tail energy: {morphology['mean_tail_energy']:.6f}")
    lines.append(f"- Mean active ratio: {morphology['mean_active_ratio']:.6f}")
    lines.append(f"- Mean roughness: {morphology['mean_roughness']:.6f}")
    lines.append("")
    lines.append("### Top harmonics by attack energy")
    lines.append("")
    lines.append("| harmonic | attack energy | peak amp | notes seen |")
    lines.append("|---|---:|---:|---:|")
    for row in morphology["top_harmonics_by_attack_energy"]:
        lines.append(
            f"| {row['harmonic']} | {row['mean_attack_energy']:.6f} | {row['mean_peak_amp']:.6f} | {row['notes_seen']} |"
        )
    lines.append("")
    lines.append("### Top harmonics by sustain energy")
    lines.append("")
    lines.append("| harmonic | sustain energy | peak time | notes seen |")
    lines.append("|---|---:|---:|---:|")
    for row in morphology["top_harmonics_by_sustain_energy"]:
        lines.append(
            f"| {row['harmonic']} | {row['mean_sustain_energy']:.6f} | {row['mean_peak_time']:.4f} | {row['notes_seen']} |"
        )
    lines.append("")
    lines.append("### Closest instruments by harmonic morphology")
    lines.append("")
    lines.append("| instrument | notes | distance | shape l2 | corr distance |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in morphology["closest_instruments"]:
        lines.append(
            f"| {row['instrument']} | {row['notes_compared']} | {row['mean_morphology_distance_score']:.6f} | "
            f"{row['mean_shape_l2']:.6f} | {row['mean_corr_distance']:.6f} |"
        )
    lines.append("")
    lines.append("### Furthest instruments by harmonic morphology")
    lines.append("")
    lines.append("| instrument | notes | distance | shape l2 | corr distance |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in morphology["furthest_instruments"]:
        lines.append(
            f"| {row['instrument']} | {row['notes_compared']} | {row['mean_morphology_distance_score']:.6f} | "
            f"{row['mean_shape_l2']:.6f} | {row['mean_corr_distance']:.6f} |"
        )
    lines.append("")

    md_path.write_text(text.rstrip() + "\n\n" + "\n".join(lines), encoding="utf-8")


def main() -> None:
    if not FEATURES_CSV.exists() or not PAIRWISE_SUMMARY_CSV.exists():
        raise FileNotFoundError("Morphology batch outputs are missing. Run build_block004_harmonic_morphology_batch.py first.")

    features_df = pd.read_csv(FEATURES_CSV)
    pair_df = pd.read_csv(PAIRWISE_SUMMARY_CSV)

    for dataset_dir in tonal_dataset_dirs():
        range_dir = dataset_dir / "20_range_research"
        json_candidates = sorted(range_dir.glob("*__instrument_passport.json"))
        md_candidates = sorted(range_dir.glob("*__instrument_passport.md"))
        if not json_candidates or not md_candidates:
            continue

        instrument = dataset_dir.name
        passport_json = json_candidates[0]
        passport_md = md_candidates[0]

        morphology = build_morphology_block(instrument=instrument, features_df=features_df, pair_df=pair_df)
        data = load_json(passport_json)

        meaning = dict(data.get("meaning", {}))
        meaning["harmonic_morphology_compare"] = (
            "Cross-instrument comparison of harmonic time-shape behavior across shared notes."
        )
        data["meaning"] = meaning

        summary = dict(data.get("summary", {}))
        summary["harmonic_morphology_notes_compared"] = morphology["notes_compared"]
        summary["harmonic_morphology_feature_rows"] = morphology["feature_rows"]
        summary["harmonic_morphology_mean_attack_energy"] = morphology["mean_attack_energy"]
        summary["harmonic_morphology_mean_sustain_energy"] = morphology["mean_sustain_energy"]
        summary["harmonic_morphology_mean_tail_energy"] = morphology["mean_tail_energy"]
        summary["harmonic_morphology_mean_active_ratio"] = morphology["mean_active_ratio"]
        summary["harmonic_morphology_mean_roughness"] = morphology["mean_roughness"]
        data["summary"] = summary

        data["harmonic_morphology_compare"] = morphology
        save_json(passport_json, data)
        update_markdown(passport_md, morphology)
        print(f"UPDATED {instrument}")


if __name__ == "__main__":
    main()
