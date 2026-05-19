# -*- coding: utf-8 -*-
"""
PERCUSSION PASSPORT BUILDER

Собирает паспорта ударных по результатам percussion_event_pipeline_cli.

Вход:
  Block004_data/percussion/10_reports/percussion__event_pipeline_summary.csv
  + папки событий с:
      *_percussion_frequency_clusters.csv
      *_percussion_spectrum.png
      *_percussion_spiral.png

Выход:
  40_passports/
    percussion__family_passport.md
    percussion__family_passport.json
    <instrument_name>__percussion_passport.md
    <instrument_name>__percussion_passport.json
"""

import os
import json
import argparse
import pandas as pd


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_name(s):
    return str(s).replace(" ", "_").replace("/", "_").replace("\\", "_")


def load_csv_safe(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def find_cluster_file(event_dir):
    if not os.path.isdir(event_dir):
        return None

    for f in os.listdir(event_dir):
        if f.endswith("__percussion_frequency_clusters.csv"):
            return os.path.join(event_dir, f)

    return None


def collect_top_clusters(reports_root, event_stem, top_n=12):
    event_dir = os.path.join(reports_root, safe_name(event_stem))

    if not os.path.isdir(event_dir):
        return []

    cluster_path = find_cluster_file(event_dir)

    if not cluster_path:
        return []

    df = load_csv_safe(cluster_path)

    if df is None or len(df) == 0:
        return []

    sort_cols = []
    ascending = []

    if "frame_count" in df.columns:
        sort_cols.append("frame_count")
        ascending.append(False)

    if "mean_amp" in df.columns:
        sort_cols.append("mean_amp")
        ascending.append(False)

    if sort_cols:
        df = df.sort_values(by=sort_cols, ascending=ascending)

    records = []

    for _, r in df.head(top_n).iterrows():
        records.append(
            {
                "token": str(r.get("token", "")),
                "mean_freq_hz": float(r.get("mean_freq_hz", 0.0)),
                "frame_count": int(r.get("frame_count", 0)),
                "hit_count": int(r.get("hit_count", 0)),
                "mean_amp": float(r.get("mean_amp", 0.0)),
                "mean_rel_amp": float(r.get("mean_rel_amp", 0.0)),
            }
        )

    return records


def existing_or_empty(path):
    return path if os.path.exists(path) else ""


def build_instrument_passport(instrument_name, df, reports_root, out_dir):
    events = []
    all_clusters = []

    for _, r in df.iterrows():
        original_filename = str(r.get("original_filename", ""))
        event_stem = os.path.splitext(original_filename)[0]

        event_dir = os.path.join(reports_root, safe_name(event_stem))

        spectrum_png = os.path.join(event_dir, f"{event_stem}__percussion_spectrum.png")
        spiral_png = os.path.join(event_dir, f"{event_stem}__percussion_spiral.png")

        clusters = collect_top_clusters(reports_root, event_stem)

        event = {
            "original_filename": original_filename,
            "event_id": str(r.get("event_id", "")),
            "dynamic": str(r.get("dynamic", "")),
            "articulation": str(r.get("articulation", "")),
            "gesture_type": str(r.get("gesture_type", "")),
            "duration_sec": float(r.get("duration_sec", 0.0)),
            "attack_time_sec": float(r.get("attack_time_sec", 0.0)),
            "decay_to_20pct_sec": float(r.get("decay_to_20pct_sec", 0.0)),
            "peak_amp": float(r.get("peak_amp", 0.0)),
            "rms_amp": float(r.get("rms_amp", 0.0)),
            "spectral_centroid_hz": float(r.get("spectral_centroid_hz", 0.0)),
            "spectral_spread_hz": float(r.get("spectral_spread_hz", 0.0)),
            "dominant_freq_hz": float(r.get("dominant_freq_hz", 0.0)),
            "dominant_token": str(r.get("dominant_token", "")),
            "dense_peaks": int(r.get("dense_peaks", 0)),
            "frequency_clusters": int(r.get("frequency_clusters", 0)),
            "spectrum_png": existing_or_empty(spectrum_png),
            "spiral_png": existing_or_empty(spiral_png),
            "top_clusters": clusters,
        }

        events.append(event)

        for c in clusters:
            cc = dict(c)
            cc["event"] = original_filename
            all_clusters.append(cc)

    cluster_df = pd.DataFrame(all_clusters)

    if len(cluster_df) > 0:
        resonance_summary = (
            cluster_df.groupby("token")
            .agg(
                event_count=("event", "nunique"),
                mean_freq_hz=("mean_freq_hz", "mean"),
                avg_frame_count=("frame_count", "mean"),
                avg_hit_count=("hit_count", "mean"),
                mean_amp=("mean_amp", "mean"),
                mean_rel_amp=("mean_rel_amp", "mean"),
            )
            .reset_index()
            .sort_values(
                by=["event_count", "mean_amp"],
                ascending=[False, False],
            )
        )

        top_resonances = resonance_summary.head(30).to_dict(orient="records")
    else:
        top_resonances = []

    passport = {
        "instrument_family": "percussion",
        "instrument_name": instrument_name,
        "version": "percussion_passport_v2_events_with_visuals",
        "summary": {
            "event_count": int(len(events)),
            "gesture_types": (
                sorted(df["gesture_type"].dropna().astype(str).unique().tolist())
                if "gesture_type" in df.columns
                else []
            ),
            "dynamics": (
                sorted(df["dynamic"].dropna().astype(str).unique().tolist())
                if "dynamic" in df.columns
                else []
            ),
            "articulations": (
                sorted(df["articulation"].dropna().astype(str).unique().tolist())
                if "articulation" in df.columns
                else []
            ),
            "avg_duration_sec": (
                float(df["duration_sec"].mean())
                if "duration_sec" in df.columns
                else 0.0
            ),
            "avg_attack_time_sec": (
                float(df["attack_time_sec"].mean())
                if "attack_time_sec" in df.columns
                else 0.0
            ),
            "avg_decay_to_20pct_sec": (
                float(df["decay_to_20pct_sec"].mean())
                if "decay_to_20pct_sec" in df.columns
                else 0.0
            ),
            "avg_spectral_centroid_hz": (
                float(df["spectral_centroid_hz"].mean())
                if "spectral_centroid_hz" in df.columns
                else 0.0
            ),
            "avg_spectral_spread_hz": (
                float(df["spectral_spread_hz"].mean())
                if "spectral_spread_hz" in df.columns
                else 0.0
            ),
        },
        "top_resonances": top_resonances,
        "events": events,
    }

    base = safe_name(instrument_name)
    out_json = os.path.join(out_dir, f"{base}__percussion_passport.json")
    out_md = os.path.join(out_dir, f"{base}__percussion_passport.md")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(passport, f, ensure_ascii=False, indent=2)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write(f"# Percussion Passport: {instrument_name}\n\n")

        s = passport["summary"]

        f.write("## Summary\n\n")
        f.write(f"- Event count: {s['event_count']}\n")
        f.write(f"- Gesture types: {', '.join(s['gesture_types'])}\n")
        f.write(f"- Dynamics: {', '.join(s['dynamics'])}\n")
        f.write(f"- Articulations: {', '.join(s['articulations'])}\n")
        f.write(f"- Avg duration sec: {s['avg_duration_sec']:.4f}\n")
        f.write(f"- Avg attack time sec: {s['avg_attack_time_sec']:.4f}\n")
        f.write(f"- Avg decay to 20% sec: {s['avg_decay_to_20pct_sec']:.4f}\n")
        f.write(f"- Avg spectral centroid Hz: {s['avg_spectral_centroid_hz']:.2f}\n")
        f.write(f"- Avg spectral spread Hz: {s['avg_spectral_spread_hz']:.2f}\n\n")

        f.write("## Top resonances\n\n")
        f.write("| token | Hz | events | mean amp | rel amp |\n")
        f.write("|---|---:|---:|---:|---:|\n")

        for r in top_resonances[:25]:
            f.write(
                f"| {r.get('token', '')} "
                f"| {float(r.get('mean_freq_hz', 0.0)):.2f} "
                f"| {int(r.get('event_count', 0))} "
                f"| {float(r.get('mean_amp', 0.0)):.6f} "
                f"| {float(r.get('mean_rel_amp', 0.0)):.6f} |\n"
            )

        f.write("\n## Events\n\n")
        f.write(
            "| file | event | dynamic | articulation | gesture | dominant | "
            "centroid Hz | attack | decay | spiral | spectrum |\n"
        )
        f.write(
            "|---|---|---|---|---|---|---:|---:|---:|---|---|\n"
        )

        for e in events:
            spiral_link = e.get("spiral_png", "")
            spectrum_link = e.get("spectrum_png", "")

            if spiral_link:
                spiral_cell = f"[spiral]({spiral_link})"
            else:
                spiral_cell = ""

            if spectrum_link:
                spectrum_cell = f"[spectrum]({spectrum_link})"
            else:
                spectrum_cell = ""

            f.write(
                f"| {e['original_filename']} "
                f"| {e['event_id']} "
                f"| {e['dynamic']} "
                f"| {e['articulation']} "
                f"| {e['gesture_type']} "
                f"| {e['dominant_token']} "
                f"| {e['spectral_centroid_hz']:.2f} "
                f"| {e['attack_time_sec']:.4f} "
                f"| {e['decay_to_20pct_sec']:.4f} "
                f"| {spiral_cell} "
                f"| {spectrum_cell} |\n"
            )

    return passport


def build_family_passport(passports, out_dir):
    family = {
        "instrument_family": "percussion",
        "version": "percussion_family_passport_v2_with_visuals",
        "instrument_count": len(passports),
        "instruments": [],
    }

    for p in passports:
        family["instruments"].append(
            {
                "instrument_name": p["instrument_name"],
                "event_count": p["summary"]["event_count"],
                "avg_duration_sec": p["summary"]["avg_duration_sec"],
                "avg_attack_time_sec": p["summary"]["avg_attack_time_sec"],
                "avg_decay_to_20pct_sec": p["summary"]["avg_decay_to_20pct_sec"],
                "avg_spectral_centroid_hz": p["summary"]["avg_spectral_centroid_hz"],
                "avg_spectral_spread_hz": p["summary"]["avg_spectral_spread_hz"],
                "gesture_types": p["summary"]["gesture_types"],
                "top_resonance_tokens": [
                    r.get("token", "") for r in p.get("top_resonances", [])[:10]
                ],
                "passport_json": os.path.join(
                    out_dir,
                    f"{safe_name(p['instrument_name'])}__percussion_passport.json",
                ),
                "passport_md": os.path.join(
                    out_dir,
                    f"{safe_name(p['instrument_name'])}__percussion_passport.md",
                ),
            }
        )

    out_json = os.path.join(out_dir, "percussion__family_passport.json")
    out_md = os.path.join(out_dir, "percussion__family_passport.md")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(family, f, ensure_ascii=False, indent=2)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Percussion Family Passport\n\n")
        f.write(f"- Instrument count: {family['instrument_count']}\n\n")

        f.write(
            "| instrument | events | centroid Hz | attack | decay | "
            "top resonance tokens | passport |\n"
        )
        f.write("|---|---:|---:|---:|---:|---|---|\n")

        for i in family["instruments"]:
            passport_md = i.get("passport_md", "")
            passport_cell = f"[md]({passport_md})" if passport_md else ""

            f.write(
                f"| {i['instrument_name']} "
                f"| {i['event_count']} "
                f"| {i['avg_spectral_centroid_hz']:.2f} "
                f"| {i['avg_attack_time_sec']:.4f} "
                f"| {i['avg_decay_to_20pct_sec']:.4f} "
                f"| {', '.join(i['top_resonance_tokens'])} "
                f"| {passport_cell} |\n"
            )


def main():
    parser = argparse.ArgumentParser(description="Build percussion passports.")

    parser.add_argument("--reports_root", required=True)
    parser.add_argument("--out_dir", required=True)

    args = parser.parse_args()

    ensure_dir(args.out_dir)

    summary_csv = os.path.join(
        args.reports_root,
        "percussion__event_pipeline_summary.csv",
    )

    df = load_csv_safe(summary_csv)

    if df is None or len(df) == 0:
        raise RuntimeError(f"Empty or missing summary CSV: {summary_csv}")

    if "instrument_name" not in df.columns:
        raise RuntimeError("summary CSV has no instrument_name column")

    passports = []

    for instrument_name, sub in df.groupby("instrument_name"):
        passport = build_instrument_passport(
            instrument_name=instrument_name,
            df=sub,
            reports_root=args.reports_root,
            out_dir=args.out_dir,
        )
        passports.append(passport)

    build_family_passport(passports, args.out_dir)

    print("PERCUSSION PASSPORT BUILDER DONE")
    print(f"reports_root     : {args.reports_root}")
    print(f"out_dir          : {args.out_dir}")
    print(f"passports built  : {len(passports)}")


if __name__ == "__main__":
    main()