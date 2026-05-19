# -*- coding: utf-8 -*-
import os
import argparse
import pandas as pd


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def normalize(df):
    ren = {}
    if "freq_hz" in df.columns and "hz" not in df.columns:
        ren["freq_hz"] = "hz"
    if "amplitude" in df.columns and "amp" not in df.columns:
        ren["amplitude"] = "amp"
    if "frame_index" in df.columns and "frame_idx" not in df.columns:
        ren["frame_index"] = "frame_idx"
    if "note_token" not in df.columns and "token" in df.columns:
        ren["token"] = "note_token"
    return df.rename(columns=ren)


def note_from_folder(name):
    # 001_5.A- -> 5.A-
    parts = name.split("_", 1)
    if len(parts) == 2:
        return parts[1]
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument_name", required=True)
    ap.add_argument("--reports_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_percent_notes", type=float, default=2.0)
    ap.add_argument("--breath_max_hz", type=float, default=80.0)
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    rows = []

    for d in os.listdir(args.reports_root):
        note_dir = os.path.join(args.reports_root, d)
        if not os.path.isdir(note_dir):
            continue

        dense_path = None
        for f in os.listdir(note_dir):
            if f.endswith("__dense.csv"):
                dense_path = os.path.join(note_dir, f)
                break

        if not dense_path:
            continue

        df = load_csv(dense_path)
        if df is None or len(df) == 0:
            continue

        df = normalize(df)

        if "hz" not in df.columns:
            continue

        if "amp" not in df.columns:
            df["amp"] = 0.0

        if "note_token" not in df.columns:
            df["note_token"] = df["hz"].round(1).astype(str)

        expected_note = note_from_folder(d)

        for _, r in df.iterrows():
            rows.append({
                "source_note_dir": d,
                "expected_note": expected_note,
                "token": str(r.get("note_token", "")),
                "hz": float(r.get("hz", 0.0)),
                "amp": float(r.get("amp", 0.0)),
            })

    all_df = pd.DataFrame(rows)

    if len(all_df) == 0:
        raise RuntimeError("No dense rows found in reports_root")

    total_notes = all_df["expected_note"].nunique()

    global_presence = (
        all_df.groupby("token")
        .agg(
            note_count=("expected_note", "nunique"),
            mean_hz=("hz", "mean"),
            mean_amp=("amp", "mean"),
            median_amp=("amp", "median"),
            max_amp=("amp", "max"),
        )
        .reset_index()
    )

    global_presence["percent_notes"] = (
        global_presence["note_count"] / max(1, total_notes) * 100.0
    )

    global_presence = global_presence.sort_values(
        by=["note_count", "mean_amp"],
        ascending=[False, False],
    )

    dense_frequency_clusters = global_presence.copy()
    dense_frequency_clusters = dense_frequency_clusters.rename(
        columns={
            "mean_hz": "cluster_hz",
            "token": "cluster_token",
        }
    )

    box_all = dense_frequency_clusters[
        dense_frequency_clusters["percent_notes"] >= args.min_percent_notes
    ].copy()

    box_breath = box_all[box_all["cluster_hz"] <= args.breath_max_hz].copy()
    box_resonance = box_all[box_all["cluster_hz"] > args.breath_max_hz].copy()

    range_presence = (
        all_df.groupby(["expected_note", "token"])
        .agg(
            hit_count=("token", "count"),
            mean_hz=("hz", "mean"),
            mean_amp=("amp", "mean"),
        )
        .reset_index()
        .sort_values(by=["expected_note", "hit_count"], ascending=[True, False])
    )

    prefix = args.instrument_name

    dense_frequency_clusters.to_csv(
        os.path.join(args.out_dir, f"{prefix}__dense_frequency_clusters.csv"),
        index=False,
    )

    global_presence.to_csv(
        os.path.join(args.out_dir, f"{prefix}__dense_global_presence.csv"),
        index=False,
    )

    range_presence.to_csv(
        os.path.join(args.out_dir, f"{prefix}__dense_range_presence.csv"),
        index=False,
    )

    box_all.to_csv(
        os.path.join(args.out_dir, f"{prefix}__box_all.csv"),
        index=False,
    )

    box_breath.to_csv(
        os.path.join(args.out_dir, f"{prefix}__box_breath.csv"),
        index=False,
    )

    box_resonance.to_csv(
        os.path.join(args.out_dir, f"{prefix}__box_resonance.csv"),
        index=False,
    )

    with open(
        os.path.join(args.out_dir, f"{prefix}__dense_box_summary.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write("RANGE RESEARCH FROM EXISTING 10_REPORTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"instrument_name      : {prefix}\n")
        f.write(f"reports_root         : {args.reports_root}\n")
        f.write(f"total_notes          : {total_notes}\n")
        f.write(f"total_dense_rows     : {len(all_df)}\n")
        f.write(f"global_tokens        : {len(global_presence)}\n")
        f.write(f"box_all_components   : {len(box_all)}\n")
        f.write(f"box_breath_components: {len(box_breath)}\n")
        f.write(f"box_res_components   : {len(box_resonance)}\n\n")
        f.write("TOP TOKENS\n")
        f.write("-" * 80 + "\n")
        for _, r in global_presence.head(40).iterrows():
            f.write(
                f"{str(r['token']):14} "
                f"notes={int(r['note_count']):4d} "
                f"percent={r['percent_notes']:.2f} "
                f"hz={r['mean_hz']:.3f} "
                f"mean_amp={r['mean_amp']:.6f}\n"
            )

    print("RANGE RESEARCH BUILT")
    print(f"instrument_name : {prefix}")
    print(f"total_notes     : {total_notes}")
    print(f"out_dir         : {args.out_dir}")


if __name__ == "__main__":
    main()