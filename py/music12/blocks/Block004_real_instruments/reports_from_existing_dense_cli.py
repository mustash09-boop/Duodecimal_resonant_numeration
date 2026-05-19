# -*- coding: utf-8 -*-
"""
REPORTS FROM EXISTING DENSE

Достраивает недостающий слой отчётов из уже существующих __dense.csv.

Создаёт стандартные файлы:
- __dense_unified_clean.csv
- __dense_unified_clean_summary.txt
- __dense_unified_removed_box.csv
- __root_consensus_candidates.csv
- __root_consensus_clusters.csv
- __root_consensus_meta.json
- __root_consensus_summary.txt
- __spiral12_clean_points.csv
- __spiral12_clean.png

Важно:
это совместимый восстановительный слой, а не повтор полного WAV-сканирования.
"""

import os
import json
import math
import argparse
import pandas as pd
import matplotlib.pyplot as plt


DIGITS12 = "123456789ABC"


def load_csv_safe(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def find_dense_file(note_dir):
    for f in os.listdir(note_dir):
        if f.endswith("__dense.csv"):
            return os.path.join(note_dir, f)
    return None


def output_prefix_from_dense(dense_path):
    name = os.path.basename(dense_path)
    return name[:-len("__dense.csv")]


def normalize_dense(df):
    ren = {}

    if "freq_hz" in df.columns and "hz" not in df.columns:
        ren["freq_hz"] = "hz"

    if "amplitude" in df.columns and "amp" not in df.columns:
        ren["amplitude"] = "amp"

    if "frame_index" in df.columns and "frame_idx" not in df.columns:
        ren["frame_index"] = "frame_idx"

    if ren:
        df = df.rename(columns=ren)

    if "time_sec" not in df.columns:
        if "frame_idx" in df.columns:
            df["time_sec"] = df["frame_idx"].astype(float)
        else:
            df["time_sec"] = range(len(df))

    if "note_token" not in df.columns:
        df["note_token"] = ""

    if "amp" not in df.columns:
        df["amp"] = 0.0

    return df


def cents_diff(hz1, hz2):
    if hz1 <= 0 or hz2 <= 0:
        return 9999.0
    return 1200.0 * math.log2(hz1 / hz2)


def token_to_spiral_xy(token):
    token = str(token).strip()
    clean = token.split("'")[0].replace("-", "")

    if "." not in clean:
        return None

    try:
        octave_s, degree_s = clean.split(".", 1)
        octave = int(octave_s)
        degree = DIGITS12.index(degree_s[0]) + 1
    except Exception:
        return None

    angle = (degree - 1) * (2.0 * math.pi / 12.0)
    radius = octave + degree / 12.0

    return radius * math.cos(angle), radius * math.sin(angle)


def estimate_root_from_dense(df):
    """
    Мягкая оценка root для совместимости.
    Берём самый устойчивый/сильный нижний кандидат.
    """
    if len(df) == 0:
        return None, ""

    d = df.copy()
    d = d[d["hz"] > 20]

    if len(d) == 0:
        return None, ""

    # Предпочитаем низкие частоты с большой амплитудой
    d["score"] = d["amp"] / (d["hz"] ** 0.35)
    top = d.sort_values("score", ascending=False).head(20)

    root_hz = float(top["hz"].median())

    if "note_token" in top.columns:
        root_token = str(top.iloc[0].get("note_token", ""))
    else:
        root_token = ""

    return root_hz, root_token


def build_root_candidates(df, root_hz, root_token):
    d = df.copy()
    d["root_hz_candidate"] = root_hz
    d["root_note_token"] = root_token
    d["candidate_score"] = d["amp"]
    return d[[
        c for c in [
            "time_sec",
            "frame_idx",
            "hz",
            "amp",
            "note_token",
            "root_hz_candidate",
            "root_note_token",
            "candidate_score",
        ]
        if c in d.columns
    ]]


def build_root_clusters(df, root_hz, root_token):
    return pd.DataFrame([{
        "cluster_id": 1,
        "consensus_root_hz": root_hz,
        "consensus_root_token": root_token,
        "member_count": int(len(df)),
        "unique_frame_count": int(df["frame_idx"].nunique()) if "frame_idx" in df.columns else 0,
        "mean_observed_amplitude": float(df["amp"].mean()) if "amp" in df.columns else 0.0,
        "tuner_confidence": float(df["amp"].mean()) if "amp" in df.columns else 0.0,
    }])


def save_root_summary(path, root_hz, root_token, df):
    with open(path, "w", encoding="utf-8") as f:
        f.write("ROOT FROM EXISTING DENSE\n")
        f.write("=" * 80 + "\n")
        f.write(f"consensus_root_hz               : {root_hz}\n")
        f.write(f"consensus_root_token            : {root_token}\n")
        f.write(f"member_count                    : {len(df)}\n")
        if "frame_idx" in df.columns:
            f.write(f"unique_frame_count              : {df['frame_idx'].nunique()}\n")


def save_spiral(df, out_csv, out_png):
    rows = []

    for _, r in df.iterrows():
        xy = token_to_spiral_xy(r.get("note_token", ""))

        if not xy:
            continue

        rows.append({
            "time_sec": float(r.get("time_sec", 0.0)),
            "frame_idx": int(r.get("frame_idx", 0)) if "frame_idx" in df.columns else 0,
            "x12": float(xy[0]),
            "y12": float(xy[1]),
            "freq_hz": float(r.get("hz", 0.0)),
            "amplitude": float(r.get("amp", 0.0)),
            "note_token": str(r.get("note_token", "")),
        })

    spiral_df = pd.DataFrame(rows)

    if len(spiral_df) == 0:
        spiral_df.to_csv(out_csv, index=False)
        return

    spiral_df.to_csv(out_csv, index=False)

    plt.figure(figsize=(7, 7))
    plt.scatter(
        spiral_df["x12"],
        spiral_df["y12"],
        s=8,
        alpha=0.45,
    )
    plt.title("spiral12 clean")
    plt.axis("equal")
    plt.grid(True, alpha=0.25)
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()


def process_note_dir(note_dir):
    dense_path = find_dense_file(note_dir)

    if not dense_path:
        return False

    prefix = output_prefix_from_dense(dense_path)

    df = load_csv_safe(dense_path)

    if df is None or len(df) == 0:
        return False

    df = normalize_dense(df)

    if "hz" not in df.columns:
        return False

    root_hz, root_token = estimate_root_from_dense(df)

    if root_hz is None:
        return False

    # clean layer: пока сохраняем dense как unified clean
    clean_csv = os.path.join(note_dir, f"{prefix}__dense_unified_clean.csv")
    removed_csv = os.path.join(note_dir, f"{prefix}__dense_unified_removed_box.csv")
    clean_txt = os.path.join(note_dir, f"{prefix}__dense_unified_clean_summary.txt")

    df.to_csv(clean_csv, index=False)
    pd.DataFrame(columns=df.columns).to_csv(removed_csv, index=False)

    with open(clean_txt, "w", encoding="utf-8") as f:
        f.write("DENSE UNIFIED CLEAN FROM EXISTING DENSE\n")
        f.write("=" * 80 + "\n")
        f.write(f"source_dense_csv : {dense_path}\n")
        f.write(f"total_rows       : {len(df)}\n")
        f.write(f"kept_rows        : {len(df)}\n")
        f.write("removed_rows     : 0\n")

    # root layer
    candidates = build_root_candidates(df, root_hz, root_token)
    clusters = build_root_clusters(df, root_hz, root_token)

    candidates.to_csv(
        os.path.join(note_dir, f"{prefix}__root_consensus_candidates.csv"),
        index=False,
    )

    clusters.to_csv(
        os.path.join(note_dir, f"{prefix}__root_consensus_clusters.csv"),
        index=False,
    )

    meta = {
        "source_dense_csv": dense_path,
        "semantic_note": "Recovered compatible root layer from existing dense report.",
        "consensus_root_hz": root_hz,
        "consensus_root_token": root_token,
    }

    with open(
        os.path.join(note_dir, f"{prefix}__root_consensus_meta.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    save_root_summary(
        os.path.join(note_dir, f"{prefix}__root_consensus_summary.txt"),
        root_hz,
        root_token,
        df,
    )

    # spiral layer
    save_spiral(
        df,
        os.path.join(note_dir, f"{prefix}__spiral12_clean_points.csv"),
        os.path.join(note_dir, f"{prefix}__spiral12_clean.png"),
    )

    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports_root", required=True)
    args = ap.parse_args()

    ok = 0
    skipped = 0

    for d in os.listdir(args.reports_root):
        note_dir = os.path.join(args.reports_root, d)

        if not os.path.isdir(note_dir):
            continue

        if process_note_dir(note_dir):
            ok += 1
        else:
            skipped += 1

    print("REPORTS FROM EXISTING DENSE DONE")
    print(f"reports_root : {args.reports_root}")
    print(f"processed    : {ok}")
    print(f"skipped      : {skipped}")


if __name__ == "__main__":
    main()