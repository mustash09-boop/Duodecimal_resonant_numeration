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
import re
import csv
import pandas as pd
import matplotlib.pyplot as plt

from music12.core.pdf_spiral12_xy import (
    pdf_spiral_xy_from_frequency,
    pdf_spiral_xy_from_token,
    token_to_abs_step,
)


DIGITS12 = "123456789ABC"
NOTE12_RE = re.compile(r"([1-9ABC]+[.][1-9ABC]-?)", re.IGNORECASE)
MANIFEST_EXPECTED_NOTE_CACHE = {}


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


def output_prefix_from_note_dir(note_dir, dense_path):
    note_name = os.path.basename(str(note_dir)).strip()
    if note_name:
        return note_name
    return output_prefix_from_dense(dense_path)


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


def extract_expected_note_token(note_dir_name):
    m = NOTE12_RE.search(str(note_dir_name))
    if not m:
        return ""
    return m.group(1).replace("'", "")


def expected_note_from_manifest(note_dir):
    note_dir = os.path.abspath(str(note_dir))
    cached = MANIFEST_EXPECTED_NOTE_CACHE.get(note_dir)
    if cached is not None:
        return cached

    note_name = os.path.basename(note_dir)
    instrument_root = os.path.dirname(os.path.dirname(note_dir))
    manifest_dir = os.path.join(instrument_root, "20_manifest")
    source_name = f"{note_name}.wav"

    expected = ""
    if os.path.isdir(manifest_dir):
        for f in os.listdir(manifest_dir):
            if not f.endswith(".csv"):
                continue
            manifest_path = os.path.join(manifest_dir, f)
            try:
                with open(manifest_path, "r", encoding="utf-8-sig", newline="") as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        wav_name = (
                            row.get("original_filename")
                            or row.get("wav_name")
                            or row.get("source_file")
                            or row.get("filename")
                            or ""
                        ).strip()
                        if wav_name == source_name:
                            expected = (
                                str(
                                    row.get("note12")
                                    or row.get("note_token")
                                    or row.get("source_note")
                                    or ""
                                )
                                .replace("'", "")
                                .strip()
                            )
                            if expected:
                                MANIFEST_EXPECTED_NOTE_CACHE[note_dir] = expected
                                return expected
            except Exception:
                continue

    MANIFEST_EXPECTED_NOTE_CACHE[note_dir] = expected
    return expected


def token_to_hz(token, anchor_token, anchor_hz):
    if not token:
        return 0.0
    abs_step = float(token_to_abs_step(token))
    anchor_abs = float(token_to_abs_step(anchor_token))
    semitone_offset = abs_step - anchor_abs
    return float(anchor_hz) * (2.0 ** (semitone_offset / 12.0))


def root_token_from_expected_identity(freq_hz, expected_note, anchor_token, anchor_hz, micro_steps_per_semitone=12):
    expected_note = str(expected_note or "").replace("'", "").strip()
    if not expected_note or freq_hz <= 0:
        return ""
    expected_base = expected_note[:-1] if expected_note.endswith("-") else expected_note

    expected_hz = token_to_hz(expected_note, anchor_token, anchor_hz)
    if expected_hz <= 0:
        return ""

    semitone_delta = 12.0 * math.log2(float(freq_hz) / float(expected_hz))
    micro = int(round(semitone_delta * micro_steps_per_semitone))

    if micro == 0:
        return f"{expected_base}'-"

    sign = "i" if micro > 0 else "a"
    mag = abs(micro)

    if mag >= micro_steps_per_semitone:
        return ""

    return f"{expected_base}'{sign}{DIGITS12[mag - 1]}"


def weighted_median(pairs):
    if not pairs:
        return None
    pairs = sorted((float(v), float(w)) for v, w in pairs if float(w) > 0.0)
    if not pairs:
        return None
    total = sum(w for _, w in pairs)
    acc = 0.0
    for value, weight in pairs:
        acc += weight
        if acc >= total * 0.5:
            return value
    return pairs[-1][0]


def estimate_root_from_dense(df, expected_note="", anchor_token="9.A-", anchor_hz=440.0):
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

    expected_hz = token_to_hz(expected_note, anchor_token, anchor_hz) if expected_note else 0.0

    if expected_hz > 0.0:
        supported_roots = []
        for _, row in d.iterrows():
            hz = float(row.get("hz", 0.0))
            amp = float(row.get("amp", 0.0))
            if hz <= 0.0 or amp <= 0.0:
                continue
            for h in range(1, 13):
                candidate = hz / h
                delta_cents = abs(cents_diff(candidate, expected_hz))
                if delta_cents <= 80.0:
                    weight = (amp / (h ** 0.65)) * max(0.05, 1.0 - (delta_cents / 80.0))
                    supported_roots.append((candidate, weight))

        refined = weighted_median(supported_roots)
        if refined is not None and refined > 0.0:
            root_hz = float(refined)
            root_token = root_token_from_expected_identity(root_hz, expected_note, anchor_token, anchor_hz)
            if not root_token:
                root_token = pdf_spiral_xy_from_frequency(
                    root_hz,
                    anchor_token=anchor_token,
                    anchor_hz=float(anchor_hz),
                ).note_token
            return root_hz, root_token

    # Fallback: strongest low candidate, but still robust to a single very loud partial
    d["score"] = d["amp"] / (d["hz"] ** 0.35)
    top = d.sort_values("score", ascending=False).head(40)
    root_hz = float(top["hz"].median())
    root_token = pdf_spiral_xy_from_frequency(
        root_hz,
        anchor_token=anchor_token,
        anchor_hz=float(anchor_hz),
    ).note_token

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


def harmonic_role(freq_hz, root_hz, tolerance_cents=28.0, harmonic_max=12):
    if root_hz <= 0.0 or freq_hz <= 0.0:
        return False, 0, 9999.0
    best_h = 0
    best_delta = 9999.0
    for h in range(1, harmonic_max + 1):
        expected = root_hz * h
        delta = abs(cents_diff(freq_hz, expected))
        if delta < best_delta:
            best_delta = delta
            best_h = h
    return best_delta <= tolerance_cents, best_h, best_delta


def save_spiral(df, out_csv, out_png, anchor_token, anchor_hz, root_hz=None):
    rows = []

    for _, r in df.iterrows():
        freq_hz = float(r.get("hz", 0.0))
        note_token = str(r.get("note_token", ""))
        coords = None

        if freq_hz > 0:
            coords = pdf_spiral_xy_from_frequency(
                freq_hz,
                anchor_token=anchor_token,
                anchor_hz=float(anchor_hz),
            ).as_dict()
        elif note_token:
            token_pos = pdf_spiral_xy_from_token(note_token, anchor_token=anchor_token)
            coords = token_pos.as_dict() if token_pos else None

        if not coords:
            continue

        is_harmonic, harmonic_index, harmonic_delta = harmonic_role(freq_hz, float(root_hz or 0.0))

        rows.append({
            "time_sec": float(r.get("time_sec", 0.0)),
            "frame_idx": int(r.get("frame_idx", 0)) if "frame_idx" in df.columns else 0,
            "x12": float(coords["x12"]),
            "y12": float(coords["y12"]),
            "freq_hz": freq_hz,
            "amplitude": float(r.get("amp", 0.0)),
            "note_token": str(coords["note_token"] or note_token),
            "semitone_offset": float(coords["semitone_offset"]),
            "abs_step_float": float(coords["abs_step_float"]),
            "octave_float": float(coords["octave_float"]),
            "degree12_float": float(coords["degree12_float"]),
            "phase12_deg": float(coords["phase12_deg"]),
            "phase12_rad": float(coords["phase12_rad"]),
            "radial_level": float(coords["radial_level"]),
            "is_expected_harmonic": int(is_harmonic),
            "harmonic_index": int(harmonic_index) if is_harmonic else 0,
            "harmonic_delta_cents": float(harmonic_delta),
        })

    spiral_df = pd.DataFrame(rows)

    if len(spiral_df) == 0:
        spiral_df.to_csv(out_csv, index=False)
        return

    spiral_df.to_csv(out_csv, index=False)

    plt.figure(figsize=(7, 7))

    bg = spiral_df[spiral_df["is_expected_harmonic"] == 0]
    if len(bg) > 0:
        plt.scatter(
            bg["x12"],
            bg["y12"],
            s=5,
            alpha=0.08,
            c="#9ca3af",
        )

    fg = spiral_df[spiral_df["is_expected_harmonic"] == 1]
    if len(fg) > 0:
        plt.scatter(
            fg["x12"],
            fg["y12"],
            s=[max(10.0, min(80.0, float(a) * 0.08)) for a in fg["amplitude"]],
            alpha=0.62,
            c="#1d4ed8",
        )

        centers = (
            fg.groupby("harmonic_index")
            .agg(x12=("x12", "median"), y12=("y12", "median"))
            .reset_index()
            .sort_values("harmonic_index")
        )
        for _, row in centers.iterrows():
            plt.text(
                float(row["x12"]),
                float(row["y12"]),
                f"h{int(row['harmonic_index'])}",
                fontsize=7,
            )

    plt.title("spiral12 clean")
    plt.axis("equal")
    plt.grid(True, alpha=0.25)
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()


def process_note_dir(note_dir, anchor_token, anchor_hz):
    dense_path = find_dense_file(note_dir)

    if not dense_path:
        return False

    prefix = output_prefix_from_note_dir(note_dir, dense_path)

    df = load_csv_safe(dense_path)

    if df is None or len(df) == 0:
        return False

    df = normalize_dense(df)

    if "hz" not in df.columns:
        return False

    expected_note = extract_expected_note_token(os.path.basename(note_dir))
    if not expected_note:
        expected_note = expected_note_from_manifest(note_dir)
    root_hz, root_token = estimate_root_from_dense(
        df,
        expected_note=expected_note,
        anchor_token=anchor_token,
        anchor_hz=anchor_hz,
    )

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
        anchor_token,
        anchor_hz,
        root_hz=root_hz,
    )

    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports_root", required=True)
    ap.add_argument("--anchor_token", default="9.A-")
    ap.add_argument("--anchor_hz", type=float, default=440.0)
    args = ap.parse_args()

    ok = 0
    skipped = 0

    for d in os.listdir(args.reports_root):
        note_dir = os.path.join(args.reports_root, d)

        if not os.path.isdir(note_dir):
            continue

        if process_note_dir(note_dir, args.anchor_token, args.anchor_hz):
            ok += 1
        else:
            skipped += 1

    print("REPORTS FROM EXISTING DENSE DONE")
    print(f"reports_root : {args.reports_root}")
    print(f"processed    : {ok}")
    print(f"skipped      : {skipped}")


if __name__ == "__main__":
    main()
