# -*- coding: utf-8 -*-
"""
NOTE BOX PROFILE BUILDER

Строит box-профиль для каждой ноты:
- берёт __spiral12_clean_points.csv
- исключает гармоническую цепь по root consensus
- оставляет устойчивые компоненты note-specific box
- создаёт PNG по 12-ричной спирали:
    гармоники ноты + box-компоненты

Затем собирает общий summary по инструменту.
"""

import os
import math
import argparse

import pandas as pd
import matplotlib.pyplot as plt

from music12.core.pdf_spiral12_xy import angle_for_step, STEPS


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_csv_safe(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def cents_diff(hz1, hz2):
    if hz1 <= 0 or hz2 <= 0:
        return 9999.0
    return 1200.0 * math.log2(hz1 / hz2)


def normalize_columns(df):
    """
    Приводим реальные имена колонок к внутренним:
    freq_hz    -> hz
    amplitude  -> amp
    frame_index -> frame_idx
    """
    rename_map = {}

    if "freq_hz" in df.columns and "hz" not in df.columns:
        rename_map["freq_hz"] = "hz"

    if "amplitude" in df.columns and "amp" not in df.columns:
        rename_map["amplitude"] = "amp"

    if "frame_index" in df.columns and "frame_idx" not in df.columns:
        rename_map["frame_index"] = "frame_idx"

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def is_harmonic(hz, root_hz, tolerance_cents, harmonic_min=1, harmonic_max=12):
    for h in range(harmonic_min, harmonic_max + 1):
        expected = root_hz * h
        if abs(cents_diff(hz, expected)) <= tolerance_cents:
            return True
    return False


def save_note_box_spiral_png(note_name, root_hz, points_df, box_df, out_dir, tolerance_cents):
    harmonic_points = []
    box_points = []

    if points_df is None or len(points_df) == 0:
        return

    required = {"hz", "note_token", "x12", "y12"}
    if not required.issubset(set(points_df.columns)):
        return

    # Гармоники самой ноты
    for h in range(1, 13):
        expected_hz = root_hz * h

        nearest = points_df.copy()
        nearest["delta_abs"] = nearest["hz"].apply(
            lambda x: abs(cents_diff(float(x), expected_hz))
        )
        nearest = nearest.sort_values("delta_abs")

        if len(nearest) == 0:
            continue

        row = nearest.iloc[0]

        if float(row["delta_abs"]) <= tolerance_cents:
            harmonic_points.append({
                "x": float(row["x12"]),
                "y": float(row["y12"]),
                "h": h,
                "token": str(row["note_token"]),
                "hz": float(row["hz"]),
                "delta_cents": float(row["delta_abs"]),
            })

    # Box-компоненты
    if box_df is not None and len(box_df) > 0:
        grouped = box_df.groupby("note_token")

        for token, g in grouped:
            if "x12" not in g.columns or "y12" not in g.columns:
                continue

            box_points.append({
                "x": float(g["x12"].mean()),
                "y": float(g["y12"].mean()),
                "token": str(token),
                "mean_hz": float(g["hz"].mean()) if "hz" in g.columns else 0.0,
                "mean_amp": float(g["amp"].mean()) if "amp" in g.columns else 0.0,
                "frame_count": int(g["frame_idx"].nunique()) if "frame_idx" in g.columns else int(len(g)),
            })

    if not harmonic_points and not box_points:
        return

    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    # Окружности
    for r in range(1, 15):
        circle = plt.Circle((0, 0), r, fill=False, linewidth=0.4, alpha=0.25)
        ax.add_patch(circle)

    # Лучи 12 ступеней по corrected PDF-геометрии
    for step in STEPS:
        angle = math.radians(angle_for_step(step))
        plt.plot(
            [0, 15 * math.cos(angle)],
            [0, 15 * math.sin(angle)],
            linewidth=0.4,
            alpha=0.25,
        )

    if harmonic_points:
        plt.scatter(
            [p["x"] for p in harmonic_points],
            [p["y"] for p in harmonic_points],
            s=90,
            label="note harmonics",
        )

        for p in harmonic_points:
            plt.text(
                p["x"],
                p["y"],
                f"h{p['h']}\n{p['token']}",
                fontsize=7,
            )

    if box_points:
        plt.scatter(
            [p["x"] for p in box_points],
            [p["y"] for p in box_points],
            s=45,
            marker="x",
            label="note box",
        )

        box_points_sorted = sorted(
            box_points,
            key=lambda p: (p["frame_count"], p["mean_amp"]),
            reverse=True,
        )

        for p in box_points_sorted[:40]:
            plt.text(
                p["x"],
                p["y"],
                p["token"],
                fontsize=6,
            )

    plt.title(f"12-spiral note harmonics + note-box: {note_name}")
    plt.axis("equal")
    plt.grid(True, alpha=0.2)
    plt.legend()

    out_png = os.path.join(out_dir, f"{note_name}__note_box_spiral.png")
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()


def extract_root_hz(root_path):
    root_hz = None

    try:
        with open(root_path, "r", encoding="utf-8") as f:
            for line in f:
                if "consensus_root_hz" in line:
                    try:
                        root_hz = float(line.split(":")[1].strip())
                    except Exception:
                        pass
    except Exception:
        return None

    return root_hz


def process_note(note_dir, out_dir, tolerance_cents, min_presence_ratio, min_frame_count):
    note_name = os.path.basename(note_dir)
    out_csv = os.path.join(out_dir, f"{note_name}__note_box_profile.csv")
    out_txt = os.path.join(out_dir, f"{note_name}__note_box_profile.txt")

    points_path = None
    root_path = None

    for f in os.listdir(note_dir):
        if f.endswith("__spiral12_clean_points.csv"):
            points_path = os.path.join(note_dir, f)
        if f.endswith("__root_consensus_summary.txt"):
            root_path = os.path.join(note_dir, f)

    if not points_path or not root_path:
        return None

    points_df = load_csv_safe(points_path)

    if points_df is None:
        return None

    points_df = normalize_columns(points_df)

    required = {"hz", "amp", "frame_idx", "note_token", "x12", "y12"}
    if not required.issubset(set(points_df.columns)):
        return None

    root_hz = extract_root_hz(root_path)

    if root_hz is None:
        return None

    total_frames = max(1, int(points_df["frame_idx"].nunique()))
    is_banjo_note = note_name.startswith("banjo_")
    is_banjo_soft = is_banjo_note and (
        "_piano_" in note_name
        or "_pianissimo_" in note_name
        or "_mezzo-piano_" in note_name
        or "_molto-pianissimo_" in note_name
    )
    is_banjo_very_long = is_banjo_note and (
        "_long_" in note_name or "_very-long_" in note_name
    )
    is_bass_guitar_note = "_bass-guitar_" in note_name
    is_real_piano_note = "_piano_real_" in note_name
    is_guitar2_note = "_guitar2_" in note_name
    is_guitar_note = note_name.startswith("guitar_")
    is_bass_clarinet_note = note_name.startswith("bass-clarinet_")
    is_bass_clarinet_phrase = is_bass_clarinet_note and "_phrase_" in note_name
    is_bassoon_note = note_name.startswith("bassoon_")
    is_bassoon_phrase = is_bassoon_note and "_phrase_" in note_name
    is_french_horn_note = note_name.startswith("french-horn_")
    is_french_horn_phrase = is_french_horn_note and "_phrase_" in note_name
    is_french_horn_soft = is_french_horn_note and (
        "_piano_" in note_name
        or "_pianissimo_" in note_name
        or "_mezzo-piano_" in note_name
        or "_molto-pianissimo_" in note_name
    )
    is_french_horn_long = is_french_horn_note and (
        "_long_" in note_name or "_very-long_" in note_name
    )
    is_french_horn_gliss = is_french_horn_note and "glissando" in note_name
    is_french_horn_legato = is_french_horn_phrase and "legato" in note_name
    is_french_horn_nonlegato = is_french_horn_phrase and "nonlegato" in note_name
    is_french_horn_cresc = is_french_horn_note and (
        "cresc-decresc" in note_name or "decrescendo" in note_name or "crescendo" in note_name
    )
    is_flute_note = note_name.startswith("flute_")
    is_flute_phrase = is_flute_note and "_phrase_" in note_name
    is_flute_soft = is_flute_note and (
        "_piano_" in note_name
        or "_pianissimo_" in note_name
        or "_mezzo-piano_" in note_name
        or "_molto-pianissimo_" in note_name
    )
    is_flute_very_long = is_flute_note and ("_very-long_" in note_name or "_long_" in note_name)
    is_flute_cresc = is_flute_note and ("cresc-decresc" in note_name or "decresc-cresc" in note_name)
    is_clarinet_note = note_name.startswith("clarinet_")
    is_clarinet_phrase = is_clarinet_note and "_phrase_" in note_name
    is_contrabassoon_note = note_name.startswith("contrabassoon_")
    is_contrabassoon_phrase = is_contrabassoon_note and "_phrase_" in note_name
    is_double_bass2_note = "_double-bass2_" in note_name
    is_double_bass_note = note_name.startswith("double-bass_")
    is_double_bass_phrase = is_double_bass_note and "_phrase_" in note_name
    is_double_bass_pizz = is_double_bass_note and "_pizz-" in note_name
    is_double_bass_soft = is_double_bass_note and (
        "_piano_" in note_name
        or "_pianissimo_" in note_name
        or "_molto-pianissimo_" in note_name
        or "_mezzo-piano_" in note_name
    )
    frame_min = int(points_df["frame_idx"].min())
    frame_max = int(points_df["frame_idx"].max())
    frame_span = max(1, frame_max - frame_min + 1)
    early_limit = frame_min + int(frame_span / 3.0)
    late_limit = frame_min + int((2.0 * frame_span) / 3.0)

    box_rows = []

    for _, row in points_df.iterrows():
        try:
            hz = float(row["hz"])
        except Exception:
            continue

        if not is_harmonic(hz, root_hz, tolerance_cents):
            box_rows.append(row)

    if not box_rows:
        return None

    box_df = pd.DataFrame(box_rows)

    records = []

    grouped = box_df.groupby("note_token")

    for token, g in grouped:
        frame_count = int(g["frame_idx"].nunique())
        presence_ratio = frame_count / total_frames
        mean_hz = float(g["hz"].mean())
        freq_ratio = mean_hz / float(root_hz) if root_hz > 0 else 0.0
        early_count = int((g["frame_idx"] <= early_limit).sum())
        late_count = int((g["frame_idx"] >= late_limit).sum())
        total_count = max(1, int(len(g)))
        early_ratio = early_count / total_count
        late_ratio = late_count / total_count

        if frame_count < min_frame_count:
            continue

        if presence_ratio < min_presence_ratio:
            continue

        # Box/body should stay local to the note, not become an ultra-high late-tail field.
        if freq_ratio > 24.0:
            continue

        if late_ratio >= 0.85 and early_ratio <= 0.05:
            continue

        # Suppress late-emerging high-ratio tail swirls that appear after the real body has decayed.
        if freq_ratio > 14.0 and late_ratio >= 0.80 and early_ratio <= 0.15:
            continue

        if freq_ratio > 18.0 and late_ratio >= 0.70 and early_ratio <= 0.25:
            continue

        if is_banjo_note:
            if freq_ratio > 8.0 and late_ratio >= 0.88 and early_ratio <= 0.12:
                continue
            if freq_ratio > 10.0 and late_ratio >= 0.82 and early_ratio <= 0.18:
                continue
            if late_ratio >= 0.76 and early_ratio <= 0.10 and frame_count >= 10:
                continue
            if freq_ratio > 4.0 and late_ratio >= 0.70 and early_ratio <= 0.14 and frame_count >= 10:
                continue
            if frame_count <= 18 and presence_ratio <= 0.14 and freq_ratio >= 1.2 and late_ratio >= 0.66:
                continue

        if is_banjo_very_long:
            if late_ratio >= 0.70 and early_ratio <= 0.14 and frame_count >= 10:
                continue
            if freq_ratio > 3.5 and late_ratio >= 0.66 and early_ratio <= 0.18:
                continue

        if is_banjo_soft and is_banjo_very_long:
            if late_ratio >= 0.62 and early_ratio <= 0.18:
                continue
            if freq_ratio > 3.0 and late_ratio >= 0.58 and early_ratio <= 0.22:
                continue

        if is_bass_guitar_note:
            if late_ratio >= 0.70 and early_ratio <= 0.10:
                continue
            if freq_ratio > 8.0 and late_ratio >= 0.60 and early_ratio <= 0.05:
                continue
            if frame_count <= 7 and presence_ratio <= 0.07 and late_ratio >= 0.66:
                continue
            if frame_count <= 12 and presence_ratio <= 0.12 and freq_ratio >= 2.0 and late_ratio >= 0.66 and early_ratio <= 0.18:
                continue
            if freq_ratio > 4.0 and late_ratio >= 0.66 and early_ratio <= 0.18 and frame_count <= 14:
                continue

        if is_real_piano_note:
            if freq_ratio > 5.0 and late_ratio >= 0.74 and early_ratio <= 0.05:
                continue
            if freq_ratio > 7.0 and late_ratio >= 0.70 and early_ratio <= 0.08:
                continue
            if freq_ratio > 7.0 and late_ratio >= 0.78 and early_ratio <= 0.06:
                continue
            if freq_ratio > 10.0 and late_ratio >= 0.72 and early_ratio <= 0.08:
                continue
            if freq_ratio > 14.0 and late_ratio >= 0.66 and early_ratio <= 0.12:
                continue

        if is_guitar2_note:
            if late_ratio >= 0.82 and early_ratio <= 0.12:
                continue
            if freq_ratio > 6.0 and late_ratio >= 0.75 and early_ratio <= 0.20:
                continue
            if freq_ratio > 10.0 and late_ratio >= 0.65 and early_ratio <= 0.35:
                continue
            if freq_ratio > 10.0 and late_ratio >= 0.50 and early_ratio <= 0.35:
                continue

        if is_guitar_note:
            if late_ratio >= 0.88 and early_ratio <= 0.12:
                continue
            if freq_ratio > 5.0 and late_ratio >= 0.82 and early_ratio <= 0.18:
                continue
            if freq_ratio > 7.0 and late_ratio >= 0.72 and early_ratio <= 0.28:
                continue
            if freq_ratio > 10.0 and late_ratio >= 0.55 and early_ratio <= 0.35:
                continue

        if is_bass_clarinet_note:
            if frame_count <= 2 and presence_ratio <= 0.055:
                continue
            if frame_count <= 7 and presence_ratio <= 0.20 and early_ratio >= 0.66 and freq_ratio >= 2.5:
                continue
            if (
                frame_count <= 30
                and presence_ratio <= 0.11
                and late_ratio >= 0.68
                and early_ratio <= 0.08
                and freq_ratio >= 2.5
            ):
                continue
            if freq_ratio > 12.0:
                continue
            if freq_ratio > 8.0 and frame_count < 60:
                continue
            if freq_ratio > 6.0 and late_ratio >= 0.78 and early_ratio <= 0.18:
                continue

        if is_bass_clarinet_phrase:
            if frame_count <= 8 and presence_ratio <= 0.20 and freq_ratio >= 2.5:
                continue
            if freq_ratio > 6.0:
                continue

        if is_bassoon_phrase:
            if freq_ratio > 12.0:
                continue
            if freq_ratio > 9.0 and presence_ratio <= 0.10:
                continue
            if freq_ratio > 7.0 and late_ratio >= 0.76 and early_ratio <= 0.10:
                continue
            if freq_ratio > 4.0 and late_ratio >= 0.80 and early_ratio <= 0.08 and frame_count >= 30:
                continue

        if is_french_horn_phrase:
            if late_ratio >= 0.70 and early_ratio <= 0.18 and frame_count >= 14:
                continue
            if freq_ratio > 5.0 and late_ratio >= 0.60 and early_ratio <= 0.24:
                continue
            if frame_count <= 12 and presence_ratio <= 0.10 and late_ratio >= 0.58:
                continue

        if is_french_horn_gliss:
            if late_ratio >= 0.58 and early_ratio <= 0.24:
                continue

        if is_french_horn_legato:
            if freq_ratio > 4.0 and late_ratio >= 0.56 and early_ratio <= 0.26:
                continue

        if is_french_horn_nonlegato:
            if freq_ratio > 3.5 and late_ratio >= 0.50 and early_ratio <= 0.26:
                continue

        if is_french_horn_soft:
            if frame_count <= 16 and presence_ratio <= 0.18 and freq_ratio >= 1.2 and late_ratio >= 0.46:
                continue
            if freq_ratio > 5.0 and late_ratio >= 0.54 and early_ratio <= 0.28:
                continue

        if is_french_horn_long:
            if freq_ratio > 4.5 and late_ratio >= 0.62 and early_ratio <= 0.28:
                continue

        if is_french_horn_cresc:
            if late_ratio >= 0.66 and early_ratio <= 0.22 and frame_count >= 12:
                continue

        if is_flute_phrase:
            if late_ratio >= 0.72 and early_ratio <= 0.22 and frame_count >= 10:
                continue
            if late_ratio >= 0.62 and early_ratio <= 0.18 and frame_count >= 18:
                continue
            if freq_ratio > 4.0 and late_ratio >= 0.58 and early_ratio <= 0.28:
                continue
            if freq_ratio > 6.0 and presence_ratio <= 0.18:
                continue

        if is_flute_soft:
            if frame_count <= 14 and presence_ratio <= 0.18 and freq_ratio >= 1.2 and late_ratio >= 0.42:
                continue
            if root_hz >= 600.0 and frame_count <= 22 and presence_ratio <= 0.24 and freq_ratio >= 0.95:
                continue
            if freq_ratio > 5.0 and late_ratio >= 0.54 and early_ratio <= 0.26:
                continue

        if is_flute_very_long:
            if freq_ratio > 6.0 and late_ratio >= 0.68 and early_ratio <= 0.22:
                continue

        if is_flute_cresc:
            if late_ratio >= 0.64 and early_ratio <= 0.28 and frame_count >= 20:
                continue

        if is_clarinet_phrase:
            if freq_ratio > 10.0:
                continue
            if freq_ratio > 6.0 and presence_ratio <= 0.11:
                continue
            if freq_ratio > 4.0 and late_ratio >= 0.78 and early_ratio <= 0.12:
                continue
            if frame_count <= 12 and late_ratio >= 0.66 and presence_ratio <= 0.10:
                continue

        if is_clarinet_note and not is_clarinet_phrase:
            if frame_count <= 10 and presence_ratio <= 0.08 and freq_ratio >= 1.8 and late_ratio >= 0.50:
                continue

        if is_contrabassoon_phrase:
            if freq_ratio > 16.0:
                continue
            if freq_ratio > 10.0 and presence_ratio <= 0.12:
                continue
            if freq_ratio > 6.0 and presence_ratio <= 0.08 and frame_count <= 50:
                continue
            if freq_ratio > 6.0 and late_ratio >= 0.62 and early_ratio <= 0.18:
                continue
            if freq_ratio > 7.0 and late_ratio >= 0.74 and early_ratio <= 0.12:
                continue
            if freq_ratio > 4.0 and late_ratio >= 0.80 and early_ratio <= 0.10 and frame_count >= 45:
                continue

        if is_contrabassoon_note and not is_contrabassoon_phrase:
            if frame_count <= 15 and presence_ratio <= 0.12 and freq_ratio >= 1.5 and late_ratio >= 0.55:
                continue
            if frame_count <= 20 and presence_ratio <= 0.11 and freq_ratio >= 2.3:
                continue

        if is_double_bass2_note:
            if frame_count <= 16 and presence_ratio <= 0.12 and freq_ratio >= 2.5 and late_ratio >= 0.50:
                continue
            if freq_ratio > 8.0 and late_ratio >= 0.74 and early_ratio <= 0.15:
                continue
            if freq_ratio > 5.0 and late_ratio >= 0.84 and early_ratio <= 0.10:
                continue
            if frame_count <= 18 and presence_ratio <= 0.14 and late_ratio >= 0.84 and early_ratio <= 0.18 and freq_ratio >= 0.9:
                continue
            if frame_count <= 12 and presence_ratio <= 0.10 and late_ratio >= 0.78 and early_ratio <= 0.20:
                continue

        if is_double_bass_phrase:
            if late_ratio >= 0.78 and early_ratio <= 0.18 and frame_count >= 16:
                continue
            if freq_ratio > 6.0 and late_ratio >= 0.70 and early_ratio <= 0.25:
                continue
            if freq_ratio > 10.0 and late_ratio >= 0.55 and early_ratio <= 0.35:
                continue

        if is_double_bass_pizz:
            if late_ratio >= 0.72 and early_ratio <= 0.20 and frame_count >= 10:
                continue
            if freq_ratio > 4.0 and late_ratio >= 0.55 and early_ratio <= 0.30:
                continue

        if is_double_bass_soft:
            if frame_count <= 10 and presence_ratio <= 0.12 and freq_ratio >= 1.8 and late_ratio >= 0.50:
                continue
            if freq_ratio > 5.0 and late_ratio >= 0.72 and early_ratio <= 0.18:
                continue
            if freq_ratio > 7.0 and late_ratio >= 0.60 and early_ratio <= 0.30:
                continue

        if is_double_bass_note:
            if frame_count <= 8 and presence_ratio <= 0.08 and freq_ratio >= 2.5:
                continue

        records.append({
            "token": str(token),
            "mean_hz": mean_hz,
            "median_hz": float(g["hz"].median()),
            "mean_amp": float(g["amp"].mean()),
            "median_amp": float(g["amp"].median()),
            "frame_count": frame_count,
            "presence_ratio": float(presence_ratio),
            "freq_ratio": float(freq_ratio),
            "early_count": early_count,
            "late_count": late_count,
            "early_ratio": float(early_ratio),
            "late_ratio": float(late_ratio),
            "mean_x12": float(g["x12"].mean()),
            "mean_y12": float(g["y12"].mean()),
            "root_hz": float(root_hz),
            "source_note_dir": os.path.basename(note_dir),
        })

    if not records:
        empty_cols = [
            "token",
            "mean_hz",
            "median_hz",
            "mean_amp",
            "median_amp",
            "frame_count",
            "presence_ratio",
            "freq_ratio",
            "early_count",
            "late_count",
            "early_ratio",
            "late_ratio",
            "mean_x12",
            "mean_y12",
            "root_hz",
            "source_note_dir",
        ]
        pd.DataFrame(columns=empty_cols).to_csv(out_csv, index=False)
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(f"NOTE BOX PROFILE: {note_name}\n")
            f.write("=" * 80 + "\n")
            f.write(f"source_points_csv        : {points_path}\n")
            f.write(f"root_hz                  : {root_hz}\n")
            f.write(f"harmonic_tolerance_cents : {tolerance_cents}\n")
            f.write(f"min_presence_ratio       : {min_presence_ratio}\n")
            f.write(f"min_frame_count          : {min_frame_count}\n")
            f.write(f"total_frames             : {total_frames}\n")
            f.write("box_components           : 0\n")
            f.write("status                   : empty_profile_new_schema\n")
        return pd.DataFrame(columns=empty_cols)

    result_df = (
        pd.DataFrame(records)
        .sort_values(by=["presence_ratio", "mean_amp"], ascending=[False, False])
        .reset_index(drop=True)
    )

    result_df.to_csv(out_csv, index=False)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"NOTE BOX PROFILE: {note_name}\n")
        f.write("=" * 80 + "\n")
        f.write(f"source_points_csv        : {points_path}\n")
        f.write(f"root_hz                  : {root_hz}\n")
        f.write(f"harmonic_tolerance_cents : {tolerance_cents}\n")
        f.write(f"min_presence_ratio       : {min_presence_ratio}\n")
        f.write(f"min_frame_count          : {min_frame_count}\n")
        f.write(f"total_frames             : {total_frames}\n")
        f.write(f"box_components           : {len(result_df)}\n\n")

        for _, r in result_df.iterrows():
            f.write(
                f"{str(r['token']):14} "
                f"hz={r['mean_hz']:.3f} "
                f"ratio={r['freq_ratio']:.2f} "
                f"frames={int(r['frame_count']):4d} "
                f"presence={r['presence_ratio']:.3f} "
                f"late={r['late_ratio']:.3f} "
                f"mean_amp={r['mean_amp']:.6f}\n"
            )

    allowed_tokens = set(result_df["token"].astype(str).tolist())
    box_df_for_plot = box_df[box_df["note_token"].astype(str).isin(allowed_tokens)].copy()

    save_note_box_spiral_png(
        note_name=note_name,
        root_hz=root_hz,
        points_df=points_df,
        box_df=box_df_for_plot,
        out_dir=out_dir,
        tolerance_cents=tolerance_cents,
    )

    return result_df


def build_summary(all_results, out_dir, instrument_name):
    combined = []

    for note_name, df in all_results.items():
        for _, r in df.iterrows():
            combined.append({
                "note": note_name,
                "token": r["token"],
                "mean_hz": r["mean_hz"],
                "median_hz": r["median_hz"],
                "mean_amp": r["mean_amp"],
                "median_amp": r["median_amp"],
                "frame_count": r["frame_count"],
                "presence_ratio": r["presence_ratio"],
                "mean_x12": r["mean_x12"],
                "mean_y12": r["mean_y12"],
                "root_hz": r["root_hz"],
            })

    if not combined:
        return None

    df = pd.DataFrame(combined)

    summary = (
        df.groupby("token")
        .agg(
            note_count=("note", "nunique"),
            avg_presence=("presence_ratio", "mean"),
            max_presence=("presence_ratio", "max"),
            mean_hz=("mean_hz", "mean"),
            mean_amp=("mean_amp", "mean"),
            max_amp=("mean_amp", "max"),
            mean_x12=("mean_x12", "mean"),
            mean_y12=("mean_y12", "mean"),
        )
        .reset_index()
        .sort_values(by=["note_count", "avg_presence", "mean_amp"], ascending=[False, False, False])
    )

    out_csv = os.path.join(out_dir, f"{instrument_name}__note_box_summary.csv")
    out_txt = os.path.join(out_dir, f"{instrument_name}__note_box_summary.txt")

    summary.to_csv(out_csv, index=False)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"NOTE BOX SUMMARY: {instrument_name}\n")
        f.write("=" * 80 + "\n")
        f.write(f"notes_with_profiles: {len(all_results)}\n")
        f.write(f"unique_box_tokens   : {len(summary)}\n\n")

        for _, r in summary.iterrows():
            f.write(
                f"{str(r['token']):14} "
                f"notes={int(r['note_count']):4d} "
                f"avg_presence={r['avg_presence']:.3f} "
                f"max_presence={r['max_presence']:.3f} "
                f"hz={r['mean_hz']:.3f} "
                f"mean_amp={r['mean_amp']:.6f}\n"
            )

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Build per-note note-box profiles from spiral12 clean points."
    )

    parser.add_argument("--instrument_name", required=True)
    parser.add_argument("--reports_root", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--harmonic_tolerance_cents", type=float, default=18.0)
    parser.add_argument("--min_presence_ratio", type=float, default=0.05)
    parser.add_argument("--min_frame_count", type=int, default=2)

    args = parser.parse_args()

    ensure_dir(args.out_dir)

    note_dirs = [
        os.path.join(args.reports_root, d)
        for d in os.listdir(args.reports_root)
        if os.path.isdir(os.path.join(args.reports_root, d))
    ]

    all_results = {}

    skipped = 0

    for note_dir in note_dirs:
        note_name = os.path.basename(note_dir)

        df = process_note(
            note_dir=note_dir,
            out_dir=args.out_dir,
            tolerance_cents=args.harmonic_tolerance_cents,
            min_presence_ratio=args.min_presence_ratio,
            min_frame_count=args.min_frame_count,
        )

        if df is not None:
            all_results[note_name] = df
        else:
            skipped += 1

    build_summary(
        all_results=all_results,
        out_dir=args.out_dir,
        instrument_name=args.instrument_name,
    )

    print("NOTE BOX PROFILE BUILDER DONE")
    print(f"instrument_name : {args.instrument_name}")
    print(f"reports_root    : {args.reports_root}")
    print(f"out_dir         : {args.out_dir}")
    print(f"profiles built  : {len(all_results)}")
    print(f"skipped         : {skipped}")


if __name__ == "__main__":
    main()
