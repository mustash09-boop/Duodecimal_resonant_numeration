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

    # Лучи 12 ступеней
    for i in range(12):
        angle = i * (2.0 * math.pi / 12.0)
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

        if frame_count < min_frame_count:
            continue

        if presence_ratio < min_presence_ratio:
            continue

        records.append({
            "token": str(token),
            "mean_hz": float(g["hz"].mean()),
            "median_hz": float(g["hz"].median()),
            "mean_amp": float(g["amp"].mean()),
            "median_amp": float(g["amp"].median()),
            "frame_count": frame_count,
            "presence_ratio": float(presence_ratio),
            "mean_x12": float(g["x12"].mean()),
            "mean_y12": float(g["y12"].mean()),
            "root_hz": float(root_hz),
            "source_note_dir": os.path.basename(note_dir),
        })

    if not records:
        return None

    result_df = (
        pd.DataFrame(records)
        .sort_values(by=["presence_ratio", "mean_amp"], ascending=[False, False])
        .reset_index(drop=True)
    )

    note_name = os.path.basename(note_dir)

    out_csv = os.path.join(out_dir, f"{note_name}__note_box_profile.csv")
    out_txt = os.path.join(out_dir, f"{note_name}__note_box_profile.txt")

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
                f"frames={int(r['frame_count']):4d} "
                f"presence={r['presence_ratio']:.3f} "
                f"mean_amp={r['mean_amp']:.6f}\n"
            )

    save_note_box_spiral_png(
        note_name=note_name,
        root_hz=root_hz,
        points_df=points_df,
        box_df=box_df,
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