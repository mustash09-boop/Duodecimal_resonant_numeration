from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_points(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def normalize_note_curve(chain_df: pd.DataFrame, grid_size: int) -> list[dict]:
    rows: list[dict] = []
    if len(chain_df) == 0:
        return rows

    time_vals = chain_df["time_sec"].astype(float).to_numpy()
    amp_vals = chain_df["relative_amp"].astype(float).to_numpy()
    h_vals = chain_df["harmonic_index"].astype(int).to_numpy()

    t0 = float(np.min(time_vals))
    t1 = float(np.max(time_vals))
    if t1 <= t0:
        rel = np.zeros_like(time_vals)
    else:
        rel = (time_vals - t0) / (t1 - t0)

    grid = np.linspace(0.0, 1.0, grid_size)

    for h in sorted(set(h_vals.tolist())):
        mask = h_vals == h
        ht = rel[mask]
        ha = amp_vals[mask]
        if len(ht) == 0:
            continue
        order = np.argsort(ht)
        ht = ht[order]
        ha = ha[order]

        if len(ht) == 1:
            interp = np.full(grid.shape, float(ha[0]), dtype=float)
        else:
            uniq_t, uniq_idx = np.unique(ht, return_index=True)
            uniq_a = ha[uniq_idx]
            interp = np.interp(grid, uniq_t, uniq_a, left=uniq_a[0], right=uniq_a[-1])

        for g, a in zip(grid, interp):
            rows.append(
                {
                    "harmonic": int(h),
                    "time_norm_grid": float(g),
                    "amplitude_curve": float(a),
                }
            )
    return rows


def infer_note_name(path: Path) -> str:
    return path.name.replace("__spiral3d_points.csv", "")


def build_instrument_curves(instrument: str, spiral3d_dir: Path, grid_size: int) -> pd.DataFrame:
    all_rows: list[dict] = []
    for csv_path in sorted(spiral3d_dir.glob("*__spiral3d_points.csv")):
        df = load_points(csv_path)
        if df is None or len(df) == 0:
            continue

        if "component_type" not in df.columns or "harmonic_index" not in df.columns:
            continue
        chain_df = df[df["component_type"] == "chain"].copy()
        chain_df = chain_df[pd.to_numeric(chain_df["harmonic_index"], errors="coerce").notna()].copy()
        if len(chain_df) == 0:
            continue

        if "relative_amp" not in chain_df.columns:
            if "amplitude" in chain_df.columns:
                max_amp = float(chain_df["amplitude"].astype(float).max())
                if max_amp > 0:
                    chain_df["relative_amp"] = chain_df["amplitude"].astype(float) / max_amp
                else:
                    chain_df["relative_amp"] = 0.0
            else:
                continue

        rows = normalize_note_curve(chain_df, grid_size=grid_size)
        note_name = infer_note_name(csv_path)
        for r in rows:
            r["instrument"] = instrument
            r["note_name"] = note_name
        all_rows.extend(rows)

    if not all_rows:
        return pd.DataFrame(columns=["instrument", "note_name", "harmonic", "time_norm_grid", "amplitude_curve"])

    raw_df = pd.DataFrame(all_rows)
    grouped = (
        raw_df.groupby(["instrument", "harmonic", "time_norm_grid"], as_index=False)
        .agg(
            amplitude_curve=("amplitude_curve", "mean"),
            note_count=("note_name", "nunique"),
        )
    )
    return grouped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument_name", action="append", required=True)
    ap.add_argument("--spiral3d_dir", action="append", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--grid_size", type=int, default=64)
    args = ap.parse_args()

    if len(args.instrument_name) != len(args.spiral3d_dir):
        raise ValueError("instrument_name and spiral3d_dir counts must match")

    frames = []
    meta = {"grid_size": int(args.grid_size), "instruments": []}

    for instrument, folder in zip(args.instrument_name, args.spiral3d_dir):
        spiral3d_dir = Path(folder)
        df = build_instrument_curves(instrument=instrument, spiral3d_dir=spiral3d_dir, grid_size=int(args.grid_size))
        frames.append(df)
        meta["instruments"].append(
            {
                "instrument": instrument,
                "spiral3d_dir": str(spiral3d_dir),
                "rows": int(len(df)),
                "harmonics": sorted(set(int(x) for x in df["harmonic"].tolist())) if len(df) else [],
            }
        )

    out_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    meta["rows_total"] = int(len(out_df))
    Path(args.out_meta_json).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("OK")
    print(f"rows_total: {len(out_df)}")
    print(f"out_csv   : {out_path}")


if __name__ == "__main__":
    main()
