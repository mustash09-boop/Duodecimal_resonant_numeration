import argparse
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd


def norm_curve(y):
    y = np.asarray(y, dtype=float)
    m = np.max(np.abs(y))
    if m <= 1e-12:
        return y * 0.0
    return y / m


def xcorr_lag(a, b, max_lag):
    a = norm_curve(a)
    b = norm_curve(b)

    best = None
    for lag in range(1, max_lag + 1):
        aa = a[:-lag]
        bb = b[lag:]
        if len(aa) < 5:
            continue
        if np.std(aa) < 1e-12 or np.std(bb) < 1e-12:
            corr = 0.0
        else:
            corr = float(np.corrcoef(aa, bb)[0, 1])
        if best is None or corr > best["corr"]:
            best = {"lag_steps": lag, "corr": corr}
    return best or {"lag_steps": 0, "corr": 0.0}


def curve_energy(y):
    return float(np.sum(np.asarray(y, dtype=float)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves_csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max_lag_steps", type=int, default=24)
    ap.add_argument("--min_corr", type=float, default=0.35)
    ap.add_argument("--min_target_energy", type=float, default=0.001)
    args = ap.parse_args()

    curves_csv = Path(args.curves_csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(curves_csv)

    required = {"instrument", "harmonic", "time_norm_grid", "amplitude_curve"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    edges = []
    nodes = []

    for instrument, g in df.groupby("instrument"):
        pivot = (
            g.pivot_table(
                index="time_norm_grid",
                columns="harmonic",
                values="amplitude_curve",
                aggfunc="max",
                fill_value=0.0,
            )
            .sort_index()
        )

        harmonics = sorted(pivot.columns)

        for h in harmonics:
            y = pivot[h].to_numpy(dtype=float)
            nodes.append({
                "instrument": instrument,
                "harmonic": int(h),
                "energy": curve_energy(y),
                "peak_amp": float(np.max(y)),
                "peak_time": float(pivot.index[int(np.argmax(y))]) if len(y) else 0.0,
            })

        for source_h, target_h in product(harmonics, harmonics):
            if source_h == target_h:
                continue

            source = pivot[source_h].to_numpy(dtype=float)
            target = pivot[target_h].to_numpy(dtype=float)

            target_energy = curve_energy(target)
            if target_energy < args.min_target_energy:
                continue

            lag = xcorr_lag(source, target, args.max_lag_steps)

            if lag["corr"] >= args.min_corr:
                dt = float(pivot.index[min(lag["lag_steps"], len(pivot.index)-1)] - pivot.index[0])

                role = "SECONDARY_RESPONSE"
                if target_h == source_h * 2:
                    role = "OCTAVE_RESONANCE"
                elif target_h > source_h:
                    role = "UPPER_HARMONIC_RESPONSE"
                elif target_h < source_h:
                    role = "LOWER_BODY_RESPONSE"

                edges.append({
                    "instrument": instrument,
                    "source_harmonic": int(source_h),
                    "target_harmonic": int(target_h),
                    "lag_steps": int(lag["lag_steps"]),
                    "lag_time_norm": dt,
                    "correlation": float(lag["corr"]),
                    "source_energy": curve_energy(source),
                    "target_energy": target_energy,
                    "relation_role": role,
                    "lineage": f"h{int(source_h)} -> h{int(target_h)}",
                })

    nodes_df = pd.DataFrame(nodes).sort_values(["instrument", "harmonic"])
    edges_df = pd.DataFrame(edges).sort_values(
        ["instrument", "correlation"], ascending=[True, False]
    )

    nodes_df.to_csv(outdir / "01_resonance_nodes.csv", index=False, encoding="utf-8-sig")
    edges_df.to_csv(outdir / "02_causal_resonance_edges.csv", index=False, encoding="utf-8-sig")

    if not edges_df.empty:
        summary = (
            edges_df.groupby(["instrument", "relation_role"])
            .agg(
                edge_count=("correlation", "count"),
                mean_corr=("correlation", "mean"),
                max_corr=("correlation", "max"),
                mean_lag=("lag_time_norm", "mean"),
            )
            .reset_index()
            .sort_values(["instrument", "edge_count"], ascending=[True, False])
        )
    else:
        summary = pd.DataFrame()

    summary.to_csv(outdir / "03_lineage_summary.csv", index=False, encoding="utf-8-sig")

    print("OK")
    print(f"Input: {curves_csv}")
    print(f"Output: {outdir}")
    print(f"Edges found: {len(edges_df)}")


if __name__ == "__main__":
    main()