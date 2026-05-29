import argparse
import json
import re
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd


def extract_traces_from_plotly_html(html_path: Path):
    text = html_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"const\s+traces\s*=\s*(\[.*?\]);", text, re.DOTALL)
    if not m:
        raise ValueError("Could not find Plotly trace block: const traces = [...]")
    return json.loads(m.group(1))


def normalize_instrument_name(name: str) -> str:
    return name.split("/")[0].strip()


def build_curve(points, grid):
    if not points:
        return np.zeros_like(grid, dtype=float)

    df = pd.DataFrame(points, columns=["t", "amp"])
    df = df.groupby("t", as_index=False)["amp"].max().sort_values("t")

    if len(df) == 1:
        curve = np.zeros_like(grid, dtype=float)
        idx = np.argmin(np.abs(grid - float(df["t"].iloc[0])))
        curve[idx] = float(df["amp"].iloc[0])
        return curve

    return np.interp(
        grid,
        df["t"].to_numpy(dtype=float),
        df["amp"].to_numpy(dtype=float),
        left=0.0,
        right=0.0,
    )


def curve_features(curve, grid):
    eps = 1e-12
    area = float(np.trapezoid(curve, grid))
    peak = float(np.max(curve)) if len(curve) else 0.0
    peak_time = float(grid[int(np.argmax(curve))]) if peak > eps else 0.0

    energy_sum = float(np.sum(curve))
    centroid = float(np.sum(grid * curve) / (energy_sum + eps))

    attack_mask = grid <= 0.20
    sustain_mask = (grid > 0.20) & (grid <= 0.70)
    tail_mask = grid > 0.70

    attack_energy = float(np.trapezoid(curve[attack_mask], grid[attack_mask])) if np.any(attack_mask) else 0.0
    sustain_energy = float(np.trapezoid(curve[sustain_mask], grid[sustain_mask])) if np.any(sustain_mask) else 0.0
    tail_energy = float(np.trapezoid(curve[tail_mask], grid[tail_mask])) if np.any(tail_mask) else 0.0

    active_ratio = float(np.mean(curve > peak * 0.10)) if peak > eps else 0.0
    roughness = float(np.mean(np.abs(np.diff(curve)))) if len(curve) > 1 else 0.0

    return {
        "area_integral": area,
        "peak_amp": peak,
        "peak_time": peak_time,
        "temporal_centroid": centroid,
        "attack_energy": attack_energy,
        "sustain_energy": sustain_energy,
        "tail_energy": tail_energy,
        "active_ratio": active_ratio,
        "roughness": roughness,
    }


def curve_distance(a, b):
    eps = 1e-12

    raw_l2 = float(np.sqrt(np.mean((a - b) ** 2)))
    raw_l1 = float(np.mean(np.abs(a - b)))

    a_norm = a / (np.max(a) + eps)
    b_norm = b / (np.max(b) + eps)

    shape_l2 = float(np.sqrt(np.mean((a_norm - b_norm) ** 2)))
    shape_l1 = float(np.mean(np.abs(a_norm - b_norm)))

    if np.std(a_norm) < eps or np.std(b_norm) < eps:
        corr_distance = 1.0
    else:
        corr = float(np.corrcoef(a_norm, b_norm)[0, 1])
        corr_distance = float(1.0 - corr)

    return {
        "raw_l2": raw_l2,
        "raw_l1": raw_l1,
        "shape_l2": shape_l2,
        "shape_l1": shape_l1,
        "corr_distance": corr_distance,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Compare instrument harmonic temporal morphology from Plotly 3D harmonic amplitude HTML."
    )
    ap.add_argument("--html", required=True, help="Input Plotly HTML file")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--time-grid", type=int, default=240, help="Number of normalized time samples")
    ap.add_argument("--max-harmonic", type=int, default=24, help="Maximum harmonic index to keep")
    args = ap.parse_args()

    html_path = Path(args.html)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    traces = extract_traces_from_plotly_html(html_path)
    grid = np.linspace(0.0, 1.0, args.time_grid)

    raw_points = []
    grouped = {}

    for tr in traces:
        instrument = normalize_instrument_name(tr.get("name", "unknown"))
        xs = tr.get("x", [])
        ys = tr.get("y", [])
        zs = tr.get("z", [])

        for h, t, amp in zip(xs, ys, zs):
            try:
                h = int(h)
                t = float(t)
                amp = float(amp)
            except Exception:
                continue

            if h < 1 or h > args.max_harmonic:
                continue

            grouped.setdefault(instrument, {}).setdefault(h, []).append((t, amp))
            raw_points.append({
                "instrument": instrument,
                "harmonic": h,
                "time_norm": t,
                "relative_amplitude": amp,
            })

    if not grouped:
        raise ValueError("Could not extract harmonic points from the input HTML.")

    pd.DataFrame(raw_points).to_csv(outdir / "01_raw_harmonic_points.csv", index=False, encoding="utf-8-sig")

    curves = {}
    feature_rows = []
    curve_rows = []

    for instrument, by_harm in grouped.items():
        curves[instrument] = {}
        for h in range(1, args.max_harmonic + 1):
            curve = build_curve(by_harm.get(h, []), grid)
            curves[instrument][h] = curve

            feats = curve_features(curve, grid)
            feature_rows.append({
                "instrument": instrument,
                "harmonic": h,
                **feats,
            })

            for t, amp in zip(grid, curve):
                curve_rows.append({
                    "instrument": instrument,
                    "harmonic": h,
                    "time_norm_grid": float(t),
                    "amplitude_curve": float(amp),
                })

    pd.DataFrame(curve_rows).to_csv(outdir / "02_harmonic_temporal_curves.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(feature_rows).to_csv(outdir / "03_harmonic_morphology_features.csv", index=False, encoding="utf-8-sig")

    distance_rows = []
    summary_rows = []

    instruments = sorted(curves.keys())

    for a, b in combinations(instruments, 2):
        total_shape = []
        total_raw = []
        total_corr = []

        for h in range(1, args.max_harmonic + 1):
            d = curve_distance(curves[a][h], curves[b][h])

            distance_rows.append({
                "instrument_a": a,
                "instrument_b": b,
                "harmonic": h,
                **d,
            })

            total_shape.append(d["shape_l2"])
            total_raw.append(d["raw_l2"])
            total_corr.append(d["corr_distance"])

        summary_rows.append({
            "instrument_a": a,
            "instrument_b": b,
            "harmonics_used": args.max_harmonic,
            "mean_shape_l2": float(np.mean(total_shape)),
            "max_shape_l2": float(np.max(total_shape)),
            "mean_raw_l2": float(np.mean(total_raw)),
            "mean_corr_distance": float(np.mean(total_corr)),
            "morphology_distance_score": float(
                0.60 * np.mean(total_shape)
                + 0.25 * np.mean(total_corr)
                + 0.15 * np.mean(total_raw)
            ),
        })

    pd.DataFrame(distance_rows).to_csv(outdir / "04_pairwise_harmonic_curve_distances.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).sort_values(
        "morphology_distance_score", ascending=False
    ).to_csv(outdir / "05_pairwise_instrument_morphology_summary.csv", index=False, encoding="utf-8-sig")

    print("OK")
    print(f"HTML: {html_path}")
    print(f"Instruments: {', '.join(instruments)}")
    print(f"Output: {outdir}")


if __name__ == "__main__":
    main()
