from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from music12.core.pdf_spiral12_xy import pdf_spiral_xy_from_frequency


def sf(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def spiral12_coords(freq_hz: float, *, anchor_token: str, anchor_hz: float) -> dict[str, float | str]:
    return pdf_spiral_xy_from_frequency(
        freq_hz,
        anchor_token=anchor_token,
        anchor_hz=anchor_hz,
    ).as_dict()


def main() -> None:
    ap = argparse.ArgumentParser(description="Build PDF-compatible 12-radix spiral from cleaned dense CSV.")
    ap.add_argument("--dense_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--anchor_token", default="9.A-")
    ap.add_argument("--anchor_hz", type=float, default=440.0)
    ap.add_argument("--title", default="PDF-compatible 12-radix cleaned dense spiral")
    args = ap.parse_args()

    dense_csv = Path(args.dense_csv).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_png = Path(args.out_png).resolve()

    rows_out: list[dict[str, Any]] = []

    with dense_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        source_fields = list(reader.fieldnames or [])

        for row in reader:
            freq = sf(row.get("freq_hz"), sf(row.get("hz")))
            amp = sf(row.get("amplitude"), sf(row.get("amp")))
            coords = spiral12_coords(freq, anchor_token=args.anchor_token, anchor_hz=float(args.anchor_hz))

            out = dict(row)
            out.update(coords)
            out["plot_size"] = max(1.0, min(80.0, amp * 0.08))
            rows_out.append(out)

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    extra_fields = [
        "note_token",
        "semitone_offset",
        "abs_step_float",
        "octave_float",
        "degree12_float",
        "phase12_deg",
        "phase12_rad",
        "radial_level",
        "x12",
        "y12",
        "plot_size",
    ]

    merged_fields = list(source_fields)
    for field in extra_fields:
        if field not in merged_fields:
            merged_fields.append(field)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=merged_fields)
        writer.writeheader()
        writer.writerows(rows_out)

    xs = [sf(r["x12"]) for r in rows_out]
    ys = [sf(r["y12"]) for r in rows_out]
    sizes = [sf(r["plot_size"], 2.0) for r in rows_out]

    plt.figure(figsize=(8, 8))
    plt.scatter(xs, ys, s=sizes, alpha=0.35)
    plt.axhline(0, linewidth=0.5)
    plt.axvline(0, linewidth=0.5)
    plt.title(args.title)
    plt.xlabel("PDF spiral X")
    plt.ylabel("PDF spiral Y")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180)
    plt.close()

    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote PNG: {out_png}")
    print(f"Rows: {len(rows_out)}")


if __name__ == "__main__":
    main()
