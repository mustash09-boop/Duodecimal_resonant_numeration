from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


DIGITS12 = "123456789ABC"
_VAL12 = {ch: i + 1 for i, ch in enumerate(DIGITS12)}
_CH12 = {i + 1: ch for i, ch in enumerate(DIGITS12)}


def sf(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def normalize(s: str) -> str:
    return (s or "").replace("А", "A").replace("В", "B").replace("С", "C").upper().strip()


def bij12_to_int(s: str) -> int:
    s = normalize(s)
    n = 0
    for ch in s:
        if ch not in _VAL12:
            raise ValueError(f"Bad 12 digit: {ch!r}")
        n = n * 12 + _VAL12[ch]
    return n


def int_to_bij12(n: int) -> str:
    n = int(n)
    if n <= 0:
        return ""
    out = []
    while n > 0:
        n, r = divmod(n - 1, 12)
        out.append(_CH12[r + 1])
    return "".join(reversed(out))


def int_to_base12_digit(i0: int) -> str:
    return _CH12[i0 + 1]


def parse_token(tok: str) -> tuple[str, str]:
    tok = normalize(tok).replace("'", "").rstrip("-")
    if "." not in tok:
        raise ValueError(f"Bad note token: {tok!r}")
    oct_s, step_s = tok.split(".", 1)
    return oct_s, step_s[:1]


def token_to_abs_step(token: str) -> int:
    oct_s, step = parse_token(token)
    octave0 = bij12_to_int(oct_s) - 1
    degree0 = _VAL12[step] - 1
    return octave0 * 12 + degree0


def abs_step_to_token(abs_step: int, micro: str = "-") -> str:
    octave0, degree0 = divmod(int(abs_step), 12)
    base = f"{int_to_bij12(octave0 + 1)}.{int_to_base12_digit(degree0)}"
    return f"{base}'{micro}" if micro else base


def hz_to_token_with_micro(
    freq_hz: float,
    *,
    anchor_token: str,
    anchor_hz: float,
    micro_steps_per_semitone: int = 12,
) -> str:
    if freq_hz <= 0:
        return ""

    anchor_abs = token_to_abs_step(anchor_token)
    semitone_offset = 12.0 * math.log2(freq_hz / anchor_hz)

    nearest = int(round(semitone_offset))
    residual = semitone_offset - nearest
    abs_step = anchor_abs + nearest

    micro = int(round(residual * micro_steps_per_semitone))

    if micro == 0:
        return abs_step_to_token(abs_step, "-")

    sign = "i" if micro > 0 else "a"
    mag = abs(micro)

    while mag >= micro_steps_per_semitone:
        abs_step += 1 if sign == "i" else -1
        mag -= micro_steps_per_semitone

    if mag == 0:
        return abs_step_to_token(abs_step, "-")

    return abs_step_to_token(abs_step, f"{sign}{int_to_base12_digit(mag)}")


def spiral12_coords(freq_hz: float, *, anchor_token: str, anchor_hz: float) -> dict[str, float | str]:
    """
    12-ричная спираль:
    - anchor_token / anchor_hz задаёт нулевую опору.
    - semitone_abs = абсолютное положение в 12-ричной шкале.
    - degree12_float = положение внутри витка 0..12.
    - phase12_deg = degree * 30 градусов.
    - radial_level = octave + degree/12.
    """
    if freq_hz <= 0:
        return {
            "note_token": "",
            "semitone_offset": 0.0,
            "abs_step_float": 0.0,
            "octave_float": 0.0,
            "degree12_float": 0.0,
            "phase12_deg": 0.0,
            "phase12_rad": 0.0,
            "radial_level": 0.0,
            "x12": 0.0,
            "y12": 0.0,
        }

    anchor_abs = token_to_abs_step(anchor_token)
    semitone_offset = 12.0 * math.log2(freq_hz / anchor_hz)
    abs_step_float = anchor_abs + semitone_offset

    octave_float = math.floor(abs_step_float / 12.0)
    degree12_float = abs_step_float - octave_float * 12.0

    phase12_deg = degree12_float * 30.0
    phase12_rad = math.radians(phase12_deg)

    radial_level = octave_float + degree12_float / 12.0

    # Центрируем радиус относительно anchor, чтобы картинка была читаемой.
    anchor_octave = math.floor(anchor_abs / 12.0)
    radial_relative = radial_level - anchor_octave

    x12 = radial_relative * math.cos(phase12_rad)
    y12 = radial_relative * math.sin(phase12_rad)

    return {
        "note_token": hz_to_token_with_micro(freq_hz, anchor_token=anchor_token, anchor_hz=anchor_hz),
        "semitone_offset": semitone_offset,
        "abs_step_float": abs_step_float,
        "octave_float": octave_float,
        "degree12_float": degree12_float,
        "phase12_deg": phase12_deg,
        "phase12_rad": phase12_rad,
        "radial_level": radial_level,
        "x12": x12,
        "y12": y12,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build true 12-radix spiral from cleaned dense CSV.")
    ap.add_argument("--dense_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--anchor_token", default="9.A-")
    ap.add_argument("--anchor_hz", type=float, default=440.0)
    ap.add_argument("--title", default="12-radix cleaned dense spiral")
    args = ap.parse_args()

    dense_csv = Path(args.dense_csv).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_png = Path(args.out_png).resolve()

    rows_out = []

    with dense_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        source_fields = list(reader.fieldnames or [])

        for row in reader:
            freq = sf(row.get("freq_hz"))
            amp = sf(row.get("amplitude"))
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

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=source_fields + extra_fields)
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
    plt.xlabel("12-radix spiral X")
    plt.ylabel("12-radix spiral Y")
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