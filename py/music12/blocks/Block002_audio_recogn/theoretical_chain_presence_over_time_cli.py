from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


DIGITS12 = "123456789ABC"
_VAL12 = {ch: i + 1 for i, ch in enumerate(DIGITS12)}
_CH12 = {i + 1: ch for i, ch in enumerate(DIGITS12)}


# ============================================================
# 12-radix helpers
# ============================================================

def normalize_letters(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("А", "A").replace("В", "B").replace("С", "C")
    s = s.replace("а", "A").replace("в", "B").replace("с", "C")
    return s


def bij12_to_int(s: str) -> int:
    s = normalize_letters(s).upper()
    if not s or any(ch not in _VAL12 for ch in s):
        raise ValueError(f"Bad bij12 number: {s!r}")
    n = 0
    for ch in s:
        n = n * 12 + _VAL12[ch]
    return n


def int_to_bij12(n: int) -> str:
    n = int(n)
    if n <= 0:
        raise ValueError("int_to_bij12 expects n >= 1")
    out: list[str] = []
    while n > 0:
        n, r = divmod(n - 1, 12)
        out.append(_CH12[r + 1])
    return "".join(reversed(out))


def int_to_base12_digit(i0: int) -> str:
    i0 = int(i0)
    if not 0 <= i0 < 12:
        raise ValueError("int_to_base12_digit expects 0..11")
    return _CH12[i0 + 1]


def parse_base_note_token(tok: str) -> tuple[str, str]:
    tok = normalize_letters(tok).upper().strip()
    tok = tok.replace("’-", "'-")

    if "'" in tok:
        tok = tok.split("'", 1)[0]

    if "." not in tok:
        raise ValueError(f"Bad note token: {tok!r}")

    oct_s, step = tok.split(".", 1)
    step = step[:1]
    if not oct_s or any(ch not in _VAL12 for ch in oct_s):
        raise ValueError(f"Bad octave in token: {tok!r}")
    if step not in _VAL12:
        raise ValueError(f"Bad step in token: {tok!r}")

    return oct_s, step


def token_to_abs_step(token: str) -> int:
    oct_s, step = parse_base_note_token(token)
    oct0 = bij12_to_int(oct_s) - 1
    step0 = _VAL12[step] - 1
    return oct0 * 12 + step0


def abs_step_to_token(abs_step: int, micro: str = "-") -> str:
    abs_step = int(abs_step)
    if abs_step < 0:
        raise ValueError("abs_step must be >= 0")

    oct0, step0 = divmod(abs_step, 12)
    oct_s = int_to_bij12(oct0 + 1)
    step = int_to_base12_digit(step0)

    if micro:
        return f"{oct_s}.{step}'{micro}"
    return f"{oct_s}.{step}"


# ============================================================
# Micro token projection helpers
# ============================================================

def split_token_micro(token: str) -> tuple[str, str]:
    token = normalize_letters(token).strip()
    if "'" not in token:
        return token, ""
    coarse, micro = token.split("'", 1)
    return coarse, micro


def micro_suffix_to_fraction_semitones(micro: str, *, micro_depth: int = 2) -> float:
    micro = normalize_letters(micro).strip().upper()

    if not micro or micro == "-":
        return 0.0

    direction = micro[0].lower()
    if direction not in ("i", "a"):
        return 0.0

    digits = micro[1:]
    if not digits:
        return 0.0

    if any(ch not in _VAL12 for ch in digits):
        return 0.0

    digits = digits[: max(1, int(micro_depth))]

    n = 0
    for ch in digits:
        n = n * 12 + (_VAL12[ch] - 1)

    denom = 12 ** len(digits)
    frac = float(n) / float(denom)

    if direction == "a":
        frac = -frac

    return frac


def root_hz_from_note_token(
    token: str,
    anchor_token: str = "9.A-",
    anchor_hz: float = 440.0,
    *,
    micro_depth: int = 2,
) -> float:
    coarse, micro = split_token_micro(token)
    anchor_coarse, anchor_micro = split_token_micro(anchor_token)

    coarse_delta = token_to_abs_step(coarse) - token_to_abs_step(anchor_coarse)
    token_micro_delta = micro_suffix_to_fraction_semitones(micro, micro_depth=micro_depth)
    anchor_micro_delta = micro_suffix_to_fraction_semitones(anchor_micro, micro_depth=micro_depth)

    semitone_delta = float(coarse_delta) + token_micro_delta - anchor_micro_delta

    return float(anchor_hz * (2.0 ** (semitone_delta / 12.0)))


def hz_to_token_with_micro(
    freq_hz: float,
    *,
    anchor_token: str = "9.A-",
    anchor_hz: float = 440.0,
    micro_depth: int = 2,
    exact_mark: bool = True,
) -> str:
    if freq_hz <= 0:
        return ""

    abs_anchor = token_to_abs_step(anchor_token)
    _anchor_coarse, anchor_micro = split_token_micro(anchor_token)
    anchor_micro_delta = micro_suffix_to_fraction_semitones(anchor_micro, micro_depth=micro_depth)

    semitone_offset_from_anchor = 12.0 * math.log2(freq_hz / anchor_hz)
    absolute_semitone_float = float(abs_anchor) + anchor_micro_delta + semitone_offset_from_anchor

    nearest_abs_step = int(round(absolute_semitone_float))
    residual = absolute_semitone_float - float(nearest_abs_step)

    base_token = abs_step_to_token(nearest_abs_step, micro="")
    steps_per_semitone = 12 ** int(max(1, micro_depth))
    micro_rounded = int(round(residual * steps_per_semitone))

    if micro_rounded == 0:
        return f"{base_token}'-" if exact_mark else base_token

    sign = "i" if micro_rounded > 0 else "a"
    magnitude = abs(micro_rounded)

    while magnitude >= steps_per_semitone:
        if sign == "i":
            nearest_abs_step += 1
        else:
            nearest_abs_step -= 1
        magnitude -= steps_per_semitone

    base_token = abs_step_to_token(nearest_abs_step, micro="")

    if magnitude == 0:
        return f"{base_token}'-" if exact_mark else base_token

    digits: list[str] = []
    remaining = int(magnitude)
    for power in reversed(range(int(max(1, micro_depth)))):
        denom = 12 ** power
        digit0 = remaining // denom
        remaining = remaining % denom
        digit0 = max(0, min(11, int(digit0)))
        digits.append(int_to_base12_digit(digit0))

    micro_digits = "".join(digits).lstrip("1")
    if not micro_digits:
        micro_digits = "1"

    return f"{base_token}'{sign}{micro_digits}"


def token_coarse(token: str) -> str:
    coarse, _micro = split_token_micro(token)
    return coarse


# ============================================================
# Helpers
# ============================================================

def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def median(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def std(xs: list[float]) -> float:
    return float(statistics.pstdev(xs)) if len(xs) >= 2 else 0.0


def cents_error(observed_hz: float, target_hz: float) -> float:
    if observed_hz <= 0 or target_hz <= 0:
        return 1e9
    return 1200.0 * math.log2(observed_hz / target_hz)


# ============================================================
# Dense loading
# ============================================================

def load_dense_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "time_sec": safe_float(row.get("time_sec", 0.0)),
                    "freq_hz": safe_float(row.get("freq_hz", 0.0)),
                    "amplitude": safe_float(row.get("amplitude", 0.0)),
                    "phase_rad": safe_float(row.get("phase_rad", 0.0)),
                    "frame_index": safe_int(row.get("frame_index", 0)),
                    "peak_index": safe_int(row.get("peak_index", 0)),
                }
            )
    return rows


def group_by_frame(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        fi = row["frame_index"]
        out.setdefault(fi, []).append(row)
    return out


# ============================================================
# Matching
# ============================================================

def find_matches_for_target(
    frame_rows: list[dict[str, Any]],
    target_hz: float,
    tolerance_cents: float,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for row in frame_rows:
        hz = row["freq_hz"]
        ce = cents_error(hz, target_hz)
        ace = abs(ce)
        if ace <= tolerance_cents:
            matches.append(
                {
                    "matched_hz": hz,
                    "matched_amplitude": row["amplitude"],
                    "matched_phase_rad": row["phase_rad"],
                    "matched_peak_index": row["peak_index"],
                    "delta_cents": ce,
                    "abs_delta_cents": ace,
                }
            )
    matches.sort(key=lambda x: (x["abs_delta_cents"], -safe_float(x["matched_amplitude"], 0.0)))
    return matches


def choose_best_match(matches: list[dict[str, Any]]) -> dict[str, Any] | None:
    return matches[0] if matches else None


# ============================================================
# Core analysis
# ============================================================

def build_presence_rows(
    dense_rows: list[dict[str, Any]],
    *,
    expected_note: str,
    anchor_token: str,
    anchor_hz: float,
    max_harmonic: int,
    tolerance_cents: float,
    micro_depth: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root_hz = root_hz_from_note_token(
        expected_note,
        anchor_token=anchor_token,
        anchor_hz=anchor_hz,
        micro_depth=micro_depth,
    )
    by_frame = group_by_frame(dense_rows)

    expected_note_micro = expected_note
    expected_note_coarse = token_coarse(expected_note)

    frame_rows_out: list[dict[str, Any]] = []
    harmonic_summary_input: dict[int, list[dict[str, Any]]] = {h: [] for h in range(1, max_harmonic + 1)}

    for frame_index in sorted(by_frame):
        frame_rows = by_frame[frame_index]
        time_sec = safe_float(frame_rows[0]["time_sec"], 0.0) if frame_rows else 0.0

        for h in range(1, max_harmonic + 1):
            target_hz = root_hz * h

            theory_token_micro = hz_to_token_with_micro(
                target_hz,
                anchor_token=anchor_token,
                anchor_hz=anchor_hz,
                micro_depth=micro_depth,
            )
            theory_token_coarse = token_coarse(theory_token_micro)

            matches = find_matches_for_target(frame_rows, target_hz, tolerance_cents)
            best = choose_best_match(matches)

            if best is None:
                frame_rows_out.append(
                    {
                        "frame_index": frame_index,
                        "time_sec": time_sec,
                        "expected_note_micro": expected_note_micro,
                        "expected_note_coarse": expected_note_coarse,
                        "harmonic_index": h,
                        "theoretical_hz": target_hz,
                        "theoretical_token_micro": theory_token_micro,
                        "theoretical_token_coarse": theory_token_coarse,
                        "is_present": 0,
                        "matched_hz": "",
                        "matched_token_micro": "",
                        "matched_token_coarse": "",
                        "matched_amplitude": "",
                        "matched_phase_rad": "",
                        "delta_cents": "",
                        "match_count_in_frame": 0,
                    }
                )
                continue

            matched_hz = safe_float(best["matched_hz"], 0.0)
            matched_amp = safe_float(best["matched_amplitude"], 0.0)
            matched_phase = safe_float(best["matched_phase_rad"], 0.0)
            delta = safe_float(best["delta_cents"], 0.0)

            matched_token_micro = hz_to_token_with_micro(
                matched_hz,
                anchor_token=anchor_token,
                anchor_hz=anchor_hz,
                micro_depth=micro_depth,
            )
            matched_token_coarse = token_coarse(matched_token_micro)

            row = {
                "frame_index": frame_index,
                "time_sec": time_sec,
                "expected_note_micro": expected_note_micro,
                "expected_note_coarse": expected_note_coarse,
                "harmonic_index": h,
                "theoretical_hz": target_hz,
                "theoretical_token_micro": theory_token_micro,
                "theoretical_token_coarse": theory_token_coarse,
                "is_present": 1,
                "matched_hz": matched_hz,
                "matched_token_micro": matched_token_micro,
                "matched_token_coarse": matched_token_coarse,
                "matched_amplitude": matched_amp,
                "matched_phase_rad": matched_phase,
                "delta_cents": delta,
                "match_count_in_frame": len(matches),
            }
            frame_rows_out.append(row)
            harmonic_summary_input[h].append(row)

    harmonic_summary_rows: list[dict[str, Any]] = []
    total_frames = len(by_frame)

    for h in range(1, max_harmonic + 1):
        rows_h = harmonic_summary_input[h]
        present_count = len(rows_h)
        percent_frames = (100.0 * present_count / total_frames) if total_frames > 0 else 0.0

        amps = [safe_float(r["matched_amplitude"], 0.0) for r in rows_h]
        deltas = [safe_float(r["delta_cents"], 0.0) for r in rows_h]
        freqs = [safe_float(r["matched_hz"], 0.0) for r in rows_h]

        theoretical_hz = root_hz * h
        theoretical_token_micro = hz_to_token_with_micro(
            theoretical_hz,
            anchor_token=anchor_token,
            anchor_hz=anchor_hz,
            micro_depth=micro_depth,
        )
        theoretical_token_coarse = token_coarse(theoretical_token_micro)

        harmonic_summary_rows.append(
            {
                "expected_note_micro": expected_note_micro,
                "expected_note_coarse": expected_note_coarse,
                "harmonic_index": h,
                "theoretical_hz": theoretical_hz,
                "theoretical_token_micro": theoretical_token_micro,
                "theoretical_token_coarse": theoretical_token_coarse,
                "present_frame_count": present_count,
                "present_percent_frames": percent_frames,
                "mean_matched_hz": mean(freqs),
                "median_matched_hz": median(freqs),
                "std_matched_hz": std(freqs),
                "mean_amplitude": mean(amps),
                "median_amplitude": median(amps),
                "std_amplitude": std(amps),
                "mean_delta_cents": mean(deltas),
                "median_delta_cents": median(deltas),
                "std_delta_cents": std(deltas),
            }
        )

    return frame_rows_out, harmonic_summary_rows


# ============================================================
# Writers
# ============================================================

def write_presence_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame_index",
        "time_sec",
        "expected_note_micro",
        "expected_note_coarse",
        "harmonic_index",
        "theoretical_hz",
        "theoretical_token_micro",
        "theoretical_token_coarse",
        "is_present",
        "matched_hz",
        "matched_token_micro",
        "matched_token_coarse",
        "matched_amplitude",
        "matched_phase_rad",
        "delta_cents",
        "match_count_in_frame",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_harmonic_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "expected_note_micro",
        "expected_note_coarse",
        "harmonic_index",
        "theoretical_hz",
        "theoretical_token_micro",
        "theoretical_token_coarse",
        "present_frame_count",
        "present_percent_frames",
        "mean_matched_hz",
        "median_matched_hz",
        "std_matched_hz",
        "mean_amplitude",
        "median_amplitude",
        "std_amplitude",
        "mean_delta_cents",
        "median_delta_cents",
        "std_delta_cents",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_txt(
    path: Path,
    *,
    expected_note: str,
    frame_rows: list[dict[str, Any]],
    harmonic_summary_rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    total_frames = len({safe_int(r["frame_index"], 0) for r in frame_rows})

    lines: list[str] = []
    lines.append("THEORETICAL CHAIN PRESENCE OVER TIME")
    lines.append("=" * 100)
    lines.append(f"expected_note_micro : {expected_note}")
    lines.append(f"expected_note_coarse: {token_coarse(expected_note)}")
    lines.append(f"total_frames        : {total_frames}")
    lines.append("")

    lines.append("HARMONIC SUMMARY")
    lines.append("-" * 100)

    for row in harmonic_summary_rows:
        h = row["harmonic_index"]
        lines.append(
            f"h{h}: theory={row['theoretical_token_micro']} "
            f"[coarse={row['theoretical_token_coarse']}] "
            f"({row['theoretical_hz']:.6f})  "
            f"present_frames={row['present_frame_count']}  "
            f"present_percent={row['present_percent_frames']:.2f}  "
            f"mean_hz={row['mean_matched_hz']:.6f}  "
            f"mean_amp={row['mean_amplitude']:.6f}  "
            f"mean_delta_cents={row['mean_delta_cents']:.6f}"
        )

    lines.append("")
    lines.append("INTERPRETATION HINTS")
    lines.append("-" * 100)
    lines.append("Look first at h1..h6 presence across frames, not only at one frame.")
    lines.append("If h1 is weak but h3..h7 are stable, the note may still be valid in low register.")
    lines.append("If a false higher root is chosen later, compare its early harmonic profile against this report.")
    lines.append("Micro tokens are preserved separately from coarse tokens.")
    lines.append("Do not collapse theoretical_token_micro or matched_token_micro into coarse identity.")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_meta_json(
    path: Path,
    *,
    dense_csv: Path,
    expected_note: str,
    out_presence_csv: Path,
    out_harmonic_summary_csv: Path,
    out_summary_txt: Path,
    max_harmonic: int,
    tolerance_cents: float,
    micro_depth: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "inputs": {
            "dense_csv": str(dense_csv),
            "expected_note_micro": expected_note,
            "expected_note_coarse": token_coarse(expected_note),
        },
        "outputs": {
            "presence_csv": str(out_presence_csv),
            "harmonic_summary_csv": str(out_harmonic_summary_csv),
            "summary_txt": str(out_summary_txt),
            "meta_json": str(path),
        },
        "settings": {
            "max_harmonic": int(max_harmonic),
            "tolerance_cents": float(tolerance_cents),
            "micro_depth": int(micro_depth),
        },
        "semantic_note": (
            "This report checks one theoretical note chain across all dense frames. "
            "It is intended for diagnosing whether the note really exists in the dense scan "
            "before global chain-building logic selects a root. "
            "Micro and coarse identities are kept as separate fields."
        ),
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# CLI
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Check one theoretical harmonic chain against all frames of a dense CSV."
    )
    ap.add_argument("--dense_csv", required=True)
    ap.add_argument("--expected_note", required=True)
    ap.add_argument("--out_presence_csv", required=True)
    ap.add_argument("--out_harmonic_summary_csv", required=True)
    ap.add_argument("--out_summary_txt", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--anchor_token", default="9.A-")
    ap.add_argument("--anchor_hz", type=float, default=440.0)
    ap.add_argument("--max_harmonic", type=int, default=12)
    ap.add_argument("--tolerance_cents", type=float, default=35.0)
    ap.add_argument(
        "--micro_depth",
        type=int,
        default=2,
        help="Recursive micro depth for Hz/token projection. 2 means 144-like layer.",
    )
    args = ap.parse_args()

    dense_csv = Path(args.dense_csv).resolve()
    out_presence_csv = Path(args.out_presence_csv).resolve()
    out_harmonic_summary_csv = Path(args.out_harmonic_summary_csv).resolve()
    out_summary_txt = Path(args.out_summary_txt).resolve()
    out_meta_json = Path(args.out_meta_json).resolve()

    dense_rows = load_dense_rows(dense_csv)

    frame_rows, harmonic_summary_rows = build_presence_rows(
        dense_rows,
        expected_note=str(args.expected_note),
        anchor_token=str(args.anchor_token),
        anchor_hz=float(args.anchor_hz),
        max_harmonic=int(args.max_harmonic),
        tolerance_cents=float(args.tolerance_cents),
        micro_depth=int(args.micro_depth),
    )

    write_presence_csv(out_presence_csv, frame_rows)
    write_harmonic_summary_csv(out_harmonic_summary_csv, harmonic_summary_rows)
    write_summary_txt(
        out_summary_txt,
        expected_note=str(args.expected_note),
        frame_rows=frame_rows,
        harmonic_summary_rows=harmonic_summary_rows,
    )
    write_meta_json(
        out_meta_json,
        dense_csv=dense_csv,
        expected_note=str(args.expected_note),
        out_presence_csv=out_presence_csv,
        out_harmonic_summary_csv=out_harmonic_summary_csv,
        out_summary_txt=out_summary_txt,
        max_harmonic=int(args.max_harmonic),
        tolerance_cents=float(args.tolerance_cents),
        micro_depth=int(args.micro_depth),
    )

    print(f"Wrote presence CSV        : {out_presence_csv}")
    print(f"Wrote harmonic summary CSV: {out_harmonic_summary_csv}")
    print(f"Wrote summary TXT         : {out_summary_txt}")
    print(f"Wrote meta JSON           : {out_meta_json}")


if __name__ == "__main__":
    main()
