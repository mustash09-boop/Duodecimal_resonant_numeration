from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import re
from pathlib import Path
from typing import Any


DIGITS12 = "123456789ABC"
_VAL12 = {ch: i + 1 for i, ch in enumerate(DIGITS12)}
_CH12 = {i + 1: ch for i, ch in enumerate(DIGITS12)}


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


def normalize_letters(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("А", "A").replace("В", "B").replace("С", "C")
    s = s.replace("а", "A").replace("в", "B").replace("с", "C")
    return s.upper()


def bij12_to_int(s: str) -> int:
    s = normalize_letters(s)
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
    tok = normalize_letters(tok).replace("'", "").rstrip("-")
    if "." not in tok:
        raise ValueError(f"Bad note token: {tok!r}")
    oct_s, step_s = tok.split(".", 1)
    step = step_s[:1]
    if not oct_s or any(ch not in _VAL12 for ch in oct_s):
        raise ValueError(f"Bad octave in token: {tok!r}")
    if step not in _VAL12:
        raise ValueError(f"Bad step in token: {tok!r}")
    return oct_s, step


def token_to_abs_step(token: str) -> int:
    oct_s, step = parse_base_note_token(token)
    octave0 = bij12_to_int(oct_s) - 1
    degree0 = _VAL12[step] - 1
    return octave0 * 12 + degree0


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


def root_hz_from_note_token(token: str, anchor_token: str = "9.A-", anchor_hz: float = 440.0) -> float:
    semitone_delta = token_to_abs_step(token) - token_to_abs_step(anchor_token)
    return float(anchor_hz * (2.0 ** (semitone_delta / 12.0)))


def hz_to_token_with_micro(
    freq_hz: float,
    *,
    anchor_token: str = "9.A-",
    anchor_hz: float = 440.0,
    micro_steps_per_semitone: int = 12,
    exact_mark: bool = True,
) -> str:
    if freq_hz <= 0:
        return ""

    abs_anchor = token_to_abs_step(anchor_token)
    semitone_offset = 12.0 * math.log2(freq_hz / anchor_hz)

    nearest_semitone = int(round(semitone_offset))
    residual = semitone_offset - nearest_semitone

    abs_note = abs_anchor + nearest_semitone
    base_token = abs_step_to_token(abs_note, micro="")

    micro_float = residual * micro_steps_per_semitone
    micro_rounded = int(round(micro_float))

    if micro_rounded == 0:
        return f"{base_token}'-" if exact_mark else base_token

    sign = "i" if micro_rounded > 0 else "a"
    magnitude = abs(micro_rounded)

    while magnitude >= micro_steps_per_semitone:
        if sign == "i":
            abs_note += 1
        else:
            abs_note -= 1
        magnitude -= micro_steps_per_semitone

    if magnitude == 0:
        base_token = abs_step_to_token(abs_note, micro="")
        return f"{base_token}'-" if exact_mark else base_token

    digit = int_to_base12_digit(magnitude)
    base_token = abs_step_to_token(abs_note, micro="")
    return f"{base_token}'{sign}{digit}"


def extract_note(folder_name: str) -> str:
    m = re.search(r"([1-9ABC]+\.[1-9ABC]+-)$", folder_name, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot extract note from folder name: {folder_name}")
    return m.group(1).upper()


def load_dense_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "time_sec": safe_float(row.get("time_sec")),
                "freq_hz": safe_float(row.get("freq_hz")),
                "amplitude": safe_float(row.get("amplitude")),
                "phase_rad": safe_float(row.get("phase_rad")),
                "frame_index": safe_int(row.get("frame_index")),
                "peak_index": safe_int(row.get("peak_index")),
            })
    return rows


def group_by_frame(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(safe_int(r["frame_index"]), []).append(r)
    return out


def best_match(frame_rows: list[dict[str, Any]], target_hz: float, tolerance_cents: float) -> dict[str, Any] | None:
    best = None
    best_abs = 1e18
    best_amp = -1e18

    for r in frame_rows:
        hz = safe_float(r["freq_hz"])
        ce = cents_error(hz, target_hz)
        ace = abs(ce)
        amp = safe_float(r["amplitude"])

        if ace <= tolerance_cents:
            if ace < best_abs or (abs(ace - best_abs) < 1e-9 and amp > best_amp):
                best_abs = ace
                best_amp = amp
                best = {
                    "matched_hz": hz,
                    "matched_amplitude": amp,
                    "matched_phase_rad": safe_float(r["phase_rad"]),
                    "delta_cents": ce,
                    "peak_index": safe_int(r["peak_index"]),
                }

    return best


def expected_visible_harmonics(root_hz: float, max_harmonic: int, max_freq_hz: float) -> list[int]:
    return [h for h in range(1, max_harmonic + 1) if root_hz * h <= max_freq_hz]


def analyze_high_note(
    dense_rows: list[dict[str, Any]],
    *,
    expected_note: str,
    anchor_token: str,
    anchor_hz: float,
    max_harmonic: int,
    max_freq_hz: float,
    tolerance_cents: float,
    min_present_percent: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root_hz = root_hz_from_note_token(expected_note, anchor_token=anchor_token, anchor_hz=anchor_hz)
    root_token = hz_to_token_with_micro(root_hz, anchor_token=anchor_token, anchor_hz=anchor_hz)

    harmonics = expected_visible_harmonics(root_hz, max_harmonic=max_harmonic, max_freq_hz=max_freq_hz)
    by_frame = group_by_frame(dense_rows)
    total_frames = len(by_frame)

    rows: list[dict[str, Any]] = []

    for h in harmonics:
        target_hz = root_hz * h
        theoretical_token = hz_to_token_with_micro(target_hz, anchor_token=anchor_token, anchor_hz=anchor_hz)

        matches: list[dict[str, Any]] = []

        for frame_index, frame_rows in sorted(by_frame.items()):
            m = best_match(frame_rows, target_hz, tolerance_cents)
            if m is not None:
                m["frame_index"] = frame_index
                m["time_sec"] = safe_float(frame_rows[0]["time_sec"]) if frame_rows else 0.0
                matches.append(m)

        freqs = [safe_float(m["matched_hz"]) for m in matches]
        amps = [safe_float(m["matched_amplitude"]) for m in matches]
        deltas = [safe_float(m["delta_cents"]) for m in matches]
        phases = [safe_float(m["matched_phase_rad"]) for m in matches]

        mean_hz = mean(freqs)
        median_hz = median(freqs)

        present_count = len(matches)
        present_percent = (100.0 * present_count / total_frames) if total_frames else 0.0

        mean_token = hz_to_token_with_micro(mean_hz, anchor_token=anchor_token, anchor_hz=anchor_hz) if mean_hz > 0 else ""
        median_token = hz_to_token_with_micro(median_hz, anchor_token=anchor_token, anchor_hz=anchor_hz) if median_hz > 0 else ""

        rows.append({
            "expected_note": expected_note,
            "root_hz": root_hz,
            "root_token": root_token,
            "harmonic_index": h,
            "theoretical_hz": target_hz,
            "theoretical_token": theoretical_token,
            "present_frame_count": present_count,
            "present_percent_frames": present_percent,
            "mean_matched_hz": mean_hz,
            "mean_matched_token": mean_token,
            "median_matched_hz": median_hz,
            "median_matched_token": median_token,
            "std_matched_hz": std(freqs),
            "mean_amplitude": mean(amps),
            "median_amplitude": median(amps),
            "std_amplitude": std(amps),
            "mean_delta_cents": mean(deltas),
            "median_delta_cents": median(deltas),
            "std_delta_cents": std(deltas),
            "mean_phase_rad": mean(phases),
            "is_stable": 1 if present_percent >= min_present_percent else 0,
        })

    stable = [r for r in rows if safe_int(r["is_stable"]) == 1]
    stable_indices = [safe_int(r["harmonic_index"]) for r in stable]

    has_root = 1 in stable_indices
    has_h2 = 2 in stable_indices
    has_h3 = 3 in stable_indices

    if has_root and (has_h2 or has_h3):
        verdict = "HIGH_CONFIRMED"
    elif has_root or (has_h2 and has_h3):
        verdict = "HIGH_WEAK_CONFIRMED"
    elif len(stable_indices) >= 2:
        verdict = "HIGH_PARTIAL_TRACE"
    else:
        verdict = "HIGH_NOT_CONFIRMED"

    summary = {
        "expected_note": expected_note,
        "root_hz": root_hz,
        "root_token": root_token,
        "total_frames": total_frames,
        "visible_harmonics": harmonics,
        "stable_harmonics": stable_indices,
        "stable_count": len(stable_indices),
        "verdict": verdict,
        "tolerance_cents": tolerance_cents,
        "min_present_percent": min_present_percent,
        "max_freq_hz": max_freq_hz,
    }

    return rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "expected_note",
        "root_hz",
        "root_token",
        "harmonic_index",
        "theoretical_hz",
        "theoretical_token",
        "present_frame_count",
        "present_percent_frames",
        "mean_matched_hz",
        "mean_matched_token",
        "median_matched_hz",
        "median_matched_token",
        "std_matched_hz",
        "mean_amplitude",
        "median_amplitude",
        "std_amplitude",
        "mean_delta_cents",
        "median_delta_cents",
        "std_delta_cents",
        "mean_phase_rad",
        "is_stable",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_txt(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("HIGH REGISTER CHAIN PRESENCE")
    lines.append("=" * 100)
    lines.append(f"expected_note       : {summary['expected_note']}")
    lines.append(f"root_hz             : {summary['root_hz']}")
    lines.append(f"root_token          : {summary['root_token']}")
    lines.append(f"total_frames        : {summary['total_frames']}")
    lines.append(f"visible_harmonics   : {summary['visible_harmonics']}")
    lines.append(f"stable_harmonics    : {summary['stable_harmonics']}")
    lines.append(f"stable_count        : {summary['stable_count']}")
    lines.append(f"verdict             : {summary['verdict']}")
    lines.append(f"tolerance_cents     : {summary['tolerance_cents']}")
    lines.append(f"min_present_percent : {summary['min_present_percent']}")
    lines.append(f"max_freq_hz         : {summary['max_freq_hz']}")
    lines.append("")
    lines.append("HARMONICS")
    lines.append("-" * 100)

    for r in rows:
        lines.append(
            f"h{r['harmonic_index']:>2}: "
            f"theory={r['theoretical_token']:10s} ({r['theoretical_hz']:.6f})  "
            f"present={r['present_frame_count']:4d}  "
            f"percent={r['present_percent_frames']:7.2f}  "
            f"mean={r['mean_matched_token']:10s} ({r['mean_matched_hz']:.6f})  "
            f"median={r['median_matched_token']:10s} ({r['median_matched_hz']:.6f})  "
            f"mean_amp={r['mean_amplitude']:.6f}  "
            f"mean_delta={r['mean_delta_cents']:.6f}  "
            f"stable={r['is_stable']}"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="High-register note confirmation from dense CSV over all frames.")
    ap.add_argument("--dense_csv", required=True)
    ap.add_argument("--expected_note", default="")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_txt", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--anchor_token", default="9.A-")
    ap.add_argument("--anchor_hz", type=float, default=440.0)
    ap.add_argument("--max_harmonic", type=int, default=12)
    ap.add_argument("--max_freq_hz", type=float, default=21000.0)
    ap.add_argument("--tolerance_cents", type=float, default=28.0)
    ap.add_argument("--min_present_percent", type=float, default=1.0)
    args = ap.parse_args()

    dense_csv = Path(args.dense_csv).resolve()
    expected_note = args.expected_note.strip() or extract_note(dense_csv.parent.name)

    rows = load_dense_rows(dense_csv)
    report_rows, summary = analyze_high_note(
        rows,
        expected_note=expected_note,
        anchor_token=args.anchor_token,
        anchor_hz=float(args.anchor_hz),
        max_harmonic=int(args.max_harmonic),
        max_freq_hz=float(args.max_freq_hz),
        tolerance_cents=float(args.tolerance_cents),
        min_present_percent=float(args.min_present_percent),
    )

    write_csv(Path(args.out_csv).resolve(), report_rows)
    write_txt(Path(args.out_txt).resolve(), report_rows, summary)
    write_json(Path(args.out_json).resolve(), summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()