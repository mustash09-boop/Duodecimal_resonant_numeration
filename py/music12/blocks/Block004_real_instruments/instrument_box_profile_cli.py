from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


DIGITS12 = "123456789ABC"
_VAL12 = {ch: i + 1 for i, ch in enumerate(DIGITS12)}
_CH12 = {i + 1: ch for i, ch in enumerate(DIGITS12)}


def sf(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else 0.0


def cents_error(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return 1e9
    return 1200.0 * math.log2(a / b)


def normalize_letters(s: str) -> str:
    return (s or "").replace("А", "A").replace("В", "B").replace("С", "C").upper().strip()


def bij12_to_int(s: str) -> int:
    s = normalize_letters(s)
    n = 0
    for ch in s:
        n = n * 12 + _VAL12[ch]
    return n


def int_to_bij12(n: int) -> str:
    out = []
    while n > 0:
        n, r = divmod(n - 1, 12)
        out.append(_CH12[r + 1])
    return "".join(reversed(out))


def int_to_base12_digit(i0: int) -> str:
    return _CH12[i0 + 1]


def parse_token(tok: str) -> tuple[str, str]:
    tok = normalize_letters(tok).replace("'", "").rstrip("-")
    oct_s, step_s = tok.split(".", 1)
    return oct_s, step_s[:1]


def token_to_abs_step(token: str) -> int:
    oct_s, step = parse_token(token)
    return (bij12_to_int(oct_s) - 1) * 12 + (_VAL12[step] - 1)


def abs_step_to_token(abs_step: int, micro: str = "-") -> str:
    oct0, step0 = divmod(abs_step, 12)
    base = f"{int_to_bij12(oct0 + 1)}.{int_to_base12_digit(step0)}"
    return f"{base}'{micro}" if micro else base


def hz_to_token(freq_hz: float, anchor_token: str = "9.A-", anchor_hz: float = 440.0) -> str:
    if freq_hz <= 0:
        return ""

    anchor_step = token_to_abs_step(anchor_token)
    semitone_offset = 12.0 * math.log2(freq_hz / anchor_hz)
    nearest = int(round(semitone_offset))
    residual = semitone_offset - nearest

    abs_note = anchor_step + nearest
    micro = int(round(residual * 12))

    if micro == 0:
        return abs_step_to_token(abs_note, "-")

    sign = "i" if micro > 0 else "a"
    mag = abs(micro)

    while mag >= 12:
        abs_note += 1 if sign == "i" else -1
        mag -= 12

    if mag == 0:
        return abs_step_to_token(abs_note, "-")

    return abs_step_to_token(abs_note, f"{sign}{int_to_base12_digit(mag)}")


def extract_note(folder_name: str) -> str:
    m = re.search(r"([1-9ABC]+\.[1-9ABC]+-)$", folder_name, flags=re.IGNORECASE)
    if not m:
        return ""
    return m.group(1).upper()


def extract_index(folder_name: str) -> int:
    try:
        return int(folder_name.split("_")[0])
    except Exception:
        return -1


def note_root_hz(note: str, anchor_token: str, anchor_hz: float) -> float:
    return anchor_hz * (2.0 ** ((token_to_abs_step(note) - token_to_abs_step(anchor_token)) / 12.0))


def load_dense(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "time_sec": sf(row.get("time_sec")),
                "freq_hz": sf(row.get("freq_hz")),
                "amplitude": sf(row.get("amplitude")),
                "phase_rad": sf(row.get("phase_rad")),
                "frame_index": int(sf(row.get("frame_index"))),
            })
    return rows


def is_explained_by_note_harmonic(
    freq_hz: float,
    root_hz: float,
    *,
    max_harmonic: int,
    tolerance_cents: float,
) -> tuple[bool, int]:
    for h in range(1, max_harmonic + 1):
        target = root_hz * h
        if abs(cents_error(freq_hz, target)) <= tolerance_cents:
            return True, h
    return False, 0


def cluster_key_hz(freq_hz: float, bucket_hz: float) -> float:
    return round(freq_hz / bucket_hz) * bucket_hz


def collect_box(
    reports_root: Path,
    *,
    anchor_token: str,
    anchor_hz: float,
    max_harmonic: int,
    harmonic_tolerance_cents: float,
    bucket_hz: float,
    min_amp: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_clusters: dict[float, dict[str, Any]] = {}
    residual_clusters: dict[float, dict[str, Any]] = {}

    folders = sorted([p for p in reports_root.iterdir() if p.is_dir()])

    for folder in folders:
        note = extract_note(folder.name)
        if not note:
            continue

        dense_files = sorted(folder.glob("*__dense.csv"))
        if not dense_files:
            continue

        dense_rows = load_dense(dense_files[0])
        root_hz = note_root_hz(note, anchor_token, anchor_hz)

        # Берём не один кадр, а весь dense.
        # Для "коробки" важно повторение по нотам, а не мгновенная красота.
        seen_raw_in_note = set()
        seen_residual_in_note = set()

        for r in dense_rows:
            freq = sf(r["freq_hz"])
            amp = sf(r["amplitude"])
            if freq <= 0 or amp < min_amp:
                continue

            key = cluster_key_hz(freq, bucket_hz)
            token = hz_to_token(key, anchor_token, anchor_hz)

            if key not in raw_clusters:
                raw_clusters[key] = {
                    "cluster_hz": key,
                    "cluster_token": token,
                    "notes": set(),
                    "amps": [],
                    "freqs": [],
                    "examples": [],
                }

            raw_clusters[key]["amps"].append(amp)
            raw_clusters[key]["freqs"].append(freq)
            if key not in seen_raw_in_note:
                raw_clusters[key]["notes"].add(note)
                raw_clusters[key]["examples"].append(folder.name)
                seen_raw_in_note.add(key)

            explained, h = is_explained_by_note_harmonic(
                freq,
                root_hz,
                max_harmonic=max_harmonic,
                tolerance_cents=harmonic_tolerance_cents,
            )

            if explained:
                continue

            if key not in residual_clusters:
                residual_clusters[key] = {
                    "cluster_hz": key,
                    "cluster_token": token,
                    "notes": set(),
                    "amps": [],
                    "freqs": [],
                    "examples": [],
                }

            residual_clusters[key]["amps"].append(amp)
            residual_clusters[key]["freqs"].append(freq)
            if key not in seen_residual_in_note:
                residual_clusters[key]["notes"].add(note)
                residual_clusters[key]["examples"].append(folder.name)
                seen_residual_in_note.add(key)

    total_notes = len([p for p in folders if extract_note(p.name)])

    def finalize(d: dict[float, dict[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        for key, data in d.items():
            note_count = len(data["notes"])
            percent = 100.0 * note_count / total_notes if total_notes else 0.0
            rows.append({
                "cluster_hz": data["cluster_hz"],
                "cluster_token": data["cluster_token"],
                "note_count": note_count,
                "percent_notes": percent,
                "mean_freq_hz": mean(data["freqs"]),
                "median_freq_hz": median(data["freqs"]),
                "mean_amp": mean(data["amps"]),
                "median_amp": median(data["amps"]),
                "examples": " | ".join(data["examples"][:12]),
            })
        rows.sort(key=lambda x: (-x["percent_notes"], -x["mean_amp"], x["cluster_hz"]))
        return rows

    return finalize(raw_clusters), finalize(residual_clusters)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cluster_hz",
        "cluster_token",
        "note_count",
        "percent_notes",
        "mean_freq_hz",
        "median_freq_hz",
        "mean_amp",
        "median_amp",
        "examples",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_txt(path: Path, raw_rows: list[dict[str, Any]], residual_rows: list[dict[str, Any]], top_n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("INSTRUMENT BOX PROFILE")
    lines.append("=" * 100)
    lines.append("")
    lines.append("RAW REPEATED COMPONENTS")
    lines.append("-" * 100)
    for r in raw_rows[:top_n]:
        lines.append(
            f"{r['cluster_token']:12s} {r['cluster_hz']:10.1f} Hz  "
            f"notes={r['note_count']:3d}  percent={r['percent_notes']:7.2f}  "
            f"mean_amp={r['mean_amp']:.6f}"
        )

    lines.append("")
    lines.append("RESIDUAL BOX CANDIDATES — after removing note harmonics")
    lines.append("-" * 100)
    for r in residual_rows[:top_n]:
        lines.append(
            f"{r['cluster_token']:12s} {r['cluster_hz']:10.1f} Hz  "
            f"notes={r['note_count']:3d}  percent={r['percent_notes']:7.2f}  "
            f"mean_amp={r['mean_amp']:.6f}"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build instrument resonance box profile from dense scans.")
    ap.add_argument("--reports_root", required=True)
    ap.add_argument("--out_raw_csv", required=True)
    ap.add_argument("--out_residual_csv", required=True)
    ap.add_argument("--out_txt", required=True)
    ap.add_argument("--anchor_token", default="9.A-")
    ap.add_argument("--anchor_hz", type=float, default=440.0)
    ap.add_argument("--max_harmonic", type=int, default=12)
    ap.add_argument("--harmonic_tolerance_cents", type=float, default=28.0)
    ap.add_argument("--bucket_hz", type=float, default=5.0)
    ap.add_argument("--min_amp", type=float, default=0.0)
    ap.add_argument("--top_n", type=int, default=80)
    args = ap.parse_args()

    raw, residual = collect_box(
        Path(args.reports_root).resolve(),
        anchor_token=args.anchor_token,
        anchor_hz=float(args.anchor_hz),
        max_harmonic=int(args.max_harmonic),
        harmonic_tolerance_cents=float(args.harmonic_tolerance_cents),
        bucket_hz=float(args.bucket_hz),
        min_amp=float(args.min_amp),
    )

    write_csv(Path(args.out_raw_csv).resolve(), raw)
    write_csv(Path(args.out_residual_csv).resolve(), residual)
    write_txt(Path(args.out_txt).resolve(), raw, residual, int(args.top_n))

    print(json.dumps({
        "raw_components": len(raw),
        "residual_components": len(residual),
        "out_raw_csv": args.out_raw_csv,
        "out_residual_csv": args.out_residual_csv,
        "out_txt": args.out_txt,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()