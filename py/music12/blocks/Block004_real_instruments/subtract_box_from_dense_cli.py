from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any


DIGITS12 = "123456789ABC"
_VAL12 = {ch: i + 1 for i, ch in enumerate(DIGITS12)}


def sf(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def cents_error(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return 1e9
    return 1200.0 * math.log2(a / b)


def normalize(s: str) -> str:
    return (s or "").replace("А", "A").replace("В", "B").replace("С", "C").upper().strip()


def bij12_to_int(s: str) -> int:
    s = normalize(s)
    n = 0
    for ch in s:
        n = n * 12 + _VAL12[ch]
    return n


def parse_token(tok: str) -> tuple[str, str]:
    tok = normalize(tok).replace("'", "").rstrip("-")
    if "." not in tok:
        raise ValueError(f"Bad token: {tok}")
    oct_s, step_s = tok.split(".", 1)
    return oct_s, step_s[:1]


def token_to_abs_step(token: str) -> int:
    oct_s, step = parse_token(token)
    return (bij12_to_int(oct_s) - 1) * 12 + (_VAL12[step] - 1)


def note_root_hz(note: str, anchor_token: str, anchor_hz: float) -> float:
    return anchor_hz * (2.0 ** ((token_to_abs_step(note) - token_to_abs_step(anchor_token)) / 12.0))


def extract_note(folder_name: str) -> str:
    m = re.search(r"([1-9ABC]+\.[1-9ABC]+-)$", folder_name, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot extract note from folder name: {folder_name}")
    return m.group(1).upper()


def is_note_harmonic(freq_hz: float, root_hz: float, max_harmonic: int, tolerance_cents: float) -> tuple[bool, int]:
    for h in range(1, max_harmonic + 1):
        target = root_hz * h
        if abs(cents_error(freq_hz, target)) <= tolerance_cents:
            return True, h
    return False, 0


def load_box_rows(path: Path, min_percent_notes: float, min_amp: float, max_box_hz: float) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            hz = sf(row.get("cluster_hz"))
            percent = sf(row.get("percent_notes"))
            amp = sf(row.get("mean_amp"))
            if hz <= 0:
                continue
            if hz > max_box_hz:
                continue
            if percent < min_percent_notes:
                continue
            if amp < min_amp:
                continue
            rows.append({
                "cluster_hz": hz,
                "cluster_token": row.get("cluster_token", ""),
                "percent_notes": percent,
                "mean_amp": amp,
            })
    return rows


def is_box_component(freq_hz: float, box_rows: list[dict[str, Any]], tolerance_hz: float) -> tuple[bool, dict[str, Any] | None]:
    best = None
    best_delta = 1e18
    for b in box_rows:
        delta = abs(freq_hz - sf(b["cluster_hz"]))
        if delta <= tolerance_hz and delta < best_delta:
            best_delta = delta
            best = b
    return (best is not None), best


def process_dense(
    dense_csv: Path,
    box_csv: Path,
    out_clean_csv: Path,
    out_removed_csv: Path,
    out_summary_txt: Path,
    *,
    expected_note: str,
    anchor_token: str,
    anchor_hz: float,
    max_harmonic: int,
    harmonic_tolerance_cents: float,
    box_tolerance_hz: float,
    min_box_percent_notes: float,
    min_box_amp: float,
    max_box_hz: float,
) -> None:
    box_rows = load_box_rows(
        box_csv,
        min_percent_notes=min_box_percent_notes,
        min_amp=min_box_amp,
        max_box_hz=max_box_hz,
    )

    root_hz = note_root_hz(expected_note, anchor_token, anchor_hz)

    kept = []
    removed = []

    with dense_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])

        extra_fields = [
            "box_removed_reason",
            "box_cluster_hz",
            "box_cluster_token",
            "box_percent_notes",
            "box_mean_amp",
            "note_harmonic_index",
        ]

        for row in reader:
            freq = sf(row.get("freq_hz"))
            is_harm, h = is_note_harmonic(
                freq,
                root_hz,
                max_harmonic=max_harmonic,
                tolerance_cents=harmonic_tolerance_cents,
            )
            is_box, box = is_box_component(freq, box_rows, tolerance_hz=box_tolerance_hz)

            if is_box and not is_harm:
                out = dict(row)
                out["box_removed_reason"] = "BOX_RESIDUAL_NOT_NOTE_HARMONIC"
                out["box_cluster_hz"] = box["cluster_hz"] if box else ""
                out["box_cluster_token"] = box["cluster_token"] if box else ""
                out["box_percent_notes"] = box["percent_notes"] if box else ""
                out["box_mean_amp"] = box["mean_amp"] if box else ""
                out["note_harmonic_index"] = h
                removed.append(out)
            else:
                kept.append(row)

    out_clean_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_clean_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    out_removed_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_removed_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + extra_fields)
        writer.writeheader()
        writer.writerows(removed)

    total = len(kept) + len(removed)
    pct = 100.0 * len(removed) / total if total else 0.0

    lines = []
    lines.append("SUBTRACT BOX FROM DENSE")
    lines.append("=" * 100)
    lines.append(f"dense_csv                  : {dense_csv}")
    lines.append(f"box_csv                    : {box_csv}")
    lines.append(f"expected_note              : {expected_note}")
    lines.append(f"root_hz                    : {root_hz}")
    lines.append(f"box_components_loaded      : {len(box_rows)}")
    lines.append(f"total_rows                 : {total}")
    lines.append(f"kept_rows                  : {len(kept)}")
    lines.append(f"removed_rows               : {len(removed)}")
    lines.append(f"removed_percent            : {pct:.3f}")
    lines.append(f"max_box_hz                 : {max_box_hz}")
    lines.append(f"min_box_percent_notes      : {min_box_percent_notes}")
    lines.append(f"min_box_amp                : {min_box_amp}")
    lines.append(f"box_tolerance_hz           : {box_tolerance_hz}")
    lines.append(f"harmonic_tolerance_cents   : {harmonic_tolerance_cents}")
    lines.append("")
    lines.append("Rule:")
    lines.append("Remove row only if it matches BOX and does NOT match expected note harmonic.")

    out_summary_txt.parent.mkdir(parents=True, exist_ok=True)
    out_summary_txt.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Subtract instrument box components from one dense CSV.")
    ap.add_argument("--dense_csv", required=True)
    ap.add_argument("--box_csv", required=True)
    ap.add_argument("--out_clean_csv", required=True)
    ap.add_argument("--out_removed_csv", required=True)
    ap.add_argument("--out_summary_txt", required=True)
    ap.add_argument("--expected_note", default="")
    ap.add_argument("--anchor_token", default="9.A-")
    ap.add_argument("--anchor_hz", type=float, default=440.0)
    ap.add_argument("--max_harmonic", type=int, default=12)
    ap.add_argument("--harmonic_tolerance_cents", type=float, default=28.0)
    ap.add_argument("--box_tolerance_hz", type=float, default=2.5)
    ap.add_argument("--min_box_percent_notes", type=float, default=70.0)
    ap.add_argument("--min_box_amp", type=float, default=0.0)
    ap.add_argument("--max_box_hz", type=float, default=320.0)
    args = ap.parse_args()

    dense_csv = Path(args.dense_csv).resolve()
    expected_note = args.expected_note.strip() or extract_note(dense_csv.parent.name)

    process_dense(
        dense_csv=dense_csv,
        box_csv=Path(args.box_csv).resolve(),
        out_clean_csv=Path(args.out_clean_csv).resolve(),
        out_removed_csv=Path(args.out_removed_csv).resolve(),
        out_summary_txt=Path(args.out_summary_txt).resolve(),
        expected_note=expected_note,
        anchor_token=args.anchor_token,
        anchor_hz=float(args.anchor_hz),
        max_harmonic=int(args.max_harmonic),
        harmonic_tolerance_cents=float(args.harmonic_tolerance_cents),
        box_tolerance_hz=float(args.box_tolerance_hz),
        min_box_percent_notes=float(args.min_box_percent_notes),
        min_box_amp=float(args.min_box_amp),
        max_box_hz=float(args.max_box_hz),
    )


if __name__ == "__main__":
    main()