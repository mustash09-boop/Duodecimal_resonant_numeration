from __future__ import annotations

import argparse
import csv
import json
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


def si(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
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
        if ch not in _VAL12:
            raise ValueError(f"Bad bij12 char: {ch!r} in {s!r}")
        n = n * 12 + _VAL12[ch]
    return n


def parse_token(tok: str) -> tuple[str, str]:
    tok = normalize(tok).replace("'", "").rstrip("-")
    if "." not in tok:
        raise ValueError(f"Bad note token: {tok!r}")
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


def classify_range(root_hz: float) -> str:
    if root_hz < 95.0:
        return "LOW"
    if root_hz < 980.0:
        return "MID"
    return "HIGH"


def visible_harmonics(root_hz: float, max_harmonic: int, max_freq_hz: float) -> list[int]:
    return [h for h in range(1, max_harmonic + 1) if root_hz * h <= max_freq_hz]


def protected_harmonics_for_range(root_hz: float, note_range: str, max_harmonic: int, max_freq_hz: float) -> list[int]:
    visible = visible_harmonics(root_hz, max_harmonic, max_freq_hz)

    if note_range == "LOW":
        # Для низа фундаментал может быть слабым, но подводящие области h1..h12 важны.
        return visible

    if note_range == "MID":
        # В середине цепь хорошо развёрнута.
        return visible

    # HIGH: цепь короткая; но сохраняем все видимые гармонические зоны,
    # чтобы не убить редкие, но физически значимые частичные.
    return visible


def is_near_note_harmonic(
    freq_hz: float,
    root_hz: float,
    harmonics: list[int],
    tolerance_cents: float,
) -> tuple[bool, int, float]:
    best_h = 0
    best_abs = 1e18
    best_delta = 0.0

    for h in harmonics:
        target = root_hz * h
        delta = cents_error(freq_hz, target)
        ad = abs(delta)
        if ad < best_abs:
            best_abs = ad
            best_h = h
            best_delta = delta

    if best_abs <= tolerance_cents:
        return True, best_h, best_delta
    return False, best_h, best_delta


def load_box_rows(
    box_csv: Path,
    *,
    min_percent_notes: float,
    min_amp: float,
    max_box_hz: float,
    min_box_hz: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with box_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            hz = sf(row.get("cluster_hz"))
            percent = sf(row.get("percent_notes"))
            amp = sf(row.get("mean_amp"))

            if hz <= 0:
                continue
            if hz < min_box_hz or hz > max_box_hz:
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


def nearest_box_component(freq_hz: float, box_rows: list[dict[str, Any]], tolerance_hz: float) -> tuple[bool, dict[str, Any] | None, float]:
    best = None
    best_delta = 1e18

    for b in box_rows:
        d = abs(freq_hz - sf(b["cluster_hz"]))
        if d <= tolerance_hz and d < best_delta:
            best_delta = d
            best = b

    return best is not None, best, best_delta


def load_dense(dense_csv: Path) -> tuple[list[str], list[dict[str, Any]]]:
    with dense_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        fieldnames = list(r.fieldnames or [])
        rows = [dict(row) for row in r]
    return fieldnames, rows


def process(
    *,
    dense_csv: Path,
    box_csv: Path,
    out_clean_csv: Path,
    out_removed_csv: Path,
    out_summary_txt: Path,
    expected_note: str,
    anchor_token: str,
    anchor_hz: float,
    max_harmonic: int,
    max_freq_hz: float,
    protect_tolerance_cents_low: float,
    protect_tolerance_cents_mid: float,
    protect_tolerance_cents_high: float,
    box_tolerance_hz: float,
    min_box_percent_notes: float,
    min_box_amp: float,
    min_box_hz: float,
    max_box_hz: float,
) -> None:
    root_hz = note_root_hz(expected_note, anchor_token, anchor_hz)
    note_range = classify_range(root_hz)

    if note_range == "LOW":
        protect_tolerance = protect_tolerance_cents_low
    elif note_range == "MID":
        protect_tolerance = protect_tolerance_cents_mid
    else:
        protect_tolerance = protect_tolerance_cents_high

    protected_h = protected_harmonics_for_range(root_hz, note_range, max_harmonic, max_freq_hz)

    box_rows = load_box_rows(
        box_csv,
        min_percent_notes=min_box_percent_notes,
        min_amp=min_box_amp,
        min_box_hz=min_box_hz,
        max_box_hz=max_box_hz,
    )

    fieldnames, dense_rows = load_dense(dense_csv)

    clean_rows: list[dict[str, Any]] = []
    removed_rows: list[dict[str, Any]] = []

    extra = [
        "cleaner_decision",
        "note_range",
        "expected_note",
        "root_hz",
        "protected_harmonic_index",
        "protected_delta_cents",
        "box_cluster_hz",
        "box_cluster_token",
        "box_percent_notes",
        "box_mean_amp",
        "box_delta_hz",
    ]

    for row in dense_rows:
        freq = sf(row.get("freq_hz"))

        is_protected, h, delta_cents = is_near_note_harmonic(
            freq,
            root_hz,
            protected_h,
            tolerance_cents=protect_tolerance,
        )

        is_box, box, box_delta = nearest_box_component(
            freq,
            box_rows,
            tolerance_hz=box_tolerance_hz,
        )

        annotated = dict(row)
        annotated["note_range"] = note_range
        annotated["expected_note"] = expected_note
        annotated["root_hz"] = root_hz
        annotated["protected_harmonic_index"] = h if is_protected else ""
        annotated["protected_delta_cents"] = delta_cents if is_protected else ""
        annotated["box_cluster_hz"] = box["cluster_hz"] if box else ""
        annotated["box_cluster_token"] = box["cluster_token"] if box else ""
        annotated["box_percent_notes"] = box["percent_notes"] if box else ""
        annotated["box_mean_amp"] = box["mean_amp"] if box else ""
        annotated["box_delta_hz"] = box_delta if box else ""

        if is_box and not is_protected:
            annotated["cleaner_decision"] = "REMOVE_BOX_NOT_PROTECTED"
            removed_rows.append(annotated)
        else:
            annotated["cleaner_decision"] = "KEEP_PROTECTED_OR_NOT_BOX"
            clean_rows.append(annotated)

    out_clean_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_clean_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames + extra)
        w.writeheader()
        w.writerows(clean_rows)

    out_removed_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_removed_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames + extra)
        w.writeheader()
        w.writerows(removed_rows)

    total = len(clean_rows) + len(removed_rows)
    removed_pct = 100.0 * len(removed_rows) / total if total else 0.0

    lines = []
    lines.append("UNIFIED DENSE NOTE CLEANER")
    lines.append("=" * 100)
    lines.append(f"dense_csv                    : {dense_csv}")
    lines.append(f"box_csv                      : {box_csv}")
    lines.append(f"expected_note                : {expected_note}")
    lines.append(f"root_hz                      : {root_hz}")
    lines.append(f"note_range                   : {note_range}")
    lines.append(f"protected_harmonics          : {protected_h}")
    lines.append(f"protect_tolerance_cents      : {protect_tolerance}")
    lines.append(f"box_components_loaded        : {len(box_rows)}")
    lines.append(f"total_rows                   : {total}")
    lines.append(f"clean_rows                   : {len(clean_rows)}")
    lines.append(f"removed_rows                 : {len(removed_rows)}")
    lines.append(f"removed_percent              : {removed_pct:.3f}")
    lines.append("")
    lines.append("Rule:")
    lines.append("Remove only if frequency matches instrument box AND is not inside protected note-harmonic zone.")
    lines.append("Everything else remains, including transitional/non-harmonic motion around the note.")

    out_summary_txt.parent.mkdir(parents=True, exist_ok=True)
    out_summary_txt.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified dense cleaner: remove instrument box while preserving note-forming motion.")
    ap.add_argument("--dense_csv", required=True)
    ap.add_argument("--box_csv", required=True)
    ap.add_argument("--out_clean_csv", required=True)
    ap.add_argument("--out_removed_csv", required=True)
    ap.add_argument("--out_summary_txt", required=True)
    ap.add_argument("--expected_note", default="")
    ap.add_argument("--anchor_token", default="9.A-")
    ap.add_argument("--anchor_hz", type=float, default=440.0)
    ap.add_argument("--max_harmonic", type=int, default=12)
    ap.add_argument("--max_freq_hz", type=float, default=21000.0)

    ap.add_argument("--protect_tolerance_cents_low", type=float, default=45.0)
    ap.add_argument("--protect_tolerance_cents_mid", type=float, default=32.0)
    ap.add_argument("--protect_tolerance_cents_high", type=float, default=28.0)

    ap.add_argument("--box_tolerance_hz", type=float, default=2.5)
    ap.add_argument("--min_box_percent_notes", type=float, default=70.0)
    ap.add_argument("--min_box_amp", type=float, default=0.0)
    ap.add_argument("--min_box_hz", type=float, default=10.0)
    ap.add_argument("--max_box_hz", type=float, default=320.0)

    args = ap.parse_args()

    dense_csv = Path(args.dense_csv).resolve()
    expected_note = args.expected_note.strip().upper() or extract_note(dense_csv.parent.name)

    process(
        dense_csv=dense_csv,
        box_csv=Path(args.box_csv).resolve(),
        out_clean_csv=Path(args.out_clean_csv).resolve(),
        out_removed_csv=Path(args.out_removed_csv).resolve(),
        out_summary_txt=Path(args.out_summary_txt).resolve(),
        expected_note=expected_note,
        anchor_token=args.anchor_token,
        anchor_hz=float(args.anchor_hz),
        max_harmonic=int(args.max_harmonic),
        max_freq_hz=float(args.max_freq_hz),
        protect_tolerance_cents_low=float(args.protect_tolerance_cents_low),
        protect_tolerance_cents_mid=float(args.protect_tolerance_cents_mid),
        protect_tolerance_cents_high=float(args.protect_tolerance_cents_high),
        box_tolerance_hz=float(args.box_tolerance_hz),
        min_box_percent_notes=float(args.min_box_percent_notes),
        min_box_amp=float(args.min_box_amp),
        min_box_hz=float(args.min_box_hz),
        max_box_hz=float(args.max_box_hz),
    )


if __name__ == "__main__":
    main()