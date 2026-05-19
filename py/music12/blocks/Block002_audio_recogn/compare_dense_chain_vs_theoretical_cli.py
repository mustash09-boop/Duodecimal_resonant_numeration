from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


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


def expected_note_from_name(name: str) -> str:
    m = re.match(r"^\d+_piano_midi_(.+)$", name)
    if m:
        return m.group(1)

    m = re.match(r"^\d+_(.+)$", name)
    if m:
        return m.group(1)

    return ""


def load_theoretical_csv(path: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            note = str(row["expected_note"]).strip()
            out.setdefault(note, []).append(row)

    for note in out:
        out[note].sort(key=lambda r: safe_int(r["harmonic_index"]))
    return out


def load_dense_summary_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def classify_status(observed_hz: float | None, lower_hz: float, upper_hz: float) -> str:
    if observed_hz is None:
        return "MISSING"
    if lower_hz <= observed_hz <= upper_hz:
        return "MATCH"
    if observed_hz < lower_hz:
        return "SHIFTED_DOWN"
    if observed_hz > upper_hz:
        return "SHIFTED_UP"
    return "UNKNOWN"


def compare_dense_best_chain_to_theory(
    *,
    summary_json: Path,
    theoretical_csv: Path,
    expected_note: str,
) -> list[dict[str, Any]]:

    theory = load_theoretical_csv(theoretical_csv)
    if expected_note not in theory:
        raise ValueError(f"Expected note {expected_note!r} not found in theoretical CSV")

    summary = load_dense_summary_json(summary_json)

    # 🔥 ГЛАВНАЯ ПРАВКА
    best = summary.get("best_chain") or summary.get("best_track")

    if not best:
        return []

    detected_root_note = str(best.get("root_note_token", "")).strip()

    detected_root_hz = safe_float(
        best.get("root_hz")
        or best.get("root_hz_mean")
        or best.get("root_hz_median")
        or 0.0
    )

    chain_score = safe_float(
        best.get("chain_score")
        or best.get("chain_score_mean")
        or best.get("track_score")
        or 0.0
    )

    # 🔥 ПРАВКА ДЛЯ HITS
    hits_by_h = {}
    for hit in (best.get("hits") or best.get("representative_hits") or []):
        h = safe_int(hit.get("harmonic_index", 0))
        if h > 0:
            hits_by_h[h] = hit

    rows: list[dict[str, Any]] = []

    for th in theory[expected_note]:
        h = safe_int(th["harmonic_index"])
        theoretical_hz = safe_float(th["theoretical_hz"])
        lower_hz = safe_float(th["lower_hz_tolerance"])
        upper_hz = safe_float(th["upper_hz_tolerance"])

        hit = hits_by_h.get(h)

        if hit is None:
            observed_hz = None
            observed_token = ""
            observed_amp = ""
            observed_phase = ""
            delta_hz = ""
            delta_cents = ""
            status = "MISSING"
        else:
            observed_hz = safe_float(hit.get("matched_hz", 0.0))
            observed_token = str(hit.get("matched_token", "")).strip()
            observed_amp = safe_float(hit.get("matched_amplitude", 0.0))
            observed_phase = safe_float(hit.get("matched_phase_rad", 0.0))
            delta_hz = observed_hz - theoretical_hz
            delta_cents = safe_float(hit.get("delta_cents", 0.0))
            status = classify_status(observed_hz, lower_hz, upper_hz)

        rows.append(
            {
                "expected_note": expected_note,
                "detected_root_note": detected_root_note,
                "detected_root_hz": detected_root_hz,
                "chain_score": chain_score,
                "harmonic_index": h,
                "theoretical_token": str(th["theoretical_token"]).strip(),
                "theoretical_hz": theoretical_hz,
                "lower_token_tolerance": str(th["lower_token_tolerance"]).strip(),
                "upper_token_tolerance": str(th["upper_token_tolerance"]).strip(),
                "lower_hz_tolerance": lower_hz,
                "upper_hz_tolerance": upper_hz,
                "observed_token": observed_token,
                "observed_hz": "" if observed_hz is None else observed_hz,
                "observed_amplitude": observed_amp,
                "observed_phase_rad": observed_phase,
                "delta_hz": delta_hz,
                "delta_cents": delta_cents,
                "status": status,
                "summary_json": str(summary_json),
            }
        )

    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_txt(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("DENSE CHAIN VS THEORETICAL REPORT")
    lines.append("=" * 120)
    lines.append("")

    if not rows:
        lines.append("No rows.")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    first = rows[0]
    lines.append(f"expected_note     : {first['expected_note']}")
    lines.append(f"detected_root_note: {first['detected_root_note']}")
    lines.append(f"detected_root_hz  : {first['detected_root_hz']}")
    lines.append(f"chain_score       : {first['chain_score']}")
    lines.append("")

    for row in rows:
        lines.append(
            f"h{row['harmonic_index']}: "
            f"theory={row['theoretical_token']} ({row['theoretical_hz']:.3f}) "
            f"obs={row['observed_token']} ({row['observed_hz']}) "
            f"delta_cents={row['delta_cents']} "
            f"status={row['status']}"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare dense-chain best result against theoretical harmonics"
    )

    ap.add_argument("--dense_chain_summary_json", required=True)
    ap.add_argument("--theoretical_csv", required=True)
    ap.add_argument("--expected_note", default="")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_txt", required=True)

    args = ap.parse_args()

    dense_chain_summary_json = Path(args.dense_chain_summary_json).resolve()
    theoretical_csv = Path(args.theoretical_csv).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_txt = Path(args.out_txt).resolve()

    expected_note = str(args.expected_note).strip()
    if not expected_note:
        expected_note = expected_note_from_name(dense_chain_summary_json.stem)

    if not expected_note:
        raise ValueError("Expected note not found")

    rows = compare_dense_best_chain_to_theory(
        summary_json=dense_chain_summary_json,
        theoretical_csv=theoretical_csv,
        expected_note=expected_note,
    )

    write_csv(out_csv, rows)
    write_txt(out_txt, rows)

    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()