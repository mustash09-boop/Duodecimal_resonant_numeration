# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple


ALPHABET12 = "123456789ABC"


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _parse_candidates_json(raw: str) -> List[Dict[str, Any]]:
    try:
        data = json.loads(raw or "[]")
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def _token_to_degree_abs(token: str) -> int | None:
    """
    Parse note token like 9.A'- into absolute 12-radix degree index.
    This is only for coarse note tokens produced by detail_depth=0.
    """
    if not token:
        return None

    token = str(token).strip().upper()
    if "." not in token:
        return None

    octave_part, rest = token.split(".", 1)
    degree_part = rest.split("'", 1)[0].strip()

    if not octave_part or not degree_part:
        return None

    octave_value = 0
    for ch in octave_part:
        if ch not in ALPHABET12:
            return None
        octave_value = octave_value * 12 + (ALPHABET12.index(ch) + 1)

    degree_symbol = degree_part[0]
    if degree_symbol not in ALPHABET12:
        return None

    degree_index = ALPHABET12.index(degree_symbol)
    return octave_value * 12 + degree_index


def _degree_abs_to_token(abs_degree: int) -> str:
    octave = abs_degree // 12
    degree = abs_degree % 12
    return f"{_octave_to_token12(octave)}.{ALPHABET12[degree]}'-"


def _octave_to_token12(n: int) -> str:
    if n <= 0:
        raise ValueError(f"Invalid octave number: {n!r}")

    if n <= 12:
        return ALPHABET12[n - 1]

    digits = []
    x = n

    while x > 0:
        r = x % 12
        if r == 0:
            digits.append("C")
            x = x // 12 - 1
        else:
            digits.append(ALPHABET12[r - 1])
            x = x // 12

    return "".join(reversed(digits))


def _hz_to_abs_degree(freq_hz: float, anchor_hz: float = 440.0, anchor_token: str = "9.A'-") -> int | None:
    if freq_hz <= 0:
        return None

    anchor_abs = _token_to_degree_abs(anchor_token)
    if anchor_abs is None:
        return None

    semis = round(12.0 * math.log2(freq_hz / anchor_hz))
    return anchor_abs + int(semis)


def _cents_distance(a_hz: float, b_hz: float) -> float:
    if a_hz <= 0 or b_hz <= 0:
        return 999999.0
    return abs(1200.0 * math.log2(a_hz / b_hz))


def _range_mode(root_hz: float) -> str:
    if root_hz < 110.0:
        return "low"
    if root_hz > 1760.0:
        return "high"
    return "mid"


def _range_weight(
    *,
    root_hz: float,
    harmonic_index: int,
    low_note_boost: float,
    high_note_missing_tolerance: float,
) -> float:
    mode = _range_mode(root_hz)

    if mode == "low":
        if harmonic_index in (2, 3, 4, 5):
            return low_note_boost
        return 1.0

    if mode == "high":
        if harmonic_index >= 5:
            return high_note_missing_tolerance
        return 1.0

    return 1.0


def _candidate_lookup(candidates: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for c in candidates:
        note = str(c.get("note_token", "")).strip()
        if note:
            out[note] = c
    return out


def _candidate_energy(c: Dict[str, Any]) -> float:
    return _safe_float(c.get("energy", 0.0), 0.0)


def _score_root(
    *,
    root: Dict[str, Any],
    all_candidates: List[Dict[str, Any]],
    max_harmonic: int,
    tolerance_cents: float,
    anchor_hz: float,
    anchor_token: str,
    low_note_boost: float,
    high_note_missing_tolerance: float,
) -> Dict[str, Any]:
    root_note = str(root.get("note_token", "")).strip()
    root_hz = _safe_float(root.get("frequency_hz", 0.0), 0.0)
    root_energy = _candidate_energy(root)

    evidence = []
    missing = []

    weighted_score = 0.0
    weighted_possible = 0.0

    # root itself gives evidence, but not too much: in polyphony root can be weak
    root_weight = 0.75
    weighted_score += root_energy * root_weight
    weighted_possible += root_weight

    for h in range(2, max_harmonic + 1):
        expected = root_hz * h
        w = _range_weight(
            root_hz=root_hz,
            harmonic_index=h,
            low_note_boost=low_note_boost,
            high_note_missing_tolerance=high_note_missing_tolerance,
        )

        # for high notes, do not strongly penalize absent unreachable harmonics
        weighted_possible += w

        best = None
        best_cents = 999999.0

        for c in all_candidates:
            hz = _safe_float(c.get("frequency_hz", 0.0), 0.0)
            cents = _cents_distance(hz, expected)
            if cents <= tolerance_cents and cents < best_cents:
                best = c
                best_cents = cents

        if best is None:
            missing.append(h)
            continue

        e = _candidate_energy(best)
        note = str(best.get("note_token", "")).strip()

        weighted_score += e * w
        evidence.append({
            "harmonic": h,
            "expected_hz": expected,
            "matched_note": note,
            "matched_hz": _safe_float(best.get("frequency_hz", 0.0), 0.0),
            "matched_energy": e,
            "cents": best_cents,
            "weight": w,
        })

    normalized = weighted_score / max(weighted_possible, 1e-9)

    return {
        "root_note": root_note,
        "root_hz": root_hz,
        "root_energy": root_energy,
        "range_mode": _range_mode(root_hz),
        "root_score": normalized,
        "evidence_count": len(evidence),
        "evidence_notes": " ".join(sorted(set(e["matched_note"] for e in evidence if e["matched_note"]))),
        "missing_harmonics": " ".join(str(x) for x in missing),
        "evidence_json": json.dumps(evidence, ensure_ascii=False),
    }


def _apply_collision_penalty(
    scored: List[Dict[str, Any]],
    collision_penalty: float,
) -> List[Dict[str, Any]]:
    """
    Penalize roots that share too many evidence notes with stronger roots.
    This is soft; we do not delete early, only reduce score.
    """
    accepted_evidence_sets = []

    out = []
    for r in sorted(scored, key=lambda x: x["root_score"], reverse=True):
        notes = set(str(r.get("evidence_notes", "")).split())
        overlap_count = 0

        for s in accepted_evidence_sets:
            overlap_count += len(notes & s)

        penalty = collision_penalty * overlap_count
        adjusted = max(0.0, float(r["root_score"]) - penalty)

        r = dict(r)
        r["collision_overlap_count"] = overlap_count
        r["collision_penalty"] = penalty
        r["adjusted_score"] = adjusted

        out.append(r)

        if adjusted > 0:
            accepted_evidence_sets.append(notes)

    return sorted(out, key=lambda x: x["adjusted_score"], reverse=True)


def build_polyphonic_chains(
    *,
    framewise_csv: Path,
    out_chain_candidates_csv: Path,
    out_frame_summary_csv: Path,
    out_meta_json: Path,
    out_summary_txt: Path,
    max_roots_per_frame: int,
    max_candidate_notes: int,
    max_harmonic: int,
    tolerance_cents: float,
    min_root_score: float,
    collision_penalty: float,
    low_note_boost: float,
    high_note_missing_tolerance: float,
    anchor_hz: float,
    anchor_token: str,
) -> None:
    rows_out = []
    summary_rows = []

    with framewise_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            frame_index = _safe_int(row.get("frame_index", ""), 0)
            time_sec = _safe_float(row.get("time_sec", ""), 0.0)

            candidates = _parse_candidates_json(row.get("selected_candidates_json", "[]"))
            candidates = candidates[:max_candidate_notes]

            scored = []
            for c in candidates:
                scored.append(
                    _score_root(
                        root=c,
                        all_candidates=candidates,
                        max_harmonic=max_harmonic,
                        tolerance_cents=tolerance_cents,
                        anchor_hz=anchor_hz,
                        anchor_token=anchor_token,
                        low_note_boost=low_note_boost,
                        high_note_missing_tolerance=high_note_missing_tolerance,
                    )
                )

            scored = _apply_collision_penalty(scored, collision_penalty=collision_penalty)

            selected = [
                r for r in scored
                if float(r["adjusted_score"]) >= min_root_score
            ][:max_roots_per_frame]

            for rank, r in enumerate(selected, start=1):
                out = {
                    "frame_index": frame_index,
                    "time_sec": f"{time_sec:.9f}",
                    "root_rank": rank,
                    "root_note": r["root_note"],
                    "root_hz": f"{r['root_hz']:.6f}",
                    "range_mode": r["range_mode"],
                    "root_energy": f"{r['root_energy']:.9f}",
                    "root_score": f"{r['root_score']:.9f}",
                    "adjusted_score": f"{r['adjusted_score']:.9f}",
                    "evidence_count": r["evidence_count"],
                    "evidence_notes": r["evidence_notes"],
                    "missing_harmonics": r["missing_harmonics"],
                    "collision_overlap_count": r["collision_overlap_count"],
                    "collision_penalty": f"{r['collision_penalty']:.9f}",
                    "evidence_json": r["evidence_json"],
                }
                rows_out.append(out)

            summary_rows.append({
                "frame_index": frame_index,
                "time_sec": f"{time_sec:.9f}",
                "selected_root_count": len(selected),
                "top_roots": " | ".join(
                    f"{r['root_note']}:{r['adjusted_score']:.4f}"
                    for r in selected
                ),
            })

    out_chain_candidates_csv.parent.mkdir(parents=True, exist_ok=True)

    chain_fields = [
        "frame_index",
        "time_sec",
        "root_rank",
        "root_note",
        "root_hz",
        "range_mode",
        "root_energy",
        "root_score",
        "adjusted_score",
        "evidence_count",
        "evidence_notes",
        "missing_harmonics",
        "collision_overlap_count",
        "collision_penalty",
        "evidence_json",
    ]

    with out_chain_candidates_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=chain_fields)
        w.writeheader()
        w.writerows(rows_out)

    summary_fields = [
        "frame_index",
        "time_sec",
        "selected_root_count",
        "top_roots",
    ]

    with out_frame_summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(summary_rows)

    meta = {
        "stage": "polyphonic_harmonic_chain_builder",
        "input": str(framewise_csv),
        "outputs": {
            "chain_candidates_csv": str(out_chain_candidates_csv),
            "frame_summary_csv": str(out_frame_summary_csv),
            "meta_json": str(out_meta_json),
            "summary_txt": str(out_summary_txt),
        },
        "parameters": {
            "max_roots_per_frame": max_roots_per_frame,
            "max_candidate_notes": max_candidate_notes,
            "max_harmonic": max_harmonic,
            "tolerance_cents": tolerance_cents,
            "min_root_score": min_root_score,
            "collision_penalty": collision_penalty,
            "low_note_boost": low_note_boost,
            "high_note_missing_tolerance": high_note_missing_tolerance,
            "anchor_hz": anchor_hz,
            "anchor_token": anchor_token,
        },
        "result": {
            "chain_candidate_rows": len(rows_out),
            "frame_rows": len(summary_rows),
            "max_selected_roots_in_frame": max(
                (_safe_int(r["selected_root_count"], 0) for r in summary_rows),
                default=0,
            ),
        },
    }

    out_meta_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    txt = []
    txt.append("POLYPHONIC HARMONIC CHAIN BUILDER")
    txt.append("=" * 72)
    txt.append(f"input                     : {framewise_csv}")
    txt.append(f"chain candidates          : {out_chain_candidates_csv}")
    txt.append(f"frame summary             : {out_frame_summary_csv}")
    txt.append(f"frames                    : {len(summary_rows)}")
    txt.append(f"chain candidate rows      : {len(rows_out)}")
    txt.append(f"max roots per frame       : {meta['result']['max_selected_roots_in_frame']}")
    txt.append("")
    txt.append("Range compensation:")
    txt.append(f"  low_note_boost              : {low_note_boost}")
    txt.append(f"  high_note_missing_tolerance : {high_note_missing_tolerance}")
    txt.append("")
    txt.append("Principle:")
    txt.append("  This stage does NOT select one f0.")
    txt.append("  It creates multiple simultaneous root hypotheses per frame.")
    txt.append("  Low and high registers are compensated differently.")
    txt.append("")

    out_summary_txt.write_text("\n".join(txt), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build polyphonic harmonic chain hypotheses from framewise resonance candidates."
    )

    ap.add_argument("--framewise_csv", required=True)
    ap.add_argument("--out_chain_candidates_csv", required=True)
    ap.add_argument("--out_frame_summary_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--max_roots_per_frame", type=int, default=8)
    ap.add_argument("--max_candidate_notes", type=int, default=16)
    ap.add_argument("--max_harmonic", type=int, default=8)

    ap.add_argument("--tolerance_cents", type=float, default=35.0)
    ap.add_argument("--min_root_score", type=float, default=0.20)
    ap.add_argument("--collision_penalty", type=float, default=0.03)

    ap.add_argument("--low_note_boost", type=float, default=1.35)
    ap.add_argument("--high_note_missing_tolerance", type=float, default=0.55)

    ap.add_argument("--anchor_hz", type=float, default=440.0)
    ap.add_argument("--anchor_token", default="9.A'-")

    args = ap.parse_args()

    build_polyphonic_chains(
        framewise_csv=Path(args.framewise_csv),
        out_chain_candidates_csv=Path(args.out_chain_candidates_csv),
        out_frame_summary_csv=Path(args.out_frame_summary_csv),
        out_meta_json=Path(args.out_meta_json),
        out_summary_txt=Path(args.out_summary_txt),
        max_roots_per_frame=args.max_roots_per_frame,
        max_candidate_notes=args.max_candidate_notes,
        max_harmonic=args.max_harmonic,
        tolerance_cents=args.tolerance_cents,
        min_root_score=args.min_root_score,
        collision_penalty=args.collision_penalty,
        low_note_boost=args.low_note_boost,
        high_note_missing_tolerance=args.high_note_missing_tolerance,
        anchor_hz=args.anchor_hz,
        anchor_token=args.anchor_token,
    )

    print("polyphonic harmonic chain builder complete")
    print(json.dumps({
        "out_chain_candidates_csv": args.out_chain_candidates_csv,
        "out_frame_summary_csv": args.out_frame_summary_csv,
        "out_meta_json": args.out_meta_json,
        "out_summary_txt": args.out_summary_txt,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()