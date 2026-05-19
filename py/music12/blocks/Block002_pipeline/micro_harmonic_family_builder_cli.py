# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple


ALPHABET12 = "123456789ABC"
_VAL12 = {ch: i + 1 for i, ch in enumerate(ALPHABET12)}
_CH12 = {i + 1: ch for i, ch in enumerate(ALPHABET12)}


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


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _normalize_letters(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("Рђ", "A").replace("Р’", "B").replace("РЎ", "C")
    s = s.replace("Р°", "A").replace("РІ", "B").replace("СЃ", "C")
    return s


def _bij12_to_int(s: str) -> int | None:
    s = _normalize_letters(s).upper()
    if not s or any(ch not in _VAL12 for ch in s):
        return None
    n = 0
    for ch in s:
        n = n * 12 + _VAL12[ch]
    return n


def _int_to_base12_digit(i0: int) -> str:
    i0 = int(i0)
    if not 0 <= i0 < 12:
        raise ValueError("base12 digit index must be 0..11")
    return _CH12[i0 + 1]


def _int_to_bij12(n: int) -> str:
    n = int(n)
    if n <= 0:
        raise ValueError("_int_to_bij12 expects n >= 1")
    out: list[str] = []
    while n > 0:
        n, r = divmod(n - 1, 12)
        out.append(_CH12[r + 1])
    return "".join(reversed(out))


def _split_token_micro(token: str) -> Tuple[str, str]:
    token = _normalize_letters(token).strip()
    token = token.replace("вЂ™-", "'-")

    if "'" not in token:
        return token, ""

    coarse, micro = token.split("'", 1)
    return coarse, micro


def _parse_coarse_token(token: str) -> tuple[str, str] | None:
    coarse, _micro = _split_token_micro(token)
    coarse = _normalize_letters(coarse).upper()

    if "." not in coarse:
        return None

    octave_raw, degree_raw = coarse.split(".", 1)
    degree_raw = degree_raw[:1]

    if not octave_raw or any(ch not in _VAL12 for ch in octave_raw):
        return None

    if degree_raw not in _VAL12:
        return None

    return octave_raw, degree_raw


def _token_to_abs_degree_coarse(token: str) -> int | None:
    parsed = _parse_coarse_token(token)
    if parsed is None:
        return None

    octave_raw, degree_raw = parsed
    octave = _bij12_to_int(octave_raw)
    if octave is None:
        return None

    return octave * 12 + ALPHABET12.index(degree_raw)


def _micro_suffix_to_fraction_semitones(micro: str, *, micro_depth: int = 2) -> float:
    micro = _normalize_letters(micro).strip().upper()

    if not micro or micro == "-":
        return 0.0

    direction = micro[0].lower()
    if direction not in ("i", "a"):
        return 0.0

    digits = micro[1:]
    if not digits or any(ch not in _VAL12 for ch in digits):
        return 0.0

    depth = max(1, int(micro_depth))
    digits = digits[:depth]

    n = 0
    for ch in digits:
        n = n * 12 + (_VAL12[ch] - 1)

    denom = 12 ** len(digits)
    frac = float(n) / float(denom)

    return frac if direction == "i" else -frac


def _token_to_hz(
    token: str,
    anchor_token: str,
    anchor_hz: float,
    *,
    micro_depth: int = 2,
) -> float | None:
    a = _token_to_abs_degree_coarse(anchor_token)
    b = _token_to_abs_degree_coarse(token)

    if a is None or b is None:
        return None

    _anchor_coarse, anchor_micro = _split_token_micro(anchor_token)
    _token_coarse_local, token_micro = _split_token_micro(token)

    anchor_micro_delta = _micro_suffix_to_fraction_semitones(anchor_micro, micro_depth=micro_depth)
    token_micro_delta = _micro_suffix_to_fraction_semitones(token_micro, micro_depth=micro_depth)
    semitone_delta = float(b - a) + token_micro_delta - anchor_micro_delta

    return float(anchor_hz * (2.0 ** (semitone_delta / 12.0)))


def _hz_to_token_with_micro(
    freq_hz: float,
    *,
    anchor_token: str = "9.A'-",
    anchor_hz: float = 440.0,
    micro_depth: int = 2,
) -> str:
    if freq_hz <= 0:
        return ""

    anchor_abs = _token_to_abs_degree_coarse(anchor_token)
    if anchor_abs is None:
        anchor_abs = _token_to_abs_degree_coarse("9.A'-") or 0

    _anchor_coarse, anchor_micro = _split_token_micro(anchor_token)
    anchor_micro_delta = _micro_suffix_to_fraction_semitones(anchor_micro, micro_depth=micro_depth)
    semitone_offset = 12.0 * math.log2(freq_hz / anchor_hz)
    abs_float = float(anchor_abs) + anchor_micro_delta + semitone_offset

    nearest_abs = int(round(abs_float))
    residual = abs_float - float(nearest_abs)
    octave, degree0 = divmod(nearest_abs, 12)
    coarse = f"{_int_to_bij12(octave)}.{_int_to_base12_digit(degree0)}"

    depth = max(1, int(micro_depth))
    steps_per_semitone = 12 ** depth
    micro_steps = int(round(residual * steps_per_semitone))

    if micro_steps == 0:
        return coarse + "'-"

    sign = "i" if micro_steps > 0 else "a"
    magnitude = abs(micro_steps)

    while magnitude >= steps_per_semitone:
        nearest_abs += 1 if sign == "i" else -1
        magnitude -= steps_per_semitone
        octave, degree0 = divmod(nearest_abs, 12)
        coarse = f"{_int_to_bij12(octave)}.{_int_to_base12_digit(degree0)}"

    if magnitude == 0:
        return coarse + "'-"

    digits: list[str] = []
    remaining = int(magnitude)

    for power in reversed(range(depth)):
        denom = 12 ** power
        digit0 = remaining // denom
        remaining %= denom
        digit0 = max(0, min(11, int(digit0)))
        digits.append(_int_to_base12_digit(digit0))

    return coarse + "'" + sign + "".join(digits)


def _token_coarse(token: str) -> str:
    coarse, _micro = _split_token_micro(token)
    return coarse


def _cents_distance(a_hz: float, b_hz: float) -> float:
    if a_hz <= 0 or b_hz <= 0:
        return 999999.0
    return abs(1200.0 * math.log2(a_hz / b_hz))


def _group_by_frame(rows: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    for r in rows:
        frame = _safe_int(r.get("frame_index"), 0)
        out.setdefault(frame, []).append(r)
    return out


def _pick_token_from_row(row: Dict[str, Any]) -> str:
    for key in (
        "anchor_token_micro",
        "note_token_micro",
        "root_note_token_micro",
        "matched_token_micro",
        "theoretical_token_micro",
        "anchor_token",
        "note_token",
        "root_note_token",
        "matched_token",
        "theoretical_token",
    ):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def _build_families_for_frame(
    rows: List[Dict[str, Any]],
    *,
    anchor_token: str,
    anchor_hz: float,
    max_harmonic: int,
    tolerance_cents: float,
    max_families_per_frame: int,
    micro_depth: int,
) -> List[Dict[str, Any]]:
    items: list[dict[str, Any]] = []

    for r in rows:
        token_micro = _pick_token_from_row(r)
        if not token_micro:
            continue

        hz = _token_to_hz(token_micro, anchor_token, anchor_hz, micro_depth=micro_depth)
        if hz is None:
            continue

        token_coarse = _token_coarse(token_micro)

        items.append(
            {
                "token_micro": token_micro,
                "token_coarse": token_coarse,
                "hz": hz,
                "cluster_energy": _safe_float(r.get("cluster_energy"), 0.0),
                "max_energy": _safe_float(r.get("max_energy"), 0.0),
                "micro_count": _safe_int(r.get("micro_count"), 0),
                "micro_diversity": _safe_int(r.get("micro_diversity"), 0),
                "micro_members": str(r.get("micro_members", "")).strip(),
                "source_row_json": json.dumps(r, ensure_ascii=False),
            }
        )

    items.sort(key=lambda x: -x["cluster_energy"])
    families: list[dict[str, Any]] = []

    for root in items:
        root_hz = root["hz"]
        evidence: list[dict[str, Any]] = []
        family_score = root["cluster_energy"]

        for h in range(2, max_harmonic + 1):
            expected_hz = root_hz * h
            expected_token_micro = _hz_to_token_with_micro(
                expected_hz,
                anchor_token=anchor_token,
                anchor_hz=anchor_hz,
                micro_depth=micro_depth,
            )
            expected_token_coarse = _token_coarse(expected_token_micro)

            best = None
            best_cents = 999999.0

            for cand in items:
                if cand["token_micro"] == root["token_micro"]:
                    continue

                cents = _cents_distance(cand["hz"], expected_hz)

                if cents <= tolerance_cents and cents < best_cents:
                    best = cand
                    best_cents = cents

            if best is not None:
                weight = 1.0 / h
                family_score += best["cluster_energy"] * weight

                evidence.append(
                    {
                        "harmonic": h,
                        "expected_hz": expected_hz,
                        "expected_token_micro": expected_token_micro,
                        "expected_token_coarse": expected_token_coarse,
                        "member_token_micro": best["token_micro"],
                        "member_token_coarse": best["token_coarse"],
                        "member_hz": best["hz"],
                        "member_energy": best["cluster_energy"],
                        "cents": best_cents,
                        "micro_count": best["micro_count"],
                        "micro_diversity": best["micro_diversity"],
                        "micro_members": best["micro_members"],
                    }
                )

        members_micro = [root["token_micro"]] + [e["member_token_micro"] for e in evidence]
        members_coarse = [root["token_coarse"]] + [e["member_token_coarse"] for e in evidence]

        families.append(
            {
                "family_root_note_micro": root["token_micro"],
                "family_root_note_coarse": root["token_coarse"],
                "family_root_hz": root_hz,
                "family_score": family_score,
                "root_cluster_energy": root["cluster_energy"],
                "evidence_count": len(evidence),
                "family_members_micro": " ".join(sorted(set(members_micro))),
                "family_members_coarse": " ".join(sorted(set(members_coarse))),
                "evidence_json": json.dumps(evidence, ensure_ascii=False),
                "root_micro_count": root["micro_count"],
                "root_micro_diversity": root["micro_diversity"],
                "root_micro_members": root["micro_members"],
            }
        )

    families.sort(key=lambda x: (-x["family_score"], -x["evidence_count"]))
    return families[:max_families_per_frame]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build harmonic families from micro-preserved resonance clusters. "
            "This module keeps micro/coarse identities separate."
        )
    )

    ap.add_argument("--micro_clusters_csv", required=True)
    ap.add_argument("--out_family_csv", required=True)
    ap.add_argument("--out_frame_summary_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)
    ap.add_argument("--anchor_token", default="9.A'-")
    ap.add_argument("--anchor_hz", type=float, default=440.0)
    ap.add_argument("--max_harmonic", type=int, default=8)
    ap.add_argument("--tolerance_cents", type=float, default=35.0)
    ap.add_argument("--max_families_per_frame", type=int, default=12)
    ap.add_argument("--micro_depth", type=int, default=2)

    args = ap.parse_args()

    in_csv = Path(args.micro_clusters_csv)
    out_family = Path(args.out_family_csv)
    out_frame = Path(args.out_frame_summary_csv)
    out_meta = Path(args.out_meta_json)
    out_txt = Path(args.out_summary_txt)

    rows = _load_csv(in_csv)
    by_frame = _group_by_frame(rows)

    family_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []

    for frame in sorted(by_frame):
        families = _build_families_for_frame(
            by_frame[frame],
            anchor_token=args.anchor_token,
            anchor_hz=args.anchor_hz,
            max_harmonic=args.max_harmonic,
            tolerance_cents=args.tolerance_cents,
            max_families_per_frame=args.max_families_per_frame,
            micro_depth=args.micro_depth,
        )

        for rank, fam in enumerate(families, start=1):
            row = {
                "frame_index": frame,
                "family_rank": rank,
                **fam,
            }
            family_rows.append(row)

        frame_rows.append(
            {
                "frame_index": frame,
                "family_count": len(families),
                "top_families_micro": " | ".join(
                    f"{f['family_root_note_micro']}:{f['family_score']:.3f}"
                    for f in families[:8]
                ),
                "top_families_coarse": " | ".join(
                    f"{f['family_root_note_coarse']}:{f['family_score']:.3f}"
                    for f in families[:8]
                ),
            }
        )

    out_family.parent.mkdir(parents=True, exist_ok=True)

    family_fields = list(family_rows[0].keys()) if family_rows else [
        "frame_index",
        "family_rank",
        "family_root_note_micro",
        "family_root_note_coarse",
        "family_root_hz",
        "family_score",
        "root_cluster_energy",
        "evidence_count",
        "family_members_micro",
        "family_members_coarse",
        "evidence_json",
        "root_micro_count",
        "root_micro_diversity",
        "root_micro_members",
    ]

    with out_family.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=family_fields)
        w.writeheader()
        w.writerows(family_rows)

    with out_frame.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "frame_index",
                "family_count",
                "top_families_micro",
                "top_families_coarse",
            ],
        )
        w.writeheader()
        w.writerows(frame_rows)

    meta = {
        "stage": "micro_harmonic_family_builder",
        "semantic_version": "micro_preserved_v2",
        "inputs": {
            "micro_clusters_csv": str(in_csv),
        },
        "outputs": {
            "family_csv": str(out_family),
            "frame_summary_csv": str(out_frame),
            "meta_json": str(out_meta),
            "summary_txt": str(out_txt),
        },
        "parameters": {
            "anchor_token": args.anchor_token,
            "anchor_hz": args.anchor_hz,
            "max_harmonic": args.max_harmonic,
            "tolerance_cents": args.tolerance_cents,
            "max_families_per_frame": args.max_families_per_frame,
            "micro_depth": args.micro_depth,
        },
        "result": {
            "input_cluster_rows": len(rows),
            "frames": len(by_frame),
            "family_rows": len(family_rows),
            "max_family_count": max(
                (_safe_int(r["family_count"], 0) for r in frame_rows),
                default=0,
            ),
        },
        "ontology_note": (
            "This version does not intentionally collapse micro suffixes. "
            "It keeps family_root_note_micro and family_root_note_coarse as separate fields. "
            "Family members and evidence preserve micro tokens where input rows provide them."
        ),
    }

    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    txt = [
        "MICRO HARMONIC FAMILY BUILDER",
        "=" * 72,
        f"micro_clusters_csv : {in_csv}",
        f"family_csv         : {out_family}",
        f"frame_summary_csv  : {out_frame}",
        "",
        f"input_cluster_rows : {len(rows)}",
        f"frames             : {len(by_frame)}",
        f"family_rows        : {len(family_rows)}",
        f"max_family_count   : {meta['result']['max_family_count']}",
        "",
        "Principle:",
        "  Build harmonic families from micro-preserved resonance clusters.",
        "  Keep micro identity and coarse identity in separate fields.",
        "  Do not collapse family_root_note_micro into family_root_note_coarse.",
        "",
    ]

    out_txt.write_text("\n".join(txt), encoding="utf-8")

    print("micro harmonic family builder complete")
    print(json.dumps(meta["result"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
