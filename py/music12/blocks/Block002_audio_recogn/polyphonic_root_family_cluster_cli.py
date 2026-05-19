# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List


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


def _token_to_hz(token: str, anchor_token: str = "9.A'-", anchor_hz: float = 440.0) -> float | None:
    a = _token_to_abs_degree(anchor_token)
    b = _token_to_abs_degree(token)
    if a is None or b is None:
        return None
    return anchor_hz * (2.0 ** ((b - a) / 12.0))


def _token_to_abs_degree(token: str) -> int | None:
    if not token:
        return None

    token = str(token).strip().upper()
    if "." not in token:
        return None

    octave_part, rest = token.split(".", 1)
    degree_part = rest.split("'", 1)[0].strip()

    octave_value = 0
    for ch in octave_part:
        if ch not in ALPHABET12:
            return None
        octave_value = octave_value * 12 + (ALPHABET12.index(ch) + 1)

    if degree_part not in ALPHABET12:
        return None

    return octave_value * 12 + ALPHABET12.index(degree_part)


def _is_harmonic_relative(low_hz: float, high_hz: float, cents_tolerance: float) -> tuple[bool, int, float]:
    if low_hz <= 0 or high_hz <= 0 or high_hz <= low_hz:
        return False, 0, 999999.0

    ratio = high_hz / low_hz
    nearest = round(ratio)

    if nearest < 2 or nearest > 12:
        return False, nearest, 999999.0

    cents = abs(1200.0 * math.log2(ratio / nearest))
    return cents <= cents_tolerance, nearest, cents


def _cluster_frame(rows: List[Dict[str, Any]], cents_tolerance: float, anchor_token: str, anchor_hz: float):
    items = []

    for r in rows:
        token = str(r.get("root_note", "")).strip()
        hz = _safe_float(r.get("root_hz", ""), 0.0)

        if hz <= 0:
            hz2 = _token_to_hz(token, anchor_token=anchor_token, anchor_hz=anchor_hz)
            hz = hz2 if hz2 else 0.0

        if not token or hz <= 0:
            continue

        items.append({
            "row": r,
            "token": token,
            "hz": hz,
            "score": _safe_float(r.get("adjusted_score", r.get("root_score", 0.0)), 0.0),
        })

    items.sort(key=lambda x: x["hz"])

    families = []

    for item in items:
        placed = False

        for fam in families:
            root = fam["members"][0]
            ok, harmonic, cents = _is_harmonic_relative(root["hz"], item["hz"], cents_tolerance)

            if ok:
                item2 = dict(item)
                item2["relation_to_family_root"] = harmonic
                item2["relation_cents"] = cents
                fam["members"].append(item2)
                fam["family_score"] += item["score"] * (1.0 / harmonic)
                placed = True
                break

        if not placed:
            item2 = dict(item)
            item2["relation_to_family_root"] = 1
            item2["relation_cents"] = 0.0
            families.append({
                "members": [item2],
                "family_score": item["score"],
            })

    families.sort(key=lambda f: f["family_score"], reverse=True)
    return families


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Cluster polyphonic root hypotheses into harmonic root families."
    )

    ap.add_argument("--poly_chain_candidates_csv", required=True)
    ap.add_argument("--out_family_csv", required=True)
    ap.add_argument("--out_frame_summary_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--harmonic_relation_tolerance_cents", type=float, default=35.0)
    ap.add_argument("--max_families_per_frame", type=int, default=8)
    ap.add_argument("--anchor_token", default="9.A'-")
    ap.add_argument("--anchor_hz", type=float, default=440.0)

    args = ap.parse_args()

    in_csv = Path(args.poly_chain_candidates_csv)
    out_family_csv = Path(args.out_family_csv)
    out_frame_summary_csv = Path(args.out_frame_summary_csv)
    out_meta_json = Path(args.out_meta_json)
    out_summary_txt = Path(args.out_summary_txt)

    frames: Dict[int, List[Dict[str, Any]]] = {}

    with in_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            frame = _safe_int(r.get("frame_index", ""), 0)
            frames.setdefault(frame, []).append(r)

    family_rows = []
    summary_rows = []

    for frame_index in sorted(frames):
        rows = frames[frame_index]
        time_sec = _safe_float(rows[0].get("time_sec", ""), 0.0)

        families = _cluster_frame(
            rows,
            cents_tolerance=args.harmonic_relation_tolerance_cents,
            anchor_token=args.anchor_token,
            anchor_hz=args.anchor_hz,
        )[:args.max_families_per_frame]

        top_parts = []

        for family_rank, fam in enumerate(families, start=1):
            members = fam["members"]
            family_root = members[0]
            member_tokens = [m["token"] for m in members]

            top_parts.append(
                f"{family_root['token']}[{','.join(member_tokens)}]:{fam['family_score']:.4f}"
            )

            for m in members:
                family_rows.append({
                    "frame_index": frame_index,
                    "time_sec": f"{time_sec:.9f}",
                    "family_rank": family_rank,
                    "family_root_note": family_root["token"],
                    "family_root_hz": f"{family_root['hz']:.6f}",
                    "family_score": f"{fam['family_score']:.9f}",
                    "member_note": m["token"],
                    "member_hz": f"{m['hz']:.6f}",
                    "member_score": f"{m['score']:.9f}",
                    "relation_to_family_root": m["relation_to_family_root"],
                    "relation_cents": f"{m['relation_cents']:.6f}",
                    "family_members": " ".join(member_tokens),
                })

        summary_rows.append({
            "frame_index": frame_index,
            "time_sec": f"{time_sec:.9f}",
            "family_count": len(families),
            "top_families": " | ".join(top_parts),
        })

    out_family_csv.parent.mkdir(parents=True, exist_ok=True)

    family_fields = [
        "frame_index",
        "time_sec",
        "family_rank",
        "family_root_note",
        "family_root_hz",
        "family_score",
        "member_note",
        "member_hz",
        "member_score",
        "relation_to_family_root",
        "relation_cents",
        "family_members",
    ]

    with out_family_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=family_fields)
        w.writeheader()
        w.writerows(family_rows)

    summary_fields = [
        "frame_index",
        "time_sec",
        "family_count",
        "top_families",
    ]

    with out_frame_summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(summary_rows)

    meta = {
        "stage": "polyphonic_root_family_cluster",
        "input": str(in_csv),
        "outputs": {
            "family_csv": str(out_family_csv),
            "frame_summary_csv": str(out_frame_summary_csv),
            "meta_json": str(out_meta_json),
            "summary_txt": str(out_summary_txt),
        },
        "parameters": {
            "harmonic_relation_tolerance_cents": args.harmonic_relation_tolerance_cents,
            "max_families_per_frame": args.max_families_per_frame,
            "anchor_token": args.anchor_token,
            "anchor_hz": args.anchor_hz,
        },
        "result": {
            "input_frames": len(frames),
            "family_rows": len(family_rows),
            "summary_rows": len(summary_rows),
            "max_family_count": max((_safe_int(r["family_count"], 0) for r in summary_rows), default=0),
        },
    }

    out_meta_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    txt = []
    txt.append("POLYPHONIC ROOT FAMILY CLUSTER")
    txt.append("=" * 72)
    txt.append(f"input             : {in_csv}")
    txt.append(f"family csv        : {out_family_csv}")
    txt.append(f"frame summary     : {out_frame_summary_csv}")
    txt.append(f"frames            : {len(frames)}")
    txt.append(f"family rows       : {len(family_rows)}")
    txt.append(f"max family count  : {meta['result']['max_family_count']}")
    txt.append("")
    txt.append("Principle:")
    txt.append("  Harmonic relatives are grouped into one root-family hypothesis.")
    txt.append("  This reduces octave/harmonic duplication without returning to single-f0 logic.")
    txt.append("")

    out_summary_txt.write_text("\n".join(txt), encoding="utf-8")

    print("polyphonic root family cluster complete")
    print(json.dumps(meta["result"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()