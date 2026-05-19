# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Set


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


def _degree(token: str) -> str:
    try:
        return token.split(".", 1)[1].split("'", 1)[0]
    except Exception:
        return ""


def _octave(token: str) -> str:
    try:
        return token.split(".", 1)[0]
    except Exception:
        return ""


def _normalize_note(token: str) -> str:
    s = str(token or "").strip()
    if not s:
        return ""
    if "'" in s:
        return s.split("'", 1)[0] + "'-"
    if s.endswith("-"):
        return s[:-1] + "'-"
    return s + "'-"


def _load_centers(path: Path, min_center_score: float) -> Set[str]:
    rows = _load_csv(path)
    centers: Set[str] = set()

    for r in rows:
        node = _normalize_note(r.get("node", ""))
        score = _safe_float(r.get("center_score"), 0.0)

        if node and score >= min_center_score:
            centers.add(node)

    return centers


def _group_families_by_frame(rows: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}

    for r in rows:
        frame = _safe_int(r.get("frame_index"), 0)
        out.setdefault(frame, []).append(r)

    return out


def _collapse_same_degree(notes: List[Dict[str, Any]], max_per_degree: int = 1) -> List[Dict[str, Any]]:
    """
    Keep strongest causal candidate per degree by default.
    This prevents one pitch-class ladder from pretending to be multiple simultaneous notes.
    """
    by_degree: Dict[str, List[Dict[str, Any]]] = {}

    for n in notes:
        by_degree.setdefault(_degree(n["note_token"]), []).append(n)

    out = []

    for degree, items in by_degree.items():
        items.sort(
            key=lambda r: (
                -_safe_float(r["score"]),
                -_safe_int(r["evidence_count"]),
            )
        )
        out.extend(items[:max_per_degree])

    out.sort(
        key=lambda r: (
            -_safe_float(r["score"]),
            r["note_token"],
        )
    )

    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Disentangle simultaneous note candidates using causal centers."
    )

    ap.add_argument("--micro_family_csv", required=True)
    ap.add_argument("--causal_centers_csv", required=True)

    ap.add_argument("--out_frame_notes_csv", required=True)
    ap.add_argument("--out_readable_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--min_center_score", type=float, default=0.015)
    ap.add_argument("--min_family_score", type=float, default=0.20)
    ap.add_argument("--max_notes_per_frame", type=int, default=8)
    ap.add_argument("--max_per_degree", type=int, default=1)

    args = ap.parse_args()

    family_rows = _load_csv(Path(args.micro_family_csv))
    centers = _load_centers(Path(args.causal_centers_csv), args.min_center_score)

    by_frame = _group_families_by_frame(family_rows)

    out_rows = []
    readable_rows = []

    max_active_notes = 0
    frames_with_notes = 0

    for frame in sorted(by_frame):
        candidates = []

        for r in by_frame[frame]:
            root = _normalize_note(r.get("family_root_note", ""))

            if not root:
                continue

            if root not in centers:
                continue

            family_score = _safe_float(r.get("family_score"), 0.0)

            if family_score < args.min_family_score:
                continue

            evidence_count = _safe_int(r.get("evidence_count"), 0)
            root_micro_count = _safe_int(r.get("root_micro_count"), 0)
            root_micro_diversity = _safe_int(r.get("root_micro_diversity"), 0)

            causal_bonus = 0.0
            if root_micro_count >= 3:
                causal_bonus += 0.05
            if root_micro_diversity >= 3:
                causal_bonus += 0.05
            if evidence_count >= 2:
                causal_bonus += 0.08

            score = family_score + causal_bonus

            candidates.append({
                "note_token": root,
                "score": score,
                "family_score": family_score,
                "evidence_count": evidence_count,
                "root_micro_count": root_micro_count,
                "root_micro_diversity": root_micro_diversity,
            })

        candidates = _collapse_same_degree(candidates, args.max_per_degree)
        candidates = candidates[: args.max_notes_per_frame]

        if candidates:
            frames_with_notes += 1

        max_active_notes = max(max_active_notes, len(candidates))

        for rank, c in enumerate(candidates, start=1):
            out_rows.append({
                "frame_index": frame,
                "rank": rank,
                "note_token": c["note_token"],
                "score": f"{c['score']:.9f}",
                "family_score": f"{c['family_score']:.9f}",
                "evidence_count": c["evidence_count"],
                "root_micro_count": c["root_micro_count"],
                "root_micro_diversity": c["root_micro_diversity"],
                "degree": _degree(c["note_token"]),
                "octave": _octave(c["note_token"]),
            })

        readable_rows.append({
            "frame_index": frame,
            "active_note_count": len(candidates),
            "notes": " | ".join(
                f"{c['note_token']}:{c['score']:.3f}"
                for c in candidates
            ),
        })

    out_frame = Path(args.out_frame_notes_csv)
    out_readable = Path(args.out_readable_csv)
    out_meta = Path(args.out_meta_json)
    out_txt = Path(args.out_summary_txt)

    out_frame.parent.mkdir(parents=True, exist_ok=True)

    frame_fields = [
        "frame_index",
        "rank",
        "note_token",
        "score",
        "family_score",
        "evidence_count",
        "root_micro_count",
        "root_micro_diversity",
        "degree",
        "octave",
    ]

    with out_frame.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=frame_fields)
        w.writeheader()
        w.writerows(out_rows)

    with out_readable.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "frame_index",
                "active_note_count",
                "notes",
            ],
        )
        w.writeheader()
        w.writerows(readable_rows)

    active_counts: Dict[int, int] = {}
    for r in readable_rows:
        n = _safe_int(r.get("active_note_count"), 0)
        active_counts[n] = active_counts.get(n, 0) + 1

    meta = {
        "stage": "micro_simultaneous_note_disentangler",
        "inputs": {
            "micro_family_csv": args.micro_family_csv,
            "causal_centers_csv": args.causal_centers_csv,
        },
        "outputs": {
            "frame_notes_csv": args.out_frame_notes_csv,
            "readable_csv": args.out_readable_csv,
            "meta_json": args.out_meta_json,
            "summary_txt": args.out_summary_txt,
        },
        "parameters": {
            "min_center_score": args.min_center_score,
            "min_family_score": args.min_family_score,
            "max_notes_per_frame": args.max_notes_per_frame,
            "max_per_degree": args.max_per_degree,
        },
        "result": {
            "family_rows": len(family_rows),
            "causal_centers": len(centers),
            "frame_note_rows": len(out_rows),
            "frames": len(readable_rows),
            "frames_with_notes": frames_with_notes,
            "max_active_notes": max_active_notes,
            "active_count_distribution": active_counts,
        },
    }

    out_meta.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    txt = [
        "MICRO SIMULTANEOUS NOTE DISENTANGLER",
        "=" * 72,
        f"micro_family_csv    : {args.micro_family_csv}",
        f"causal_centers_csv  : {args.causal_centers_csv}",
        "",
        f"family_rows         : {len(family_rows)}",
        f"causal_centers      : {len(centers)}",
        f"frame_note_rows     : {len(out_rows)}",
        f"frames              : {len(readable_rows)}",
        f"frames_with_notes   : {frames_with_notes}",
        f"max_active_notes    : {max_active_notes}",
        "",
        "Active note distribution:",
    ]

    for k in sorted(active_counts):
        txt.append(f"  {k}: {active_counts[k]}")

    txt.extend([
        "",
        "Principle:",
        "  Simultaneous notes are selected as active causal centers",
        "  inside the micro-resonance field, not as independent spectral peaks.",
        "",
    ])

    out_txt.write_text("\n".join(txt), encoding="utf-8")

    print("micro simultaneous note disentangler complete")
    print(json.dumps(meta["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()