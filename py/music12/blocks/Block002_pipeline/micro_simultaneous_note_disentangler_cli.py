# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


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


def _token_to_abs_degree(token: str) -> Optional[int]:
    try:
        token = str(token).strip().upper()
        octave_raw, rest = token.split(".", 1)
        degree_raw = rest.split("'", 1)[0]

        octave = 0
        for ch in octave_raw:
            if ch not in ALPHABET12:
                return None
            octave = octave * 12 + (ALPHABET12.index(ch) + 1)

        if degree_raw not in ALPHABET12:
            return None

        return octave * 12 + ALPHABET12.index(degree_raw)
    except Exception:
        return None


def _pitch_distance(a: str, b: str) -> float:
    aa = _token_to_abs_degree(a)
    bb = _token_to_abs_degree(b)

    if aa is None or bb is None:
        return 9999.0

    return abs(float(aa - bb))


def _load_centers(path: Path, min_center_score: float) -> Dict[str, Dict[str, Any]]:
    rows = _load_csv(path)
    centers: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        node = _normalize_note(row.get("node", ""))
        score = _safe_float(row.get("center_score"), 0.0)
        if not node or score < min_center_score:
            continue
        centers[node] = {
            "center_score": score,
            "causal_role": str(row.get("causal_role", "")).strip(),
        }

    return centers


def _load_roles(path: str | None) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}

    rows = _load_csv(Path(path))
    roles: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        node = _normalize_note(row.get("node", ""))
        if not node:
            continue
        roles[node] = {
            "center_score": _safe_float(row.get("center_score"), 0.0),
            "causal_role": str(row.get("causal_role", "")).strip(),
            "asymmetry": _safe_float(row.get("asymmetry"), 0.0),
        }
    return roles


def _group_families_by_frame(rows: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        frame = _safe_int(row.get("frame_index"), 0)
        out.setdefault(frame, []).append(row)
    return out


def _family_root_token(row: Dict[str, Any]) -> str:
    return (
        str(row.get("family_root_note_micro", "")).strip()
        or str(row.get("family_root_note", "")).strip()
        or str(row.get("family_root_note_coarse", "")).strip()
    )


def _structural_support(
    family_score: float,
    evidence_count: int,
    root_micro_count: int,
    root_micro_diversity: int,
) -> float:
    support = family_score * 0.55
    support += min(max(evidence_count, 0), 4) * 0.10
    support += min(max(root_micro_count, 0), 12) * 0.015
    support += min(max(root_micro_diversity, 0), 12) * 0.015
    return support


def _collapse_candidates(notes: List[Dict[str, Any]], max_per_anchor: int = 1) -> List[Dict[str, Any]]:
    by_anchor: Dict[str, List[Dict[str, Any]]] = {}
    for note in notes:
        by_anchor.setdefault(str(note["note_token"]), []).append(note)

    out: List[Dict[str, Any]] = []
    for _, items in by_anchor.items():
        items.sort(
            key=lambda row: (
                -_safe_float(row["score"]),
                -_safe_float(row["center_score"]),
                -_safe_float(row["structural_support"]),
                -_safe_int(row["evidence_count"]),
            )
        )
        out.extend(items[:max_per_anchor])

    out.sort(
        key=lambda row: (
            -_safe_float(row["score"]),
            row["note_token"],
        )
    )
    return out


def _is_exciter_family(role: str) -> bool:
    return role in {
        "dominant_exciter",
        "exciter_like",
        "bridge_exciter_like",
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Disentangle simultaneous note candidates using causal centers."
    )

    ap.add_argument("--micro_family_csv", required=True)
    ap.add_argument("--causal_centers_csv", required=True)
    ap.add_argument("--roles_csv")

    ap.add_argument("--out_frame_notes_csv", required=True)
    ap.add_argument("--out_readable_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--min_center_score", type=float, default=0.010)
    ap.add_argument("--min_family_score", type=float, default=0.16)
    ap.add_argument("--min_structural_support", type=float, default=0.34)
    ap.add_argument("--min_structural_root_micro_count", type=int, default=6)
    ap.add_argument("--min_structural_diversity", type=int, default=4)
    ap.add_argument("--max_notes_per_frame", type=int, default=10)
    ap.add_argument("--max_per_degree", type=int, default=1)
    ap.add_argument("--max_structural_companions_per_center", type=int, default=1)
    ap.add_argument("--max_structural_companions_without_center", type=int, default=1)

    args = ap.parse_args()

    family_rows = _load_csv(Path(args.micro_family_csv))
    centers = _load_centers(Path(args.causal_centers_csv), args.min_center_score)
    roles = _load_roles(args.roles_csv)

    by_frame = _group_families_by_frame(family_rows)
    out_rows: List[Dict[str, Any]] = []
    readable_rows: List[Dict[str, Any]] = []

    max_active_notes = 0
    frames_with_notes = 0
    causal_center_candidates = 0
    structural_companions = 0

    for frame in sorted(by_frame):
        candidates: List[Dict[str, Any]] = []

        for row in by_frame[frame]:
            root = _normalize_note(_family_root_token(row))
            if not root:
                continue

            family_score = _safe_float(row.get("family_score"), 0.0)
            evidence_count = _safe_int(row.get("evidence_count"), 0)
            root_micro_count = _safe_int(row.get("root_micro_count"), 0)
            root_micro_diversity = _safe_int(row.get("root_micro_diversity"), 0)
            structural_support = _structural_support(
                family_score,
                evidence_count,
                root_micro_count,
                root_micro_diversity,
            )

            center_info = centers.get(root)
            role_info = roles.get(root, {})
            center_score = _safe_float(
                (center_info or role_info).get("center_score", 0.0),
                0.0,
            )
            causal_role = str((center_info or role_info).get("causal_role", "")).strip()

            is_structural_companion = (
                structural_support >= args.min_structural_support
                and root_micro_count >= args.min_structural_root_micro_count
                and root_micro_diversity >= args.min_structural_diversity
            )

            if family_score < args.min_family_score and not is_structural_companion:
                continue

            if center_info is None and not is_structural_companion:
                continue

            center_bonus = center_score * 0.75
            role_bonus = 0.0
            if causal_role == "dominant_exciter":
                role_bonus += 0.10
            elif causal_role == "exciter_like":
                role_bonus += 0.06
            elif causal_role == "bridge_exciter_like":
                role_bonus += 0.04
            elif causal_role == "bridge_resonator":
                role_bonus += 0.015
            elif causal_role == "bridge_response_like":
                role_bonus -= 0.035
            elif causal_role == "response_like":
                role_bonus -= 0.05
            elif causal_role == "response_sink":
                role_bonus -= 0.08

            structural_bonus = 0.0
            structural_bonus += min(root_micro_count, 12) * 0.006
            structural_bonus += min(root_micro_diversity, 12) * 0.006
            structural_bonus += min(evidence_count, 4) * 0.03

            score = family_score + center_bonus + role_bonus + structural_bonus
            candidate_kind = "CAUSAL_CENTER"
            if center_info is None:
                score -= 0.03
                candidate_kind = "STRUCTURAL_COMPANION"
            elif is_structural_companion:
                candidate_kind = "CENTER_WITH_STRUCTURAL_SUPPORT"

            candidates.append({
                "note_token": root,
                "score": score,
                "family_score": family_score,
                "center_score": center_score,
                "causal_role": causal_role or "untyped",
                "candidate_kind": candidate_kind,
                "structural_support": structural_support,
                "evidence_count": evidence_count,
                "root_micro_count": root_micro_count,
                "root_micro_diversity": root_micro_diversity,
                "degree": _degree(root),
                "octave": _octave(root),
            })

        strong_local_centers = [
            candidate for candidate in candidates
            if candidate["candidate_kind"] != "STRUCTURAL_COMPANION"
            and _is_exciter_family(str(candidate["causal_role"]))
            and _safe_float(candidate["center_score"]) >= args.min_center_score
        ]

        for candidate in candidates:
            if candidate["candidate_kind"] != "STRUCTURAL_COMPANION":
                continue

            local_penalty = 0.0
            for center_candidate in strong_local_centers:
                pitch_gap = _pitch_distance(
                    str(candidate["note_token"]),
                    str(center_candidate["note_token"]),
                )
                if pitch_gap > 4.0:
                    continue

                center_advantage = (
                    _safe_float(center_candidate["center_score"])
                    - _safe_float(candidate["center_score"])
                )
                local_penalty = max(
                    local_penalty,
                    0.07 + max(0.0, center_advantage) * 0.35,
                )

            if local_penalty > 0.0:
                candidate["score"] -= local_penalty
                candidate["local_center_penalty"] = local_penalty
            else:
                candidate["local_center_penalty"] = 0.0

        candidates = _collapse_candidates(candidates, args.max_per_degree)
        center_candidates = [
            candidate for candidate in candidates
            if candidate["candidate_kind"] != "STRUCTURAL_COMPANION"
        ]
        structural_candidates = [
            candidate for candidate in candidates
            if candidate["candidate_kind"] == "STRUCTURAL_COMPANION"
        ]

        structural_limit = args.max_structural_companions_without_center
        if center_candidates:
            structural_limit = max(
                args.max_structural_companions_without_center,
                len(center_candidates) * args.max_structural_companions_per_center,
            )

        candidates = center_candidates + structural_candidates[:structural_limit]
        candidates.sort(
            key=lambda row: (
                -_safe_float(row["score"]),
                row["note_token"],
            )
        )
        candidates = candidates[: args.max_notes_per_frame]

        if candidates:
            frames_with_notes += 1

        max_active_notes = max(max_active_notes, len(candidates))

        for rank, candidate in enumerate(candidates, start=1):
            if candidate["candidate_kind"] == "STRUCTURAL_COMPANION":
                structural_companions += 1
            else:
                causal_center_candidates += 1

            out_rows.append({
                "frame_index": frame,
                "rank": rank,
                "note_token": candidate["note_token"],
                "score": f"{candidate['score']:.9f}",
                "family_score": f"{candidate['family_score']:.9f}",
                "center_score": f"{candidate['center_score']:.9f}",
                "structural_support": f"{candidate['structural_support']:.9f}",
                "candidate_kind": candidate["candidate_kind"],
                "causal_role": candidate["causal_role"],
                "local_center_penalty": f"{_safe_float(candidate.get('local_center_penalty', 0.0)):.9f}",
                "evidence_count": candidate["evidence_count"],
                "root_micro_count": candidate["root_micro_count"],
                "root_micro_diversity": candidate["root_micro_diversity"],
                "degree": candidate["degree"],
                "octave": candidate["octave"],
            })

        readable_rows.append({
            "frame_index": frame,
            "active_note_count": len(candidates),
            "notes": " | ".join(
                f"{candidate['note_token']}:{candidate['score']:.3f}:{candidate['candidate_kind']}"
                for candidate in candidates
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
        "center_score",
        "structural_support",
        "candidate_kind",
        "causal_role",
        "local_center_penalty",
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
    for row in readable_rows:
        active_note_count = _safe_int(row.get("active_note_count"), 0)
        active_counts[active_note_count] = active_counts.get(active_note_count, 0) + 1

    meta = {
        "stage": "micro_simultaneous_note_disentangler",
        "inputs": {
            "micro_family_csv": args.micro_family_csv,
            "causal_centers_csv": args.causal_centers_csv,
            "roles_csv": args.roles_csv,
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
            "min_structural_support": args.min_structural_support,
            "min_structural_root_micro_count": args.min_structural_root_micro_count,
            "min_structural_diversity": args.min_structural_diversity,
            "max_notes_per_frame": args.max_notes_per_frame,
            "max_per_degree": args.max_per_degree,
            "max_structural_companions_per_center": args.max_structural_companions_per_center,
            "max_structural_companions_without_center": args.max_structural_companions_without_center,
        },
        "result": {
            "family_rows": len(family_rows),
            "causal_centers": len(centers),
            "frame_note_rows": len(out_rows),
            "frames": len(readable_rows),
            "frames_with_notes": frames_with_notes,
            "max_active_notes": max_active_notes,
            "causal_center_candidates": causal_center_candidates,
            "structural_companions": structural_companions,
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
        f"roles_csv           : {args.roles_csv or '-'}",
        "",
        f"family_rows         : {len(family_rows)}",
        f"causal_centers      : {len(centers)}",
        f"frame_note_rows     : {len(out_rows)}",
        f"frames              : {len(readable_rows)}",
        f"frames_with_notes   : {frames_with_notes}",
        f"max_active_notes    : {max_active_notes}",
        f"causal_center_rows  : {causal_center_candidates}",
        f"structural_companions: {structural_companions}",
        "",
        "Active note distribution:",
    ]

    for key in sorted(active_counts):
        txt.append(f"  {key}: {active_counts[key]}")

    txt.extend([
        "",
        "Principle:",
        "  Simultaneous notes are selected first as causal centers,",
        "  but structurally strong chain companions may survive even",
        "  when the center layer is still incomplete.",
        "",
    ])

    out_txt.write_text("\n".join(txt), encoding="utf-8")

    print("micro simultaneous note disentangler complete")
    print(json.dumps(meta["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
