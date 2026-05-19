# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Set


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


def _normalize_note(token: str) -> str:
    s = str(token or "").strip()
    if not s:
        return ""
    if "'" in s:
        return s.split("'", 1)[0] + "'-"
    if s.endswith("-"):
        return s[:-1] + "'-"
    return s + "'-"


def _degree(token: str) -> str:
    try:
        return _normalize_note(token).split(".", 1)[1].split("'", 1)[0]
    except Exception:
        return ""


def _octave_raw(token: str) -> str:
    try:
        return _normalize_note(token).split(".", 1)[0]
    except Exception:
        return ""


def _octave_value(token: str) -> int:
    raw = _octave_raw(token)
    value = 0
    for ch in raw:
        if ch not in ALPHABET12:
            continue
        value = value * 12 + (ALPHABET12.index(ch) + 1)
    return value


def _range_mode(token: str) -> str:
    ov = _octave_value(token)
    if ov <= 6:
        return "low"
    if ov >= 10:
        return "high"
    return "mid"


def _note_from_profile_filename(path: Path) -> str:
    left = path.name.split("__note_box_profile", 1)[0]
    raw = left.split("_")[-1]
    return _normalize_note(raw)


def _load_passports(folder: Path) -> Dict[str, Dict[str, Any]]:
    passports: Dict[str, Dict[str, Any]] = {}

    for p in sorted(folder.glob("*__note_box_profile.csv")):
        note = _note_from_profile_filename(p)
        rows = _load_csv(p)

        tokens: Set[str] = set()
        strong_tokens: Set[str] = set()
        persistent_tokens: Set[str] = set()
        echo_like_tokens: Set[str] = set()

        amp_values = []
        presence_values = []

        for r in rows:
            token = _normalize_note(r.get("token", ""))
            if not token:
                continue

            amp = _safe_float(r.get("mean_amp"), 0.0)
            presence = _safe_float(r.get("presence_ratio"), 0.0)

            tokens.add(token)
            amp_values.append(amp)
            presence_values.append(presence)

            if amp >= 0.16 and presence >= 0.06:
                strong_tokens.add(token)

            if presence >= 0.16:
                persistent_tokens.add(token)

            if presence >= 0.10 and amp < 0.05:
                echo_like_tokens.add(token)

        passports[note] = {
            "note": note,
            "tokens": tokens,
            "strong_tokens": strong_tokens,
            "persistent_tokens": persistent_tokens,
            "echo_like_tokens": echo_like_tokens,
            "mean_amp": sum(amp_values) / max(len(amp_values), 1),
            "mean_presence": sum(presence_values) / max(len(presence_values), 1),
        }

    return passports


def _find_passport(note: str, passports: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    n = _normalize_note(note)

    if n in passports:
        return passports[n]

    d = _degree(n)

    # Fallback: same degree, closest octave.
    best = None
    best_dist = 999999

    for pnote, prof in passports.items():
        if _degree(pnote) != d:
            continue

        dist = abs(_octave_value(pnote) - _octave_value(n))
        if dist < best_dist:
            best = prof
            best_dist = dist

    if best is not None:
        return best

    return {
        "note": n,
        "tokens": set(),
        "strong_tokens": set(),
        "persistent_tokens": set(),
        "echo_like_tokens": set(),
        "mean_amp": 0.0,
        "mean_presence": 0.0,
    }


def _load_roles(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = _load_csv(path)
    out = {}

    for r in rows:
        node = _normalize_note(r.get("node", ""))
        if not node:
            continue

        out[node] = {
            "causal_role": str(r.get("causal_role", "")).strip(),
            "out_weight": _safe_float(r.get("out_weight"), 0.0),
            "in_weight": _safe_float(r.get("in_weight"), 0.0),
            "center_score": _safe_float(r.get("center_score"), 0.0),
        }

    return out


def _load_family_features(path: Path) -> Dict[tuple[int, str], Dict[str, Any]]:
    rows = _load_csv(path)
    out = {}

    for r in rows:
        frame = _safe_int(r.get("frame_index"), 0)
        root = _normalize_note(r.get("family_root_note", ""))
        if not root:
            continue

        out[(frame, root)] = {
            "evidence_count": _safe_int(r.get("evidence_count"), 0),
            "family_score": _safe_float(r.get("family_score"), 0.0),
            "root_micro_count": _safe_int(r.get("root_micro_count"), 0),
            "root_micro_diversity": _safe_int(r.get("root_micro_diversity"), 0),
            "family_members": str(r.get("family_members", "")).strip(),
        }

    return out


def _family_members(raw: str) -> Set[str]:
    return {_normalize_note(x) for x in str(raw or "").split() if x.strip()}


def _passport_score(
    *,
    note: str,
    frame: int,
    base_score: float,
    role: Dict[str, Any],
    family: Dict[str, Any],
    passport: Dict[str, Any],
) -> Dict[str, Any]:

    range_mode = _range_mode(note)

    role_name = role.get("causal_role", "")
    out_w = _safe_float(role.get("out_weight"), 0.0)
    in_w = _safe_float(role.get("in_weight"), 0.0)
    center_score = _safe_float(role.get("center_score"), 0.0)

    evidence_count = _safe_int(family.get("evidence_count"), 0)
    root_micro_count = _safe_int(family.get("root_micro_count"), 0)
    root_micro_diversity = _safe_int(family.get("root_micro_diversity"), 0)
    members = _family_members(family.get("family_members", ""))

    strong_overlap = len(members & passport["strong_tokens"])
    persistent_overlap = len(members & passport["persistent_tokens"])
    echo_overlap = len(members & passport["echo_like_tokens"])

    score = base_score

    # Causal source ownership.
    score += center_score * 0.80
    score += out_w * 0.35
    score -= in_w * 0.30

    if role_name in {"response_sink", "response_like"}:
        score -= 0.65

    if role_name == "feedback_bridge":
        score += 0.08

    if role_name == "dominant_exciter":
        score += 0.25

    # Passport agreement.
    score += strong_overlap * 0.18
    score += persistent_overlap * 0.04
    score -= echo_overlap * 0.16

    # Range-aware rules from single-note research.
    if range_mode == "low":
        # Low roots may be weak; odd/partial evidence and micro richness matter.
        score += min(evidence_count, 4) * 0.08
        score += min(root_micro_diversity, 6) * 0.035
        if evidence_count == 0 and root_micro_count < 3:
            score -= 0.25

    elif range_mode == "mid":
        # Mid range is overlap-heavy; demand stronger causal ownership.
        score += min(evidence_count, 3) * 0.05
        if center_score < 0.015:
            score -= 0.20

    else:
        # High range: weak ladders are normal; do not over-penalize missing evidence.
        score += min(root_micro_count, 5) * 0.03
        score += min(root_micro_diversity, 5) * 0.03

    return {
        "passport_ownership_score": max(score, 0.0),
        "range_mode": range_mode,
        "causal_role": role_name,
        "strong_overlap": strong_overlap,
        "persistent_overlap": persistent_overlap,
        "echo_overlap": echo_overlap,
        "evidence_count": evidence_count,
        "root_micro_count": root_micro_count,
        "root_micro_diversity": root_micro_diversity,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Filter simultaneous notes using single-note passports, causal roles, range-aware rules, box and echo ownership."
    )

    ap.add_argument("--frame_notes_csv", required=True)
    ap.add_argument("--micro_family_csv", required=True)
    ap.add_argument("--causal_roles_csv", required=True)
    ap.add_argument("--passport_folder", required=True)

    ap.add_argument("--out_frame_notes_csv", required=True)
    ap.add_argument("--out_readable_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--min_ownership_score", type=float, default=0.85)
    ap.add_argument("--max_notes_per_frame", type=int, default=5)
    ap.add_argument("--max_notes_low_mid", type=int, default=4)

    args = ap.parse_args()

    frame_rows = _load_csv(Path(args.frame_notes_csv))
    roles = _load_roles(Path(args.causal_roles_csv))
    families = _load_family_features(Path(args.micro_family_csv))
    passports = _load_passports(Path(args.passport_folder))

    by_frame: Dict[int, List[Dict[str, Any]]] = {}

    for r in frame_rows:
        frame = _safe_int(r.get("frame_index"), 0)
        by_frame.setdefault(frame, []).append(r)

    out_rows = []
    readable_rows = []

    kept_total = 0
    rejected_total = 0
    frames_with_notes = 0
    max_kept = 0

    reject_reasons: Dict[str, int] = {}

    for frame in sorted(by_frame):
        scored = []

        for r in by_frame[frame]:
            note = _normalize_note(r.get("note_token", ""))
            base_score = _safe_float(r.get("score"), 0.0)

            role = roles.get(note, {})
            family = families.get((frame, note), {})
            passport = _find_passport(note, passports)

            ps = _passport_score(
                note=note,
                frame=frame,
                base_score=base_score,
                role=role,
                family=family,
                passport=passport,
            )

            row = dict(r)
            row["note_token"] = note
            row["passport_ownership_score"] = f"{ps['passport_ownership_score']:.9f}"
            row["range_mode"] = ps["range_mode"]
            row["causal_role"] = ps["causal_role"]
            row["strong_overlap"] = ps["strong_overlap"]
            row["persistent_overlap"] = ps["persistent_overlap"]
            row["echo_overlap"] = ps["echo_overlap"]
            row["evidence_count"] = ps["evidence_count"]
            row["root_micro_count"] = ps["root_micro_count"]
            row["root_micro_diversity"] = ps["root_micro_diversity"]

            scored.append(row)

        scored.sort(
            key=lambda x: (
                -_safe_float(x.get("passport_ownership_score"), 0.0),
                -_safe_float(x.get("score"), 0.0),
            )
        )

        # Soft density control: keep strongest causal owners only.
        kept = []

        for row in scored:
            own = _safe_float(row.get("passport_ownership_score"), 0.0)
            role_name = str(row.get("causal_role", ""))

            if own < args.min_ownership_score:
                rejected_total += 1
                reject_reasons["LOW_OWNERSHIP_SCORE"] = reject_reasons.get("LOW_OWNERSHIP_SCORE", 0) + 1
                continue

            if role_name in {"response_sink", "response_like"}:
                rejected_total += 1
                reject_reasons["RESPONSE_ROLE"] = reject_reasons.get("RESPONSE_ROLE", 0) + 1
                continue

            kept.append(row)

        # Keep fewer notes unless there is strong evidence.
        kept = kept[: args.max_notes_per_frame]

        if len(kept) > args.max_notes_low_mid:
            strong = [
                x for x in kept
                if _safe_float(x.get("passport_ownership_score"), 0.0) >= args.min_ownership_score + 0.35
            ]
            if len(strong) >= args.max_notes_low_mid:
                kept = strong[: args.max_notes_per_frame]
            else:
                kept = kept[: args.max_notes_low_mid]

        if kept:
            frames_with_notes += 1

        max_kept = max(max_kept, len(kept))
        kept_total += len(kept)

        for rank, row in enumerate(kept, start=1):
            rr = dict(row)
            rr["filtered_rank"] = rank
            out_rows.append(rr)

        readable_rows.append({
            "frame_index": frame,
            "active_note_count": len(kept),
            "notes": " | ".join(
                f"{r['note_token']}:{_safe_float(r.get('passport_ownership_score'), 0.0):.3f}[{r.get('range_mode')}/{r.get('causal_role')}]"
                for r in kept
            ),
        })

    out_frame = Path(args.out_frame_notes_csv)
    out_readable = Path(args.out_readable_csv)
    out_meta = Path(args.out_meta_json)
    out_txt = Path(args.out_summary_txt)

    out_frame.parent.mkdir(parents=True, exist_ok=True)

    fields = list(out_rows[0].keys()) if out_rows else []

    with out_frame.open("w", encoding="utf-8", newline="") as f:
        if fields:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(out_rows)
        else:
            f.write("")

    with out_readable.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame_index", "active_note_count", "notes"])
        w.writeheader()
        w.writerows(readable_rows)

    active_distribution: Dict[int, int] = {}
    for r in readable_rows:
        n = _safe_int(r.get("active_note_count"), 0)
        active_distribution[n] = active_distribution.get(n, 0) + 1

    meta = {
        "stage": "single_note_passport_ownership_filter",
        "inputs": {
            "frame_notes_csv": args.frame_notes_csv,
            "micro_family_csv": args.micro_family_csv,
            "causal_roles_csv": args.causal_roles_csv,
            "passport_folder": args.passport_folder,
        },
        "outputs": {
            "frame_notes_csv": args.out_frame_notes_csv,
            "readable_csv": args.out_readable_csv,
            "meta_json": args.out_meta_json,
            "summary_txt": args.out_summary_txt,
        },
        "parameters": {
            "min_ownership_score": args.min_ownership_score,
            "max_notes_per_frame": args.max_notes_per_frame,
            "max_notes_low_mid": args.max_notes_low_mid,
        },
        "result": {
            "input_frame_rows": len(frame_rows),
            "output_frame_rows": len(out_rows),
            "kept_total": kept_total,
            "rejected_total": rejected_total,
            "frames": len(readable_rows),
            "frames_with_notes": frames_with_notes,
            "max_active_notes": max_kept,
            "active_distribution": active_distribution,
            "passports_loaded": len(passports),
            "reject_reasons": reject_reasons,
        },
    }

    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    txt = [
        "SINGLE NOTE PASSPORT OWNERSHIP FILTER",
        "=" * 72,
        f"frame_notes_csv   : {args.frame_notes_csv}",
        f"micro_family_csv  : {args.micro_family_csv}",
        f"causal_roles_csv  : {args.causal_roles_csv}",
        f"passport_folder   : {args.passport_folder}",
        "",
        f"input_frame_rows  : {len(frame_rows)}",
        f"output_frame_rows : {len(out_rows)}",
        f"kept_total        : {kept_total}",
        f"rejected_total    : {rejected_total}",
        f"frames            : {len(readable_rows)}",
        f"frames_with_notes : {frames_with_notes}",
        f"max_active_notes  : {max_kept}",
        f"passports_loaded  : {len(passports)}",
        "",
        "Active note distribution:",
    ]

    for k in sorted(active_distribution):
        txt.append(f"  {k}: {active_distribution[k]}")

    txt.append("")
    txt.append("Reject reasons:")
    for k in sorted(reject_reasons):
        txt.append(f"  {k}: {reject_reasons[k]}")

    txt.extend([
        "",
        "Principle:",
        "  Use single-note passport experience to decide whether",
        "  a polyphonic candidate is direct excitation or box/echo/sympathetic consequence.",
        "  Range-aware rules preserve low/high behavior learned from individual notes.",
        "",
    ])

    out_txt.write_text("\n".join(txt), encoding="utf-8")

    print("single note passport ownership filter complete")
    print(json.dumps(meta["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()