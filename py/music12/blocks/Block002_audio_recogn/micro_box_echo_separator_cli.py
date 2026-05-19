# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Set


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _normalize_token(raw: str) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    if "'" in s:
        return s
    if s.endswith("-"):
        return s[:-1] + "'-"
    return s + "'-"


def _note_from_profile_filename(path: Path) -> str:
    left = path.name.split("__note_box_profile", 1)[0]
    raw = left.split("_")[-1]
    return _normalize_token(raw)


def _radius(row: Dict[str, Any]) -> float:
    x = _safe_float(row.get("mean_x12"), 0.0)
    y = _safe_float(row.get("mean_y12"), 0.0)
    return math.sqrt(x * x + y * y)


def _load_profiles(folder: Path) -> Dict[str, Dict[str, Set[str]]]:
    profiles: Dict[str, Dict[str, Set[str]]] = {}

    for p in sorted(folder.glob("*__note_box_profile.csv")):
        note = _note_from_profile_filename(p)
        rows = _load_csv(p)

        note_set: Set[str] = set()
        box_set: Set[str] = set()
        echo_set: Set[str] = set()

        for r in rows:
            token = _normalize_token(r.get("token", ""))
            if not token:
                continue

            amp = _safe_float(r.get("mean_amp"), 0.0)
            presence = _safe_float(r.get("presence_ratio"), 0.0)
            rad = _radius(r)

            if amp >= 0.16 and presence >= 0.06:
                note_set.add(token)
            elif presence >= 0.16 and amp >= 0.035:
                box_set.add(token)
            elif presence >= 0.10 and amp < 0.05:
                echo_set.add(token)
            elif presence >= 0.08 and rad > 0.35:
                echo_set.add(token)

        profiles[note] = {
            "note": note_set,
            "box": box_set,
            "echo": echo_set,
        }

    return profiles


def _degree(token: str) -> str:
    try:
        return token.split(".", 1)[1].split("'", 1)[0]
    except Exception:
        return ""


def _find_profile(root: str, profiles: Dict[str, Dict[str, Set[str]]]) -> Dict[str, Set[str]]:
    if root in profiles:
        return profiles[root]

    d = _degree(root)
    for note, prof in profiles.items():
        if _degree(note) == d:
            return prof

    return {"note": set(), "box": set(), "echo": set()}


def _members(raw: str) -> List[str]:
    return [x.strip() for x in str(raw or "").split() if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Separate note/box/echo components from micro harmonic families."
    )

    ap.add_argument("--micro_family_csv", required=True)
    ap.add_argument("--box_profile_folder", required=True)

    ap.add_argument("--out_separated_csv", required=True)
    ap.add_argument("--out_frame_summary_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    args = ap.parse_args()

    family_csv = Path(args.micro_family_csv)
    profile_folder = Path(args.box_profile_folder)

    out_csv = Path(args.out_separated_csv)
    out_frame = Path(args.out_frame_summary_csv)
    out_meta = Path(args.out_meta_json)
    out_txt = Path(args.out_summary_txt)

    families = _load_csv(family_csv)
    profiles = _load_profiles(profile_folder)

    out_rows = []

    note_count = 0
    box_count = 0
    echo_count = 0
    unknown_count = 0

    for r in families:
        root = str(r.get("family_root_note", "")).strip()
        prof = _find_profile(root, profiles)

        members = _members(r.get("family_members", ""))

        note_members = []
        box_members = []
        echo_members = []
        unknown_members = []

        for m in members:
            if m in prof["note"]:
                note_members.append(m)
                note_count += 1
            elif m in prof["box"]:
                box_members.append(m)
                box_count += 1
            elif m in prof["echo"]:
                echo_members.append(m)
                echo_count += 1
            else:
                unknown_members.append(m)
                unknown_count += 1

        base_score = _safe_float(r.get("family_score"), 0.0)

        clean_score = base_score
        clean_score += len(note_members) * 0.12
        clean_score -= len(box_members) * 0.05
        clean_score -= len(echo_members) * 0.03
        clean_score += len(unknown_members) * 0.01
        clean_score = max(clean_score, 0.0)

        row = dict(r)
        row["clean_note_score"] = f"{clean_score:.9f}"
        row["note_members"] = " ".join(sorted(set(note_members)))
        row["box_members"] = " ".join(sorted(set(box_members)))
        row["echo_members"] = " ".join(sorted(set(echo_members)))
        row["unknown_members"] = " ".join(sorted(set(unknown_members)))
        row["note_member_count"] = len(note_members)
        row["box_member_count"] = len(box_members)
        row["echo_member_count"] = len(echo_members)
        row["unknown_member_count"] = len(unknown_members)

        out_rows.append(row)

    out_rows.sort(
        key=lambda x: (
            int(x.get("frame_index", 0)),
            -_safe_float(x.get("clean_note_score"), 0.0),
        )
    )

    frame_map: Dict[int, List[str]] = {}

    for r in out_rows:
        frame = int(r.get("frame_index", 0))
        frame_map.setdefault(frame, []).append(
            f"{r.get('family_root_note')}:{r.get('clean_note_score')}"
            f"[N{r.get('note_member_count')}/B{r.get('box_member_count')}/E{r.get('echo_member_count')}]"
        )

    frame_rows = [
        {
            "frame_index": frame,
            "top_cleaned_families": " | ".join(items[:12]),
        }
        for frame, items in sorted(frame_map.items())
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = list(out_rows[0].keys()) if out_rows else []

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    with out_frame.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame_index", "top_cleaned_families"])
        w.writeheader()
        w.writerows(frame_rows)

    meta = {
        "stage": "micro_box_echo_separator",
        "inputs": {
            "micro_family_csv": str(family_csv),
            "box_profile_folder": str(profile_folder),
        },
        "outputs": {
            "separated_csv": str(out_csv),
            "frame_summary_csv": str(out_frame),
            "meta_json": str(out_meta),
            "summary_txt": str(out_txt),
        },
        "result": {
            "profiles_loaded": len(profiles),
            "family_rows": len(out_rows),
            "note_component_count": note_count,
            "box_component_count": box_count,
            "echo_component_count": echo_count,
            "unknown_component_count": unknown_count,
        },
    }

    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    txt = [
        "MICRO BOX / ECHO SEPARATOR",
        "=" * 72,
        f"micro_family_csv       : {family_csv}",
        f"box_profile_folder     : {profile_folder}",
        "",
        f"profiles_loaded        : {len(profiles)}",
        f"family_rows            : {len(out_rows)}",
        "",
        f"note_component_count   : {note_count}",
        f"box_component_count    : {box_count}",
        f"echo_component_count   : {echo_count}",
        f"unknown_component_count: {unknown_count}",
        "",
        "Principle:",
        "  Separate note, instrument-box and box-echo components",
        "  using micro-aware harmonic families and Block004 note_box profiles.",
        "",
    ]

    out_txt.write_text("\n".join(txt), encoding="utf-8")

    print("micro box / echo separator complete")
    print(json.dumps(meta["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()