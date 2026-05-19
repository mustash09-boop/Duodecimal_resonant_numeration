# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
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


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _degree_symbol(note: str) -> str:
    try:
        return note.split(".", 1)[1].split("'", 1)[0]
    except Exception:
        return ""


def _octave_part(note: str) -> str:
    try:
        return note.split(".", 1)[0]
    except Exception:
        return ""


def _octave_value(note: str) -> int:
    raw = _octave_part(note)

    if not raw:
        return 0

    value = 0

    for ch in raw:
        if ch not in ALPHABET12:
            continue
        value = value * 12 + (ALPHABET12.index(ch) + 1)

    return value


def _range_mode(note: str) -> str:
    ov = _octave_value(note)

    if ov <= 6:
        return "low"

    if ov >= 10:
        return "high"

    return "mid"


def _family_members(raw: str) -> List[str]:
    return [x.strip() for x in str(raw or "").split() if x.strip()]


def _group_families(rows: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}

    for r in rows:
        frame = _safe_int(r.get("frame_index"), 0)
        out.setdefault(frame, []).append(r)

    return out


def _find_matching_family(
    note_token: str,
    frame_start: int,
    frame_end: int,
    grouped_families: Dict[int, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:

    degree = _degree_symbol(note_token)

    matches = []

    for frame in range(frame_start, frame_end + 1):
        rows = grouped_families.get(frame, [])

        for r in rows:
            root = str(r.get("family_root_note", "")).strip()

            if _degree_symbol(root) != degree:
                continue

            matches.append(r)

    return matches


def _resolve_anchor(
    note_token: str,
    family_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:

    range_mode = _range_mode(note_token)

    octave_votes: Dict[str, float] = {}

    odd_support = 0.0
    even_support = 0.0

    for r in family_rows:
        members = _family_members(r.get("family_members", ""))

        for m in members:
            octave = _octave_part(m)

            octave_votes.setdefault(octave, 0.0)

            score = _safe_float(r.get("family_score"), 0.0)

            octave_votes[octave] += score

            try:
                mo = _octave_value(m)
                no = _octave_value(note_token)

                diff = abs(mo - no)

                if diff % 2 == 1:
                    odd_support += score
                else:
                    even_support += score

            except Exception:
                pass

    if not octave_votes:
        return {
            "resolved_note": note_token,
            "resolved_octave": _octave_part(note_token),
            "range_mode": range_mode,
            "odd_support": 0.0,
            "even_support": 0.0,
            "octave_confidence": 0.0,
        }

    # RANGE-AWARE RESOLUTION

    if range_mode == "low":
        # low register:
        # prioritize octave with strongest odd harmonic support
        best_octave = max(octave_votes.items(), key=lambda x: x[1])[0]

    elif range_mode == "high":
        # high register:
        # accept sparse harmonic topology
        best_octave = min(octave_votes.items(), key=lambda x: x[0])[0]

    else:
        # mid register:
        # strongest consensus
        best_octave = max(octave_votes.items(), key=lambda x: x[1])[0]

    resolved = f"{best_octave}.{_degree_symbol(note_token)}'-"

    total = odd_support + even_support

    if total <= 0:
        confidence = 0.0
    else:
        confidence = odd_support / total

    return {
        "resolved_note": resolved,
        "resolved_octave": best_octave,
        "range_mode": range_mode,
        "odd_support": odd_support,
        "even_support": even_support,
        "octave_confidence": confidence,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Resolve octave anchor ambiguity using range-aware harmonic topology."
    )

    ap.add_argument("--persistent_events_csv", required=True)
    ap.add_argument("--root_family_csv", required=True)

    ap.add_argument("--out_resolved_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    args = ap.parse_args()

    persistent_csv = Path(args.persistent_events_csv)
    family_csv = Path(args.root_family_csv)

    out_csv = Path(args.out_resolved_csv)
    out_meta = Path(args.out_meta_json)
    out_txt = Path(args.out_summary_txt)

    events = _load_csv(persistent_csv)
    families = _load_csv(family_csv)

    grouped_families = _group_families(families)

    resolved_rows = []

    changed = 0

    for ev in events:
        note = str(ev.get("note_token", "")).strip()

        start_frame = _safe_int(ev.get("start_frame"), 0)
        end_frame = _safe_int(ev.get("end_frame"), 0)

        matching_families = _find_matching_family(
            note,
            start_frame,
            end_frame,
            grouped_families,
        )

        resolved = _resolve_anchor(note, matching_families)

        if resolved["resolved_note"] != note:
            changed += 1

        row = dict(ev)

        row["original_note"] = note
        row["resolved_note"] = resolved["resolved_note"]
        row["resolved_octave"] = resolved["resolved_octave"]
        row["range_mode"] = resolved["range_mode"]

        row["odd_support"] = f"{resolved['odd_support']:.9f}"
        row["even_support"] = f"{resolved['even_support']:.9f}"
        row["octave_confidence"] = f"{resolved['octave_confidence']:.9f}"

        resolved_rows.append(row)

    fields = list(resolved_rows[0].keys()) if resolved_rows else []

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(resolved_rows)

    meta = {
        "stage": "octave_anchor_resolver",
        "inputs": {
            "persistent_events_csv": str(persistent_csv),
            "root_family_csv": str(family_csv),
        },
        "outputs": {
            "resolved_csv": str(out_csv),
            "meta_json": str(out_meta),
            "summary_txt": str(out_txt),
        },
        "result": {
            "input_events": len(events),
            "resolved_events": len(resolved_rows),
            "changed_octave_assignments": changed,
        },
    }

    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    txt = []
    txt.append("OCTAVE ANCHOR RESOLVER")
    txt.append("=" * 72)
    txt.append(f"persistent_events_csv      : {persistent_csv}")
    txt.append(f"root_family_csv            : {family_csv}")
    txt.append("")
    txt.append(f"input_events               : {len(events)}")
    txt.append(f"resolved_events            : {len(resolved_rows)}")
    txt.append(f"changed_octave_assignments : {changed}")
    txt.append("")
    txt.append("Principle:")
    txt.append("  Resolve octave ambiguity using range-aware harmonic topology.")
    txt.append("  Low register prioritizes odd harmonic support.")
    txt.append("  High register tolerates sparse harmonic ladders.")
    txt.append("")

    out_txt.write_text("\n".join(txt), encoding="utf-8")

    print("octave anchor resolver complete")
    print(json.dumps(meta["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()