# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
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


def _tokens(raw: str) -> Set[str]:
    return {x.strip() for x in str(raw or "").split() if x.strip()}


def _normalize_note(s: str) -> str:
    s = str(s or "").strip()
    if not s:
        return ""
    if "'" in s:
        return s.split("'", 1)[0] + "'-"
    if s.endswith("-"):
        return s[:-1] + "'-"
    return s + "'-"


def _root_hint(row: Dict[str, Any]) -> str:
    return _normalize_note(row.get("root_hint_not_identity", ""))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Interpret likely notes from causal ownership entities."
    )

    ap.add_argument("--ecology_entities_csv", required=True)
    ap.add_argument("--ownership_roles_csv", required=True)

    ap.add_argument("--out_note_events_csv", required=True)
    ap.add_argument("--out_frame_notes_csv", required=True)
    ap.add_argument("--out_readable_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--min_owner_strength", type=float, default=0.18)
    ap.add_argument("--allowed_roles", default="PRIMARY_OWNER,RESONANCE_CARRIER")
    ap.add_argument("--fps", type=float, default=60.0)

    args = ap.parse_args()

    entities = _load_csv(Path(args.ecology_entities_csv))
    roles = _load_csv(Path(args.ownership_roles_csv))

    role_map = {
        str(r.get("entity_id", "")).strip(): r
        for r in roles
    }

    allowed_roles = {
        x.strip()
        for x in args.allowed_roles.split(",")
        if x.strip()
    }

    note_events = []
    frame_notes = []
    readable_rows = []

    status_counts = defaultdict(int)

    for e in entities:
        eid = str(e.get("ecology_entity_id", "")).strip()
        role = role_map.get(eid, {})

        ownership_role = str(role.get("ownership_role", "")).strip()
        owner_strength = _safe_float(role.get("ownership_strength"), 0.0)
        carrying_strength = _safe_float(role.get("carrying_strength"), 0.0)
        feeding_strength = _safe_float(role.get("feeding_strength"), 0.0)
        masking_strength = _safe_float(role.get("masking_strength"), 0.0)

        note = _root_hint(e)
        if not note:
            continue

        if ownership_role not in allowed_roles:
            status = "REJECT_ROLE"
        elif owner_strength < args.min_owner_strength and ownership_role == "PRIMARY_OWNER":
            status = "WEAK_OWNER"
        else:
            status = "ACCEPT"

        status_counts[status] += 1

        confidence = (
            owner_strength * 0.55
            + carrying_strength * 0.20
            + feeding_strength * 0.12
            - masking_strength * 0.18
        )

        confidence = max(confidence, 0.0)

        row = {
            "entity_id": eid,
            "interpreted_note": note,
            "birth_frame": e.get("birth_frame", ""),
            "end_frame": e.get("end_frame", ""),
            "duration_frames": e.get("duration_frames", ""),
            "ownership_role": ownership_role,
            "note_confidence": f"{confidence:.9f}",
            "ownership_strength": f"{owner_strength:.9f}",
            "carrying_strength": f"{carrying_strength:.9f}",
            "feeding_strength": f"{feeding_strength:.9f}",
            "masking_strength": f"{masking_strength:.9f}",
            "status": status,
            "token_union_count": e.get("token_union_count", ""),
            "topology_signature_count": e.get("topology_signature_count", ""),
        }

        note_events.append(row)

        if status == "ACCEPT":
            start = _safe_int(e.get("birth_frame"), 0)
            end = _safe_int(e.get("end_frame"), 0)

            for frame in range(start, end + 1):
                frame_notes.append({
                    "frame_index": frame,
                    "time_sec": f"{(frame / max(args.fps, 1e-9)):.9f}",
                    "entity_id": eid,
                    "note_token": note,
                    "note_confidence": f"{confidence:.9f}",
                    "ownership_role": ownership_role,
                })

    by_frame: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for r in frame_notes:
        by_frame[_safe_int(r["frame_index"])].append(r)

    for frame in sorted(by_frame):
        items = sorted(
            by_frame[frame],
            key=lambda r: -_safe_float(r.get("note_confidence"), 0.0),
        )

        readable_rows.append({
            "frame_index": frame,
            "time_sec": f"{(frame / max(args.fps, 1e-9)):.9f}",
            "active_note_count": len(items),
            "notes": " | ".join(
                f"{r['note_token']}:{_safe_float(r['note_confidence']):.3f}[E{r['entity_id']}/{r['ownership_role']}]"
                for r in items[:12]
            ),
        })

    out_events = Path(args.out_note_events_csv)
    out_frames = Path(args.out_frame_notes_csv)
    out_readable = Path(args.out_readable_csv)
    out_meta = Path(args.out_meta_json)
    out_txt = Path(args.out_summary_txt)

    out_events.parent.mkdir(parents=True, exist_ok=True)

    event_fields = [
        "entity_id",
        "interpreted_note",
        "birth_frame",
        "end_frame",
        "duration_frames",
        "ownership_role",
        "note_confidence",
        "ownership_strength",
        "carrying_strength",
        "feeding_strength",
        "masking_strength",
        "status",
        "token_union_count",
        "topology_signature_count",
    ]

    with out_events.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=event_fields)
        w.writeheader()
        w.writerows(note_events)

    frame_fields = [
        "frame_index",
        "time_sec",
        "entity_id",
        "note_token",
        "note_confidence",
        "ownership_role",
    ]

    with out_frames.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=frame_fields)
        w.writeheader()
        w.writerows(frame_notes)

    with out_readable.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["frame_index", "time_sec", "active_note_count", "notes"],
        )
        w.writeheader()
        w.writerows(readable_rows)

    active_distribution = defaultdict(int)
    for r in readable_rows:
        active_distribution[_safe_int(r["active_note_count"])] += 1

    meta = {
        "stage": "ownership_note_interpreter",
        "inputs": {
            "ecology_entities_csv": args.ecology_entities_csv,
            "ownership_roles_csv": args.ownership_roles_csv,
        },
        "outputs": {
            "note_events_csv": args.out_note_events_csv,
            "frame_notes_csv": args.out_frame_notes_csv,
            "readable_csv": args.out_readable_csv,
            "meta_json": args.out_meta_json,
            "summary_txt": args.out_summary_txt,
        },
        "parameters": {
            "min_owner_strength": args.min_owner_strength,
            "allowed_roles": sorted(allowed_roles),
            "fps": args.fps,
        },
        "result": {
            "input_entities": len(entities),
            "note_events": len(note_events),
            "frame_note_rows": len(frame_notes),
            "readable_frames": len(readable_rows),
            "status_counts": dict(status_counts),
            "active_distribution": dict(active_distribution),
        },
    }

    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    txt = [
        "OWNERSHIP NOTE INTERPRETER",
        "=" * 72,
        f"ecology_entities_csv : {args.ecology_entities_csv}",
        f"ownership_roles_csv  : {args.ownership_roles_csv}",
        "",
        f"input_entities       : {len(entities)}",
        f"note_events          : {len(note_events)}",
        f"frame_note_rows      : {len(frame_notes)}",
        f"readable_frames      : {len(readable_rows)}",
        "",
        "Status counts:",
    ]

    for k in sorted(status_counts):
        txt.append(f"  {k}: {status_counts[k]}")

    txt.append("")
    txt.append("Active note distribution:")
    for k in sorted(active_distribution):
        txt.append(f"  {k}: {active_distribution[k]}")

    txt.extend([
        "",
        "Principle:",
        "  Notes are interpreted from causal ownership entities,",
        "  not directly from spectral peaks or premature note tokens.",
        "",
    ])

    out_txt.write_text("\n".join(txt), encoding="utf-8")

    print("ownership note interpreter complete")
    print(json.dumps(meta["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()