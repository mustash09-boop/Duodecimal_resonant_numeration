# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


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


def _degree(note: str) -> str:
    try:
        return str(note).split(".", 1)[1].split("'", 1)[0]
    except Exception:
        return ""


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Resolve globally scene-consistent polyphonic note interpretation."
    )

    ap.add_argument("--hypotheses_csv", required=True)
    ap.add_argument("--resolved_events_csv", required=True)

    ap.add_argument("--out_scene_events_csv", required=True)
    ap.add_argument("--out_frame_scene_csv", required=True)
    ap.add_argument("--out_readable_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--max_same_degree", type=int, default=2)
    ap.add_argument("--scene_penalty", type=float, default=0.16)
    ap.add_argument("--continuity_bonus", type=float, default=0.10)

    args = ap.parse_args()

    hypotheses = _load_csv(Path(args.hypotheses_csv))
    resolved = _load_csv(Path(args.resolved_events_csv))

    by_entity = defaultdict(list)

    for h in hypotheses:
        eid = str(h.get("entity_id", "")).strip()

        by_entity[eid].append({
            "candidate_note": str(h.get("candidate_note", "")).strip(),
            "score": _safe_float(h.get("hypothesis_probability"), 0.0),
            "ownership_role": str(h.get("ownership_role", "")).strip(),
        })

    resolved_events = []
    frame_rows = []
    readable_rows = []

    active_by_frame = defaultdict(list)

    entities_sorted = sorted(
        resolved,
        key=lambda r: (
            _safe_int(r.get("birth_frame"), 0),
            -_safe_float(r.get("resolution_confidence"), 0.0),
        )
    )

    status_counts = defaultdict(int)

    for r in entities_sorted:
        eid = str(r.get("entity_id", "")).strip()

        start = _safe_int(r.get("birth_frame"), 0)
        end = _safe_int(r.get("end_frame"), 0)

        candidates = by_entity.get(eid, [])

        if not candidates:
            status_counts["NO_CANDIDATES"] += 1
            continue

        best_note = ""
        best_scene_score = -999.0

        for cand in candidates:
            note = cand["candidate_note"]
            base = cand["score"]

            scene_penalty = 0.0
            continuity_bonus = 0.0
            decay_penalty = 0.0
            stagnation_penalty = 0.0
            competition_penalty = 0.0

            deg = _degree(note)

            overlapping = []

            max_frames = min(end + 1, start + 90)

            for frame in range(start, max_frames):
                overlapping.extend(active_by_frame.get(frame, []))

            same_degree_count = sum(
                1
                for x in overlapping
                if _degree(x["note"]) == deg
            )

            if same_degree_count > args.max_same_degree:
                scene_penalty += (
                    same_degree_count - args.max_same_degree
                ) * args.scene_penalty

            continuity_hits = sum(
                1
                for x in overlapping
                if x["note"] == note
            )

            continuity_bonus += (
                min(continuity_hits, 10)
                * args.continuity_bonus
                * 0.035
            )

            # ---------------------------------------------------------
            # identity decay
            # ---------------------------------------------------------

            entity_duration = max(end - start + 1, 1)

            if continuity_hits > 0:
                persistence_ratio = continuity_hits / entity_duration
            else:
                persistence_ratio = 0.0

            if persistence_ratio > 0.55:
                decay_penalty += (
                    persistence_ratio - 0.55
                ) * 0.45

            # ---------------------------------------------------------
            # stagnation penalty
            # ---------------------------------------------------------

            unique_neighbor_notes = len({
                x["note"]
                for x in overlapping
                if x["note"] != note
            })

            if unique_neighbor_notes <= 1 and continuity_hits > 18:
                stagnation_penalty += 0.18

            if continuity_hits > 36:
                stagnation_penalty += 0.25

            if continuity_hits > 60:
                stagnation_penalty += 0.40

            # ---------------------------------------------------------
            # competition penalty
            # ---------------------------------------------------------

            competing_same_degree = sum(
                1
                for x in overlapping
                if (
                    _degree(x["note"]) == deg
                    and x["note"] != note
                )
            )

            if competing_same_degree > 4:
                competition_penalty += (
                    competing_same_degree - 4
                ) * 0.05

            # ---------------------------------------------------------
            # final scene score
            # ---------------------------------------------------------

            scene_score = (
                base
                - scene_penalty
                - decay_penalty
                - stagnation_penalty
                - competition_penalty
                + continuity_bonus
            )

            if scene_score > best_scene_score:
                best_scene_score = scene_score
                best_note = note

        confidence = max(best_scene_score, 0.0)

        if confidence >= 0.62:
            status = "SCENE_RESOLVED"
        elif confidence >= 0.42:
            status = "SCENE_WEAK"
        else:
            status = "SCENE_REJECT"

        status_counts[status] += 1

        resolved_events.append({
            "entity_id": eid,
            "scene_note": best_note,
            "scene_confidence": f"{confidence:.9f}",
            "scene_status": status,
            "birth_frame": start,
            "end_frame": end,
            "duration_frames": end - start + 1,
            "ownership_role": r.get("ownership_role", ""),
            "alternatives": r.get("alternatives", ""),
        })

        if status != "SCENE_REJECT" and best_note:
            for frame in range(start, end + 1):
                active_by_frame[frame].append({
                    "entity_id": eid,
                    "note": best_note,
                    "confidence": confidence,
                })

                frame_rows.append({
                    "frame_index": frame,
                    "time_sec": f"{(frame / max(args.fps, 1e-9)):.9f}",
                    "entity_id": eid,
                    "note_token": best_note,
                    "scene_confidence": f"{confidence:.9f}",
                    "scene_status": status,
                })

    by_frame = defaultdict(list)

    for r in frame_rows:
        by_frame[_safe_int(r["frame_index"])].append(r)

    for frame in sorted(by_frame):
        items = sorted(
            by_frame[frame],
            key=lambda r: -_safe_float(r.get("scene_confidence"), 0.0),
        )

        readable_rows.append({
            "frame_index": frame,
            "time_sec": f"{(frame / max(args.fps, 1e-9)):.9f}",
            "active_note_count": len(items),
            "notes": " | ".join(
                f"{r['note_token']}:{_safe_float(r['scene_confidence']):.3f}[E{r['entity_id']}]"
                for r in items[:12]
            ),
        })

    active_distribution = defaultdict(int)

    for r in readable_rows:
        active_distribution[_safe_int(r["active_note_count"])] += 1

    out_events = Path(args.out_scene_events_csv)
    out_frames = Path(args.out_frame_scene_csv)
    out_readable = Path(args.out_readable_csv)
    out_meta = Path(args.out_meta_json)
    out_txt = Path(args.out_summary_txt)

    out_events.parent.mkdir(parents=True, exist_ok=True)

    event_fields = [
        "entity_id",
        "scene_note",
        "scene_confidence",
        "scene_status",
        "birth_frame",
        "end_frame",
        "duration_frames",
        "ownership_role",
        "alternatives",
    ]

    with out_events.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=event_fields)
        w.writeheader()
        w.writerows(resolved_events)

    frame_fields = [
        "frame_index",
        "time_sec",
        "entity_id",
        "note_token",
        "scene_confidence",
        "scene_status",
    ]

    with out_frames.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=frame_fields)
        w.writeheader()
        w.writerows(frame_rows)

    with out_readable.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["frame_index", "time_sec", "active_note_count", "notes"],
        )
        w.writeheader()
        w.writerows(readable_rows)

    meta = {
        "stage": "scene_consistent_note_resolution",
        "inputs": {
            "hypotheses_csv": args.hypotheses_csv,
            "resolved_events_csv": args.resolved_events_csv,
        },
        "outputs": {
            "scene_events_csv": args.out_scene_events_csv,
            "frame_scene_csv": args.out_frame_scene_csv,
            "readable_csv": args.out_readable_csv,
            "meta_json": args.out_meta_json,
            "summary_txt": args.out_summary_txt,
        },
        "parameters": {
            "fps": args.fps,
            "max_same_degree": args.max_same_degree,
            "scene_penalty": args.scene_penalty,
            "continuity_bonus": args.continuity_bonus,
        },
        "result": {
            "scene_events": len(resolved_events),
            "frame_rows": len(frame_rows),
            "readable_frames": len(readable_rows),
            "status_counts": dict(status_counts),
            "active_distribution": dict(active_distribution),
        },
    }

    out_meta.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    txt = [
        "SCENE CONSISTENT NOTE RESOLUTION",
        "=" * 72,
        f"hypotheses_csv      : {args.hypotheses_csv}",
        f"resolved_events_csv : {args.resolved_events_csv}",
        "",
        f"scene_events        : {len(resolved_events)}",
        f"frame_rows          : {len(frame_rows)}",
        f"readable_frames     : {len(readable_rows)}",
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
        "  Note identity is globally scene-consistent.",
        "  Competing hypotheses are stabilized by polyphonic context,",
        "  but stale identities decay if not refreshed by new evidence.",
        "",
    ])

    out_txt.write_text("\n".join(txt), encoding="utf-8")

    print("scene consistent note resolution complete")
    print(json.dumps(meta["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()