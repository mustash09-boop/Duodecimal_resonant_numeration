# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


DIGITS12 = ["1","2","3","4","5","6","7","8","9","A","B","C"]


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


def _octave(note: str) -> int:
    try:
        return int(str(note).split(".", 1)[0])
    except Exception:
        return 0


def _degree_index(note: str) -> int:
    d = _degree(note)
    try:
        return DIGITS12.index(d)
    except Exception:
        return -1


def _interval_distance(a: str, b: str) -> int:
    ia = _degree_index(a)
    ib = _degree_index(b)

    if ia < 0 or ib < 0:
        return 99

    d = abs(ia - ib)
    return min(d, 12 - d)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Resolve musically coherent scene expectations."
    )

    ap.add_argument("--scene_events_csv", required=True)
    ap.add_argument("--frame_scene_csv", required=True)

    ap.add_argument("--out_musical_events_csv", required=True)
    ap.add_argument("--out_frame_music_csv", required=True)
    ap.add_argument("--out_readable_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--interval_bonus", type=float, default=0.10)
    ap.add_argument("--continuity_bonus", type=float, default=0.12)
    ap.add_argument("--density_penalty", type=float, default=0.08)

    args = ap.parse_args()

    scene_events = _load_csv(Path(args.scene_events_csv))
    frame_scene = _load_csv(Path(args.frame_scene_csv))

    active_by_frame = defaultdict(list)

    for r in frame_scene:
        frame = _safe_int(r.get("frame_index"), 0)

        active_by_frame[frame].append({
            "entity_id": str(r.get("entity_id", "")).strip(),
            "note": str(r.get("note_token", "")).strip(),
            "confidence": _safe_float(r.get("scene_confidence"), 0.0),
        })

    musical_rows = []
    frame_rows = []
    readable_rows = []

    status_counts = defaultdict(int)

    for ev in scene_events:
        eid = str(ev.get("entity_id", "")).strip()

        note = str(ev.get("scene_note", "")).strip()
        confidence = _safe_float(ev.get("scene_confidence"), 0.0)

        start = _safe_int(ev.get("birth_frame"), 0)
        end = _safe_int(ev.get("end_frame"), 0)

        musical_bonus = 0.0
        density_penalty = 0.0

        observed_neighbors = []

        for frame in range(start, min(end + 1, start + 120)):
            observed_neighbors.extend(active_by_frame.get(frame, []))

        for nb in observed_neighbors:
            nb_note = nb["note"]

            if nb_note == note:
                musical_bonus += args.continuity_bonus * 0.05
                continue

            dist = _interval_distance(note, nb_note)

            if dist in {3, 4, 5, 7}:
                musical_bonus += args.interval_bonus * 0.10

            elif dist in {1, 2, 6}:
                density_penalty += args.density_penalty * 0.05

        polyphony_density = len(observed_neighbors) / 20.0

        if polyphony_density > 8:
            density_penalty += (
                polyphony_density - 8
            ) * args.density_penalty * 0.04

        musical_confidence = (
            confidence
            + musical_bonus
            - density_penalty
        )

        musical_confidence = max(musical_confidence, 0.0)

        if musical_confidence >= 0.58:
            status = "MUSICALLY_STABLE"

        elif musical_confidence >= 0.34:
            status = "MUSICALLY_PLAUSIBLE"

        else:
            status = "MUSICALLY_UNSTABLE"

        status_counts[status] += 1

        musical_rows.append({
            "entity_id": eid,
            "musical_note": note,
            "musical_confidence": f"{musical_confidence:.9f}",
            "musical_status": status,
            "scene_confidence": f"{confidence:.9f}",
            "birth_frame": start,
            "end_frame": end,
            "duration_frames": end - start + 1,
            "ownership_role": ev.get("ownership_role", ""),
            "alternatives": ev.get("alternatives", ""),
        })

        if status != "MUSICALLY_UNSTABLE":
            for frame in range(start, end + 1):
                frame_rows.append({
                    "frame_index": frame,
                    "time_sec": f"{(frame / max(args.fps, 1e-9)):.9f}",
                    "entity_id": eid,
                    "note_token": note,
                    "musical_confidence": f"{musical_confidence:.9f}",
                    "musical_status": status,
                })

    by_frame = defaultdict(list)

    for r in frame_rows:
        by_frame[_safe_int(r["frame_index"])].append(r)

    for frame in sorted(by_frame):
        items = sorted(
            by_frame[frame],
            key=lambda x: -_safe_float(x.get("musical_confidence"), 0.0),
        )

        readable_rows.append({
            "frame_index": frame,
            "time_sec": f"{(frame / max(args.fps, 1e-9)):.9f}",
            "active_note_count": len(items),
            "notes": " | ".join(
                f"{r['note_token']}:{_safe_float(r['musical_confidence']):.3f}[E{r['entity_id']}]"
                for r in items[:12]
            ),
        })

    active_distribution = defaultdict(int)

    for r in readable_rows:
        active_distribution[_safe_int(r["active_note_count"])] += 1

    out_events = Path(args.out_musical_events_csv)
    out_frames = Path(args.out_frame_music_csv)
    out_readable = Path(args.out_readable_csv)
    out_meta = Path(args.out_meta_json)
    out_txt = Path(args.out_summary_txt)

    out_events.parent.mkdir(parents=True, exist_ok=True)

    event_fields = [
        "entity_id",
        "musical_note",
        "musical_confidence",
        "musical_status",
        "scene_confidence",
        "birth_frame",
        "end_frame",
        "duration_frames",
        "ownership_role",
        "alternatives",
    ]

    with out_events.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=event_fields)
        w.writeheader()
        w.writerows(musical_rows)

    frame_fields = [
        "frame_index",
        "time_sec",
        "entity_id",
        "note_token",
        "musical_confidence",
        "musical_status",
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
        "stage": "musical_scene_expectation_resolver",
        "inputs": {
            "scene_events_csv": args.scene_events_csv,
            "frame_scene_csv": args.frame_scene_csv,
        },
        "outputs": {
            "musical_events_csv": args.out_musical_events_csv,
            "frame_music_csv": args.out_frame_music_csv,
            "readable_csv": args.out_readable_csv,
            "meta_json": args.out_meta_json,
            "summary_txt": args.out_summary_txt,
        },
        "parameters": {
            "fps": args.fps,
            "interval_bonus": args.interval_bonus,
            "continuity_bonus": args.continuity_bonus,
            "density_penalty": args.density_penalty,
        },
        "result": {
            "musical_events": len(musical_rows),
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
        "MUSICAL SCENE EXPECTATION RESOLVER",
        "=" * 72,
        f"scene_events_csv : {args.scene_events_csv}",
        f"frame_scene_csv  : {args.frame_scene_csv}",
        "",
        f"musical_events   : {len(musical_rows)}",
        f"frame_rows       : {len(frame_rows)}",
        f"readable_frames  : {len(readable_rows)}",
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
        "  Musical interpretation is stabilized by interval coherence,",
        "  continuity and ecological plausibility.",
        "",
    ])

    out_txt.write_text("\n".join(txt), encoding="utf-8")

    print("musical scene expectation resolver complete")
    print(json.dumps(meta["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()