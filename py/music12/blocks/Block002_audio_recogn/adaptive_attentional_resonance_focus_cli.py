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


def _choose_adaptive_foreground(
    scored: List[Dict[str, Any]],
    *,
    min_salience: float,
    elbow_gap: float,
    soft_max_guard: int,
) -> List[Dict[str, Any]]:

    if not scored:
        return []

    scored = [
        x
        for x in scored
        if _safe_float(x.get("salience"), 0.0) >= min_salience
    ]

    if not scored:
        return []

    scored.sort(
        key=lambda x: (
            -_safe_float(x["salience"]),
            -_safe_float(x["confidence"]),
            x["note"],
        )
    )

    if len(scored) == 1:
        return scored

    cut = len(scored)

    for i in range(len(scored) - 1):
        a = _safe_float(scored[i]["salience"])
        b = _safe_float(scored[i + 1]["salience"])

        gap = a - b

        if gap >= elbow_gap:
            cut = i + 1
            break

    return scored[: min(cut, soft_max_guard)]


def main() -> None:

    ap = argparse.ArgumentParser(
        description=(
            "Adaptive attentional resonance focus. "
            "Foreground count is inferred from salience elbow, not forced."
        )
    )

    ap.add_argument("--musical_events_csv", required=True)
    ap.add_argument("--frame_music_csv", required=True)

    ap.add_argument("--out_focus_events_csv", required=True)
    ap.add_argument("--out_focus_frames_csv", required=True)
    ap.add_argument("--out_readable_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--fps", type=float, default=60.0)

    # Не лимит произведения, а аварийный предохранитель.
    ap.add_argument("--soft_max_guard", type=int, default=12)

    ap.add_argument("--min_salience", type=float, default=0.34)
    ap.add_argument("--elbow_gap", type=float, default=0.18)

    ap.add_argument("--continuity_bonus", type=float, default=0.12)
    ap.add_argument("--stability_bonus", type=float, default=0.18)

    ap.add_argument("--masking_penalty", type=float, default=0.14)
    ap.add_argument("--duplicate_note_penalty", type=float, default=0.18)

    args = ap.parse_args()

    musical_events = _load_csv(Path(args.musical_events_csv))
    frame_music = _load_csv(Path(args.frame_music_csv))

    event_map = {
        str(r.get("entity_id", "")).strip(): r
        for r in musical_events
    }

    by_frame: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for r in frame_music:

        frame = _safe_int(r.get("frame_index"), 0)

        by_frame[frame].append(
            {
                "entity_id": str(r.get("entity_id", "")).strip(),
                "note": str(r.get("note_token", "")).strip(),
                "confidence": _safe_float(
                    r.get("musical_confidence"),
                    0.0,
                ),
                "status": str(
                    r.get("musical_status", "")
                ).strip(),
            }
        )

    focus_rows = []
    frame_rows = []
    readable_rows = []

    previous_focus: Dict[str, float] = {}

    focus_status_counts = defaultdict(int)
    active_distribution = defaultdict(int)

    for frame in sorted(by_frame):

        items = by_frame[frame]

        scored = []

        for item in items:

            eid = item["entity_id"]

            ev = event_map.get(eid, {})

            conf = item["confidence"]

            duration = _safe_int(
                ev.get("duration_frames"),
                0,
            )

            salience = conf

            salience += (
                min(duration / 120.0, 1.0)
                * args.stability_bonus
            )

            if eid in previous_focus:
                salience += args.continuity_bonus

            if item["status"] == "MUSICALLY_STABLE":
                salience += 0.10

            elif item["status"] == "MUSICALLY_PLAUSIBLE":
                salience += 0.03

            else:
                salience -= 0.20

            same_note_count = sum(
                1
                for x in items
                if x["note"] == item["note"]
            )

            if same_note_count > 1:
                salience -= (
                    (same_note_count - 1)
                    * args.duplicate_note_penalty
                )

            local_density = len(items)

            if (
                local_density > 8
                and item["status"] != "MUSICALLY_STABLE"
            ):
                salience -= (
                    (local_density - 8)
                    * args.masking_penalty
                    * 0.05
                )

            salience = max(salience, 0.0)

            scored.append(
                {
                    "entity_id": eid,
                    "note": item["note"],
                    "salience": salience,
                    "confidence": conf,
                    "status": item["status"],
                }
            )

        focused = _choose_adaptive_foreground(
            scored,
            min_salience=args.min_salience,
            elbow_gap=args.elbow_gap,
            soft_max_guard=args.soft_max_guard,
        )

        new_focus = {}

        for rank, item in enumerate(focused, start=1):

            eid = item["entity_id"]

            new_focus[eid] = item["salience"]

            focus_rows.append(
                {
                    "frame_index": frame,
                    "entity_id": eid,
                    "note_token": item["note"],
                    "attentional_salience": (
                        f"{item['salience']:.9f}"
                    ),
                    "focus_rank": rank,
                    "musical_confidence": (
                        f"{item['confidence']:.9f}"
                    ),
                    "musical_status": item["status"],
                }
            )

            frame_rows.append(
                {
                    "frame_index": frame,
                    "time_sec": (
                        f"{(frame / max(args.fps, 1e-9)):.9f}"
                    ),
                    "focus_rank": rank,
                    "entity_id": eid,
                    "note_token": item["note"],
                    "attentional_salience": (
                        f"{item['salience']:.9f}"
                    ),
                }
            )

            focus_status_counts[item["status"]] += 1

        previous_focus = new_focus

        active_distribution[len(focused)] += 1

        readable_rows.append(
            {
                "frame_index": frame,
                "time_sec": (
                    f"{(frame / max(args.fps, 1e-9)):.9f}"
                ),
                "foreground_note_count": len(focused),
                "foreground_notes": " | ".join(
                    (
                        f"{x['note']}:"
                        f"{x['salience']:.3f}"
                        f"[E{x['entity_id']}]"
                    )
                    for x in focused
                ),
            }
        )

    out_events = Path(args.out_focus_events_csv)
    out_frames = Path(args.out_focus_frames_csv)
    out_readable = Path(args.out_readable_csv)
    out_meta = Path(args.out_meta_json)
    out_txt = Path(args.out_summary_txt)

    out_events.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    focus_fields = [
        "frame_index",
        "entity_id",
        "note_token",
        "attentional_salience",
        "focus_rank",
        "musical_confidence",
        "musical_status",
    ]

    with out_events.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=focus_fields,
        )

        w.writeheader()
        w.writerows(focus_rows)

    frame_fields = [
        "frame_index",
        "time_sec",
        "focus_rank",
        "entity_id",
        "note_token",
        "attentional_salience",
    ]

    with out_frames.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=frame_fields,
        )

        w.writeheader()
        w.writerows(frame_rows)

    with out_readable.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=[
                "frame_index",
                "time_sec",
                "foreground_note_count",
                "foreground_notes",
            ],
        )

        w.writeheader()
        w.writerows(readable_rows)

    meta = {
        "stage": "adaptive_attentional_resonance_focus",
        "inputs": {
            "musical_events_csv": (
                args.musical_events_csv
            ),
            "frame_music_csv": (
                args.frame_music_csv
            ),
        },
        "outputs": {
            "focus_events_csv": (
                args.out_focus_events_csv
            ),
            "focus_frames_csv": (
                args.out_focus_frames_csv
            ),
            "readable_csv": (
                args.out_readable_csv
            ),
            "meta_json": args.out_meta_json,
            "summary_txt": args.out_summary_txt,
        },
        "parameters": {
            "soft_max_guard": (
                args.soft_max_guard
            ),
            "min_salience": (
                args.min_salience
            ),
            "elbow_gap": args.elbow_gap,
            "continuity_bonus": (
                args.continuity_bonus
            ),
            "stability_bonus": (
                args.stability_bonus
            ),
            "masking_penalty": (
                args.masking_penalty
            ),
            "duplicate_note_penalty": (
                args.duplicate_note_penalty
            ),
        },
        "result": {
            "focus_rows": len(focus_rows),
            "frame_rows": len(frame_rows),
            "readable_frames": len(readable_rows),
            "focus_status_counts": dict(
                focus_status_counts
            ),
            "foreground_distribution": dict(
                active_distribution
            ),
        },
    }

    out_meta.write_text(
        json.dumps(
            meta,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    txt = [
        "ADAPTIVE ATTENTIONAL RESONANCE FOCUS",
        "=" * 72,
        f"musical_events_csv : {args.musical_events_csv}",
        f"frame_music_csv    : {args.frame_music_csv}",
        "",
        f"focus_rows         : {len(focus_rows)}",
        f"frame_rows         : {len(frame_rows)}",
        f"readable_frames    : {len(readable_rows)}",
        "",
        "Focus status counts:",
    ]

    for k in sorted(focus_status_counts):
        txt.append(
            f"  {k}: {focus_status_counts[k]}"
        )

    txt.append("")
    txt.append("Foreground note distribution:")

    for k in sorted(active_distribution):
        txt.append(
            f"  {k}: {active_distribution[k]}"
        )

    txt.extend(
        [
            "",
            "Principle:",
            (
                "  Foreground size is inferred "
                "from salience structure."
            ),
            (
                "  The system does not force "
                "a fixed polyphony count."
            ),
            (
                "  soft_max_guard is only "
                "a safety guard against "
                "runaway resonance flooding."
            ),
            "",
        ]
    )

    out_txt.write_text(
        "\n".join(txt),
        encoding="utf-8",
    )

    print(
        "adaptive attentional resonance "
        "focus complete"
    )

    print(
        json.dumps(
            meta["result"],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()