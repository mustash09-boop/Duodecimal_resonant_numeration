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


def _parse_note_path(raw: str) -> List[str]:
    return [x.strip() for x in str(raw or "").split() if x.strip()]


def _voice_signature(row: Dict[str, Any]) -> Dict[str, Any]:
    notes = _parse_note_path(row.get("note_path", ""))

    if notes:
        start_note = notes[0]
        end_note = notes[-1]
    else:
        start_note = str(row.get("start_note", "")).strip()
        end_note = str(row.get("end_note", "")).strip()

    return {
        "voice_id": _safe_int(row.get("voice_id"), 0),
        "start_frame": _safe_int(row.get("start_frame"), 0),
        "end_frame": _safe_int(row.get("end_frame"), 0),
        "length_frames": _safe_int(row.get("length_frames"), 0),
        "mean_score": _safe_float(row.get("mean_score"), 0.0),
        "start_note": start_note,
        "end_note": end_note,
        "notes": notes,
    }


def _can_merge(a: Dict[str, Any], b: Dict[str, Any], max_gap: int, max_pitch_jump: float) -> bool:
    gap = b["start_frame"] - a["end_frame"]

    if gap < 0 or gap > max_gap:
        return False

    dist = _pitch_distance(a["end_note"], b["start_note"])

    return dist <= max_pitch_jump


def _merge_voice_group(group: List[Dict[str, Any]], stable_id: int) -> Dict[str, Any]:
    group = sorted(group, key=lambda v: (v["start_frame"], v["voice_id"]))

    all_notes: List[str] = []
    source_voice_ids = []

    total_score = 0.0
    total_len = 0

    for v in group:
        source_voice_ids.append(str(v["voice_id"]))

        notes = v["notes"] or [v["start_note"], v["end_note"]]

        all_notes.extend(notes)

        total_score += v["mean_score"] * max(v["length_frames"], 1)
        total_len += max(v["length_frames"], 1)

    start_frame = min(v["start_frame"] for v in group)
    end_frame = max(v["end_frame"] for v in group)
    mean_score = total_score / max(total_len, 1)

    return {
        "stable_voice_id": stable_id,
        "source_voice_ids": " ".join(source_voice_ids),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "duration_frames": end_frame - start_frame + 1,
        "segment_count": len(group),
        "mean_score": f"{mean_score:.9f}",
        "start_note": all_notes[0] if all_notes else "",
        "end_note": all_notes[-1] if all_notes else "",
        "unique_note_count": len(set(all_notes)),
        "note_path": " ".join(all_notes),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stabilize fragmented causal voice trajectories into longer voice identities."
    )

    ap.add_argument("--voice_events_csv", required=True)

    ap.add_argument("--out_stable_voices_csv", required=True)
    ap.add_argument("--out_mapping_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--max_merge_gap_frames", type=int, default=12)
    ap.add_argument("--max_merge_pitch_jump", type=float, default=4.0)
    ap.add_argument("--min_stable_duration_frames", type=int, default=12)

    args = ap.parse_args()

    rows = _load_csv(Path(args.voice_events_csv))
    voices = [_voice_signature(r) for r in rows]

    voices.sort(key=lambda v: (v["start_frame"], v["voice_id"]))

    used = set()
    stable_groups: List[List[Dict[str, Any]]] = []

    for i, v in enumerate(voices):
        if i in used:
            continue

        group = [v]
        used.add(i)

        current = v

        changed = True
        while changed:
            changed = False
            best_j = None
            best_cost = 999999.0

            for j, cand in enumerate(voices):
                if j in used:
                    continue

                if not _can_merge(
                    current,
                    cand,
                    args.max_merge_gap_frames,
                    args.max_merge_pitch_jump,
                ):
                    continue

                gap = cand["start_frame"] - current["end_frame"]
                dist = _pitch_distance(current["end_note"], cand["start_note"])

                cost = gap * 0.50 + dist * 1.25 - cand["mean_score"] * 0.02

                if cost < best_cost:
                    best_cost = cost
                    best_j = j

            if best_j is not None:
                nxt = voices[best_j]
                group.append(nxt)
                used.add(best_j)
                current = nxt
                changed = True

        stable_groups.append(group)

    stable_rows = []
    mapping_rows = []

    for sid, group in enumerate(stable_groups, start=1):
        merged = _merge_voice_group(group, sid)

        if _safe_int(merged["duration_frames"], 0) < args.min_stable_duration_frames:
            continue

        stable_rows.append(merged)

        for v in group:
            mapping_rows.append({
                "stable_voice_id": sid,
                "source_voice_id": v["voice_id"],
                "source_start_frame": v["start_frame"],
                "source_end_frame": v["end_frame"],
                "source_start_note": v["start_note"],
                "source_end_note": v["end_note"],
            })

    stable_rows.sort(
        key=lambda r: (
            _safe_int(r["start_frame"]),
            _safe_int(r["stable_voice_id"]),
        )
    )

    out_stable = Path(args.out_stable_voices_csv)
    out_mapping = Path(args.out_mapping_csv)
    out_meta = Path(args.out_meta_json)
    out_txt = Path(args.out_summary_txt)

    out_stable.parent.mkdir(parents=True, exist_ok=True)

    stable_fields = [
        "stable_voice_id",
        "source_voice_ids",
        "start_frame",
        "end_frame",
        "duration_frames",
        "segment_count",
        "mean_score",
        "start_note",
        "end_note",
        "unique_note_count",
        "note_path",
    ]

    with out_stable.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=stable_fields)
        w.writeheader()
        w.writerows(stable_rows)

    mapping_fields = [
        "stable_voice_id",
        "source_voice_id",
        "source_start_frame",
        "source_end_frame",
        "source_start_note",
        "source_end_note",
    ]

    with out_mapping.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=mapping_fields)
        w.writeheader()
        w.writerows(mapping_rows)

    max_duration = max(
        (_safe_int(r["duration_frames"]) for r in stable_rows),
        default=0,
    )

    mean_duration = (
        sum(_safe_int(r["duration_frames"]) for r in stable_rows)
        / max(len(stable_rows), 1)
    )

    meta = {
        "stage": "voice_identity_stabilizer",
        "inputs": {
            "voice_events_csv": args.voice_events_csv,
        },
        "outputs": {
            "stable_voices_csv": args.out_stable_voices_csv,
            "mapping_csv": args.out_mapping_csv,
            "meta_json": args.out_meta_json,
            "summary_txt": args.out_summary_txt,
        },
        "parameters": {
            "max_merge_gap_frames": args.max_merge_gap_frames,
            "max_merge_pitch_jump": args.max_merge_pitch_jump,
            "min_stable_duration_frames": args.min_stable_duration_frames,
        },
        "result": {
            "input_voices": len(voices),
            "stable_voice_groups_total": len(stable_groups),
            "stable_voices_kept": len(stable_rows),
            "mapping_rows": len(mapping_rows),
            "max_duration_frames": max_duration,
            "mean_duration_frames": f"{mean_duration:.6f}",
        },
    }

    out_meta.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    txt = [
        "VOICE IDENTITY STABILIZER",
        "=" * 72,
        f"voice_events_csv          : {args.voice_events_csv}",
        "",
        f"input_voices              : {len(voices)}",
        f"stable_voice_groups_total : {len(stable_groups)}",
        f"stable_voices_kept        : {len(stable_rows)}",
        f"mapping_rows              : {len(mapping_rows)}",
        f"max_duration_frames       : {max_duration}",
        f"mean_duration_frames      : {mean_duration:.6f}",
        "",
        "Principle:",
        "  Merge fragmented causal trajectories into longer",
        "  stable voice identities.",
        "",
    ]

    out_txt.write_text("\n".join(txt), encoding="utf-8")

    print("voice identity stabilizer complete")
    print(json.dumps(meta["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()