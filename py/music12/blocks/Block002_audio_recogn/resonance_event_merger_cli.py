# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
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


def _normalize_note(token: str) -> str:
    s = str(token or "").strip()
    if not s:
        return ""
    if "'" in s:
        return s.split("'", 1)[0] + "'-"
    if s.endswith("-"):
        return s[:-1] + "'-"
    return s + "'-"


def _parse_path(raw: str) -> List[str]:
    return [x.strip() for x in str(raw or "").split() if x.strip()]


def _can_merge(a: Dict[str, Any], b: Dict[str, Any], max_gap_frames: int, max_birth_jump_ratio: float) -> bool:
    if _normalize_note(a.get("candidate_note", "")) != _normalize_note(b.get("candidate_note", "")):
        return False

    gap = _safe_int(b.get("birth_frame"), 0) - _safe_int(a.get("end_frame"), 0)

    if gap < 0 or gap > max_gap_frames:
        return False

    a_final = _safe_float(a.get("final_score"), 0.0)
    b_birth = _safe_float(b.get("birth_score"), 0.0)
    a_max = max(_safe_float(a.get("max_score"), 0.0), 1e-9)

    # Если второе событие родилось не как сильный новый удар,
    # а как продолжение/возврат той же энергии — сшиваем.
    birth_jump_ratio = max(0.0, b_birth - a_final) / a_max

    if birth_jump_ratio <= max_birth_jump_ratio:
        return True

    # Даже если jump есть, но gap совсем маленький — вероятно дробление tracking.
    if gap <= 2 and b_birth <= a_max * 1.15:
        return True

    return False


def _merge_group(group: List[Dict[str, Any]], merged_id: int) -> Dict[str, Any]:
    group = sorted(group, key=lambda r: _safe_int(r.get("birth_frame"), 0))

    note = _normalize_note(group[0].get("candidate_note", ""))

    birth_frame = min(_safe_int(r.get("birth_frame"), 0) for r in group)
    end_frame = max(_safe_int(r.get("end_frame"), 0) for r in group)

    frame_count = sum(_safe_int(r.get("frame_count"), 0) for r in group)

    weighted_score_sum = 0.0
    total_weight = 0

    max_score = 0.0
    birth_score = _safe_float(group[0].get("birth_score"), 0.0)
    final_score = _safe_float(group[-1].get("final_score"), 0.0)
    attack_peak_frame = _safe_int(group[0].get("attack_peak_frame"), birth_frame)

    note_path: List[str] = []
    state_path: List[str] = []

    counts = {
        "birth_count": 0,
        "birth_like_count": 0,
        "re_excitation_count": 0,
        "active_body_count": 0,
        "sustain_body_count": 0,
        "response_trace_count": 0,
        "decay_trace_count": 0,
    }

    source_ids = []

    for r in group:
        source_ids.append(str(r.get("event_id", "")))

        fc = max(_safe_int(r.get("frame_count"), 0), 1)
        mean_s = _safe_float(r.get("mean_score"), 0.0)

        weighted_score_sum += mean_s * fc
        total_weight += fc

        rmax = _safe_float(r.get("max_score"), 0.0)
        if rmax > max_score:
            max_score = rmax
            attack_peak_frame = _safe_int(r.get("attack_peak_frame"), attack_peak_frame)

        note_path.extend(_parse_path(r.get("note_path", "")))
        state_path.extend(_parse_path(r.get("state_path", "")))

        for k in counts:
            counts[k] += _safe_int(r.get(k), 0)

    mean_score = weighted_score_sum / max(total_weight, 1)

    if counts["re_excitation_count"] > 0 and len(group) > 1:
        lifecycle_kind = "merged_re_excited_lifecycle"
    elif counts["active_body_count"] + counts["sustain_body_count"] >= counts["decay_trace_count"]:
        lifecycle_kind = "merged_sustained_lifecycle"
    else:
        lifecycle_kind = "merged_trace_lifecycle"

    return {
        "merged_event_id": merged_id,
        "source_event_ids": " ".join(source_ids),
        "candidate_note": note,
        "birth_frame": birth_frame,
        "end_frame": end_frame,
        "duration_frames": end_frame - birth_frame + 1,
        "frame_count": frame_count,
        "segment_count": len(group),
        "mean_score": f"{mean_score:.9f}",
        "max_score": f"{max_score:.9f}",
        "birth_score": f"{birth_score:.9f}",
        "final_score": f"{final_score:.9f}",
        "attack_peak_frame": attack_peak_frame,
        "unique_note_count": len(set(note_path)),
        "note_path": " ".join(note_path),
        "state_path": " ".join(state_path),
        **counts,
        "lifecycle_kind": lifecycle_kind,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge fragmented same-note resonance lifecycle events."
    )

    ap.add_argument("--events_csv", required=True)

    ap.add_argument("--out_merged_events_csv", required=True)
    ap.add_argument("--out_mapping_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--max_merge_gap_frames", type=int, default=10)
    ap.add_argument("--max_birth_jump_ratio", type=float, default=0.28)
    ap.add_argument("--min_merged_frames", type=int, default=4)

    args = ap.parse_args()

    rows = _load_csv(Path(args.events_csv))
    rows.sort(
        key=lambda r: (
            _normalize_note(r.get("candidate_note", "")),
            _safe_int(r.get("birth_frame"), 0),
            _safe_int(r.get("event_id"), 0),
        )
    )

    groups: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []

    for r in rows:
        if not current:
            current = [r]
            continue

        last = current[-1]

        if _can_merge(
            last,
            r,
            max_gap_frames=args.max_merge_gap_frames,
            max_birth_jump_ratio=args.max_birth_jump_ratio,
        ):
            current.append(r)
        else:
            groups.append(current)
            current = [r]

    if current:
        groups.append(current)

    merged_rows = []
    mapping_rows = []

    for idx, group in enumerate(groups, start=1):
        merged = _merge_group(group, idx)

        if _safe_int(merged.get("frame_count"), 0) < args.min_merged_frames:
            continue

        merged_rows.append(merged)

        for src in group:
            mapping_rows.append({
                "merged_event_id": idx,
                "source_event_id": src.get("event_id", ""),
                "candidate_note": _normalize_note(src.get("candidate_note", "")),
                "source_birth_frame": src.get("birth_frame", ""),
                "source_end_frame": src.get("end_frame", ""),
            })

    merged_rows.sort(
        key=lambda r: (
            _safe_int(r.get("birth_frame"), 0),
            _normalize_note(r.get("candidate_note", "")),
        )
    )

    out_events = Path(args.out_merged_events_csv)
    out_mapping = Path(args.out_mapping_csv)
    out_meta = Path(args.out_meta_json)
    out_txt = Path(args.out_summary_txt)

    out_events.parent.mkdir(parents=True, exist_ok=True)

    event_fields = [
        "merged_event_id",
        "source_event_ids",
        "candidate_note",
        "birth_frame",
        "end_frame",
        "duration_frames",
        "frame_count",
        "segment_count",
        "mean_score",
        "max_score",
        "birth_score",
        "final_score",
        "attack_peak_frame",
        "unique_note_count",
        "note_path",
        "state_path",
        "birth_count",
        "birth_like_count",
        "re_excitation_count",
        "active_body_count",
        "sustain_body_count",
        "response_trace_count",
        "decay_trace_count",
        "lifecycle_kind",
    ]

    with out_events.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=event_fields)
        w.writeheader()
        w.writerows(merged_rows)

    with out_mapping.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "merged_event_id",
                "source_event_id",
                "candidate_note",
                "source_birth_frame",
                "source_end_frame",
            ],
        )
        w.writeheader()
        w.writerows(mapping_rows)

    lifecycle_counts: Dict[str, int] = {}
    for r in merged_rows:
        k = str(r.get("lifecycle_kind", ""))
        lifecycle_counts[k] = lifecycle_counts.get(k, 0) + 1

    segment_distribution: Dict[int, int] = {}
    for r in merged_rows:
        n = _safe_int(r.get("segment_count"), 0)
        segment_distribution[n] = segment_distribution.get(n, 0) + 1

    meta = {
        "stage": "resonance_event_merger",
        "inputs": {
            "events_csv": args.events_csv,
        },
        "outputs": {
            "merged_events_csv": args.out_merged_events_csv,
            "mapping_csv": args.out_mapping_csv,
            "meta_json": args.out_meta_json,
            "summary_txt": args.out_summary_txt,
        },
        "parameters": {
            "max_merge_gap_frames": args.max_merge_gap_frames,
            "max_birth_jump_ratio": args.max_birth_jump_ratio,
            "min_merged_frames": args.min_merged_frames,
        },
        "result": {
            "input_events": len(rows),
            "merged_events": len(merged_rows),
            "mapping_rows": len(mapping_rows),
            "lifecycle_counts": lifecycle_counts,
            "segment_distribution": segment_distribution,
        },
    }

    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    txt = [
        "RESONANCE EVENT MERGER",
        "=" * 72,
        f"events_csv    : {args.events_csv}",
        "",
        f"input_events  : {len(rows)}",
        f"merged_events : {len(merged_rows)}",
        f"mapping_rows  : {len(mapping_rows)}",
        "",
        "Lifecycle counts:",
    ]

    for k in sorted(lifecycle_counts):
        txt.append(f"  {k}: {lifecycle_counts[k]}")

    txt.append("")
    txt.append("Segment distribution:")
    for k in sorted(segment_distribution):
        txt.append(f"  {k}: {segment_distribution[k]}")

    txt.extend([
        "",
        "Principle:",
        "  One musical/resonance event may be fragmented by local score pulsation.",
        "  This module merges same-note lifecycle fragments when there is no strong new birth.",
        "",
    ])

    out_txt.write_text("\n".join(txt), encoding="utf-8")

    print("resonance event merger complete")
    print(json.dumps(meta["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()