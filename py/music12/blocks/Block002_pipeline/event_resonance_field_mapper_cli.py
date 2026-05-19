from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _normalize_note(note: str) -> str:
    s = str(note or "").strip()
    if not s:
        return ""
    if "'" in s:
        return s.split("'", 1)[0] + "'-"
    if s.endswith("-"):
        return s[:-1] + "'-"
    return s + "'-"


def _pitch_class(note: str) -> str:
    try:
        return _normalize_note(note).split(".", 1)[1].split("'", 1)[0]
    except Exception:
        return ""


def _build_families_by_frame(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[_safe_int(row.get("frame_index"), 0)].append(row)
    for frame_rows in out.values():
        frame_rows.sort(key=lambda r: _safe_int(r.get("family_rank"), 999999))
    return out


def _field_row_score(row: dict[str, Any], anchor_pc: str) -> float:
    family_score = _safe_float(row.get("family_score"), 0.0)
    root_micro_count = _safe_int(row.get("root_micro_count"), 0)
    root_micro_diversity = _safe_int(row.get("root_micro_diversity"), 0)
    family_note = _normalize_note(row.get("family_root_note_micro", ""))
    same_pc_bonus = 0.35 if anchor_pc and _pitch_class(family_note) == anchor_pc else 0.0
    return family_score + min(root_micro_count / 32.0, 1.2) + min(root_micro_diversity / 24.0, 0.7) + same_pc_bonus


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Map event-like proto-exciters into gesture/resonance-field persistence without forcing note-chain identity."
    )
    ap.add_argument("--event-field-proto-exciters-csv", required=True)
    ap.add_argument("--micro-families-csv", required=True)
    ap.add_argument("--out-event-field-frames-csv", required=True)
    ap.add_argument("--out-event-field-entities-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--lookahead-frames", type=int, default=12)
    ap.add_argument("--family-rank-limit", type=int, default=8)
    ap.add_argument("--min-family-score", type=float, default=0.90)
    ap.add_argument("--max-gap-frames", type=int, default=2)
    args = ap.parse_args()

    proto_rows = _load_csv(Path(args.event_field_proto_exciters_csv))
    family_rows = _load_csv(Path(args.micro_families_csv))
    families_by_frame = _build_families_by_frame(family_rows)

    frame_rows: list[dict[str, Any]] = []
    entity_rows: list[dict[str, Any]] = []
    phase_counter: Counter[str] = Counter()

    for proto in proto_rows:
        proto_id = _safe_int(proto.get("proto_exciter_id"), 0)
        coarse_note = _normalize_note(proto.get("coarse_note", ""))
        anchor_pc = _pitch_class(coarse_note)
        start_frame = _safe_int(proto.get("start_frame"), 0)
        end_frame = _safe_int(proto.get("end_frame"), start_frame)
        duration_frames = _safe_int(proto.get("duration_frames"), max(1, end_frame - start_frame + 1))
        peak_frame = _safe_int(proto.get("peak_frame"), start_frame)
        exciter_confidence = _safe_float(proto.get("exciter_confidence"), 0.0)

        proto_frame_rows: list[dict[str, Any]] = []
        note_counter: Counter[str] = Counter()
        phase_local_counter: Counter[str] = Counter()
        gaps = 0

        for frame_index in range(start_frame, end_frame + int(args.lookahead_frames) + 1):
            rows = families_by_frame.get(frame_index, [])
            selected: list[tuple[dict[str, Any], float]] = []
            for row in rows[: int(args.family_rank_limit)]:
                family_score = _safe_float(row.get("family_score"), 0.0)
                if family_score < float(args.min_family_score):
                    continue
                selected.append((row, _field_row_score(row, anchor_pc)))
            if not selected:
                gaps += 1
                if gaps > int(args.max_gap_frames):
                    break
                continue

            gaps = 0
            selected.sort(key=lambda item: item[1], reverse=True)
            dominant_row, dominant_score = selected[0]
            dominant_note = _normalize_note(dominant_row.get("family_root_note_micro", ""))
            dominant_pc = _pitch_class(dominant_note)
            same_pc_count = sum(1 for row, _ in selected if anchor_pc and _pitch_class(_normalize_note(row.get("family_root_note_micro", ""))) == anchor_pc)
            field_strength = sum(score for _, score in selected)
            diversity = len({_normalize_note(row.get("family_root_note_micro", "")) for row, _ in selected if _normalize_note(row.get("family_root_note_micro", ""))})
            anchor_match_ratio = same_pc_count / max(len(selected), 1)

            if frame_index <= peak_frame:
                phase = "ATTACK"
            elif dominant_pc == anchor_pc and anchor_match_ratio >= 0.5:
                phase = "RESONANCE_FIELD"
            else:
                phase = "DECAY_FIELD"

            phase_counter[phase] += 1
            phase_local_counter[phase] += 1
            note_counter[dominant_note] += 1
            proto_frame_rows.append(
                {
                    "proto_exciter_id": proto_id,
                    "frame_index": frame_index,
                    "phase": phase,
                    "dominant_note_token": dominant_note,
                    "anchor_note_token": coarse_note,
                    "field_strength": f"{field_strength:.9f}",
                    "dominant_score": f"{dominant_score:.9f}",
                    "field_diversity": diversity,
                    "anchor_match_ratio": f"{anchor_match_ratio:.9f}",
                    "selected_family_count": len(selected),
                }
            )

        if not proto_frame_rows:
            continue

        frame_rows.extend(proto_frame_rows)
        entity_rows.append(
            {
                "proto_exciter_id": proto_id,
                "anchor_note_token": coarse_note,
                "start_frame": _safe_int(proto_frame_rows[0].get("frame_index"), 0),
                "end_frame": _safe_int(proto_frame_rows[-1].get("frame_index"), 0),
                "proto_duration_frames": duration_frames,
                "frame_count": len(proto_frame_rows),
                "exciter_confidence": f"{exciter_confidence:.9f}",
                "dominant_note_token": note_counter.most_common(1)[0][0] if note_counter else "",
                "mean_anchor_match_ratio": f"{(sum(_safe_float(row.get('anchor_match_ratio'), 0.0) for row in proto_frame_rows) / max(len(proto_frame_rows), 1)):.9f}",
                "mean_field_strength": f"{(sum(_safe_float(row.get('field_strength'), 0.0) for row in proto_frame_rows) / max(len(proto_frame_rows), 1)):.9f}",
                "max_field_diversity": max(_safe_int(row.get("field_diversity"), 0) for row in proto_frame_rows),
                "phase_counts_json": json.dumps(dict(phase_local_counter), ensure_ascii=False, sort_keys=True),
                "dominant_notes_json": json.dumps(dict(note_counter), ensure_ascii=False, sort_keys=True),
            }
        )

    frame_rows.sort(key=lambda r: (_safe_int(r.get("frame_index"), 0), _safe_int(r.get("proto_exciter_id"), 0)))
    entity_rows.sort(key=lambda r: (_safe_int(r.get("start_frame"), 0), _safe_int(r.get("proto_exciter_id"), 0)))

    out_frames = Path(args.out_event_field_frames_csv)
    out_entities = Path(args.out_event_field_entities_csv)
    out_summary = Path(args.out_summary_txt)
    out_meta = Path(args.out_meta_json)
    out_frames.parent.mkdir(parents=True, exist_ok=True)

    frame_fields = [
        "proto_exciter_id",
        "frame_index",
        "phase",
        "dominant_note_token",
        "anchor_note_token",
        "field_strength",
        "dominant_score",
        "field_diversity",
        "anchor_match_ratio",
        "selected_family_count",
    ]
    with out_frames.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=frame_fields)
        w.writeheader()
        w.writerows(frame_rows)

    entity_fields = [
        "proto_exciter_id",
        "anchor_note_token",
        "start_frame",
        "end_frame",
        "proto_duration_frames",
        "frame_count",
        "exciter_confidence",
        "dominant_note_token",
        "mean_anchor_match_ratio",
        "mean_field_strength",
        "max_field_diversity",
        "phase_counts_json",
        "dominant_notes_json",
    ]
    with out_entities.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=entity_fields)
        w.writeheader()
        w.writerows(entity_rows)

    summary_lines = [
        "EVENT RESONANCE FIELD MAPPER",
        "=" * 72,
        f"input_event_field_proto: {len(proto_rows)}",
        f"event_field_entities   : {len(entity_rows)}",
        f"event_field_frames     : {len(frame_rows)}",
    ]
    for key in sorted(phase_counter):
        summary_lines.append(f"{key.lower():<22}: {phase_counter[key]}")
    out_summary.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    meta = {
        "stage": "event_resonance_field_mapper",
        "inputs": {
            "event_field_proto_exciters_csv": args.event_field_proto_exciters_csv,
            "micro_families_csv": args.micro_families_csv,
        },
        "parameters": {
            "lookahead_frames": int(args.lookahead_frames),
            "family_rank_limit": int(args.family_rank_limit),
            "min_family_score": float(args.min_family_score),
            "max_gap_frames": int(args.max_gap_frames),
        },
        "result": {
            "input_event_field_proto": len(proto_rows),
            "event_field_entities": len(entity_rows),
            "event_field_frames": len(frame_rows),
            "phase_counts": dict(phase_counter),
        },
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
