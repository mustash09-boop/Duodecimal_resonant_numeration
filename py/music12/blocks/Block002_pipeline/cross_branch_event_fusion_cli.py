from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(str(x).replace(",", "."))
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


def _fusion_score(notechain: dict[str, Any], event_group: dict[str, Any], onset_window_frames: int) -> float:
    note_start = _safe_int(notechain.get("start_frame"), 0)
    event_start = _safe_int(event_group.get("start_frame"), 0)
    delta = abs(note_start - event_start)
    if delta > onset_window_frames:
        return -1.0e9

    note_dom = _normalize_note(notechain.get("dominant_note_token", "") or notechain.get("anchor_note_token", ""))
    note_coarse = _normalize_note(notechain.get("coarse_note", ""))
    event_dom = _normalize_note(event_group.get("dominant_note_token", ""))
    note_pcs = {pc for pc in (_pitch_class(note_dom), _pitch_class(note_coarse)) if pc}
    event_pcs = set(json.loads(str(event_group.get("pitch_classes_json", "[]")) or "[]"))
    if not event_pcs and event_dom:
        event_pcs.add(_pitch_class(event_dom))

    score = float(onset_window_frames - delta) / max(onset_window_frames, 1)
    if note_dom and event_dom and note_dom == event_dom:
        score += 1.20
    elif note_pcs and event_pcs and note_pcs.intersection(event_pcs):
        score += 0.55
    else:
        score += 0.10

    score += min(_safe_int(event_group.get("entity_count"), 0) / 12.0, 0.60)
    score += min(_safe_float(notechain.get("mean_exciter_confidence", notechain.get("mean_field_strength", 0.0)), 0.0), 1.0) * 0.0
    return score


def _support_kind(notechain: dict[str, Any], event_group: dict[str, Any]) -> str:
    note_dom = _normalize_note(notechain.get("dominant_note_token", "") or notechain.get("anchor_note_token", ""))
    note_coarse = _normalize_note(notechain.get("coarse_note", ""))
    event_dom = _normalize_note(event_group.get("dominant_note_token", ""))
    if note_dom and event_dom and note_dom == event_dom:
        return "exact_onset_support"
    note_pcs = {pc for pc in (_pitch_class(note_dom), _pitch_class(note_coarse)) if pc}
    event_pcs = set(json.loads(str(event_group.get("pitch_classes_json", "[]")) or "[]"))
    if not event_pcs and event_dom:
        event_pcs.add(_pitch_class(event_dom))
    if note_pcs and event_pcs and note_pcs.intersection(event_pcs):
        return "pitchclass_onset_support"
    return "foreign_field_support"


def _standalone_event_kind(
    event_group: dict[str, Any],
    max_standalone_onset_span_frames: int,
    min_standalone_compactness: float,
    min_standalone_attack_ratio: float,
    min_standalone_attack_compactness: float,
    nearby_notechain_rows: list[dict[str, Any]],
    field_claim_lookback_frames: int,
    field_claim_lookahead_frames: int,
) -> str:
    onset_span_frames = _safe_int(event_group.get("onset_span_frames"), 999999)
    onset_compactness = _safe_float(event_group.get("onset_compactness"), 0.0)
    attack_ratio = _safe_float(event_group.get("attack_ratio"), 0.0)
    attack_compactness = _safe_float(event_group.get("attack_compactness"), 0.0)
    event_start = _safe_int(event_group.get("start_frame"), 0)
    event_end = _safe_int(event_group.get("end_frame"), event_start)

    tail_pressure = False
    for row in nearby_notechain_rows:
        note_start = _safe_int(row.get("start_frame"), 0)
        note_end = _safe_int(row.get("end_frame"), note_start)
        if note_start > event_end + field_claim_lookahead_frames:
            continue
        if note_end < event_start - field_claim_lookback_frames:
            continue
        tail_pressure = True
        break

    if (
        onset_span_frames <= max_standalone_onset_span_frames
        and onset_compactness >= min_standalone_compactness
        and attack_ratio >= min_standalone_attack_ratio
        and attack_compactness >= min_standalone_attack_compactness
        and not (
            tail_pressure
            and onset_compactness < 0.93
            and attack_compactness < 0.88
        )
    ):
        return "event_field_only"
    return "ambient_field_residue"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fuse notechain events with event-field onset groups using only onset-local evidence, so long sustains remain stable across instruments."
    )
    ap.add_argument("--notechain-chains-csv", required=True)
    ap.add_argument("--event-field-groups-csv", required=True)
    ap.add_argument("--out-fused-events-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--onset-window-frames", type=int, default=5)
    ap.add_argument("--min-fusion-score", type=float, default=0.55)
    ap.add_argument("--max-pitchclass-support-onset-span-frames", type=int, default=6)
    ap.add_argument("--min-pitchclass-support-compactness", type=float, default=0.90)
    ap.add_argument("--min-pitchclass-support-attack-ratio", type=float, default=0.05)
    ap.add_argument("--max-standalone-onset-span-frames", type=int, default=8)
    ap.add_argument("--min-standalone-compactness", type=float, default=0.80)
    ap.add_argument("--min-standalone-attack-ratio", type=float, default=0.05)
    ap.add_argument("--min-standalone-attack-compactness", type=float, default=0.60)
    ap.add_argument("--field-claim-lookback-frames", type=int, default=3)
    ap.add_argument("--field-claim-lookahead-frames", type=int, default=2)
    ap.add_argument(
        "--allowed-support-kinds",
        nargs="+",
        default=["exact_onset_support", "pitchclass_onset_support"],
    )
    args = ap.parse_args()

    notechain_rows = _load_csv(Path(args.notechain_chains_csv))
    event_rows = _load_csv(Path(args.event_field_groups_csv))

    notechain_rows = [
        {
            **row,
            "start_frame": _safe_int(row.get("start_frame"), row.get("chain_start_frame", 0)),
            "end_frame": _safe_int(row.get("end_frame"), row.get("chain_end_frame", row.get("start_frame", 0))),
        }
        for row in notechain_rows
    ]
    notechain_rows.sort(key=lambda r: (_safe_int(r.get("start_frame"), 0), _safe_int(r.get("end_frame"), 0)))
    event_rows.sort(key=lambda r: (_safe_int(r.get("start_frame"), 0), _safe_int(r.get("end_frame"), 0)))

    fused_rows: list[dict[str, Any]] = []
    consumed_event_ids: set[str] = set()
    next_id = 1
    allowed_support_kinds = {str(value).strip() for value in args.allowed_support_kinds}
    support_kind_counter: dict[str, int] = {
        "exact_onset_support": 0,
        "pitchclass_onset_support": 0,
        "foreign_field_support": 0,
    }

    for notechain in notechain_rows:
        best_event = None
        best_score = -1.0
        for event_group in event_rows:
            event_id = str(event_group.get("event_group_id", ""))
            if event_id in consumed_event_ids:
                continue
            score = _fusion_score(notechain, event_group, int(args.onset_window_frames))
            if score > best_score:
                best_score = score
                best_event = event_group

        attached_event_ids: list[str] = []
        attached_event_count = 0
        attached_event_dominants: list[str] = []
        field_support = False
        field_start_frame = ""
        field_end_frame = ""
        field_support_kind = ""

        if best_event is not None and best_score >= float(args.min_fusion_score):
            field_support_kind = _support_kind(notechain, best_event)
            if field_support_kind == "pitchclass_onset_support":
                onset_span_frames = _safe_int(best_event.get("onset_span_frames"), 999999)
                onset_compactness = _safe_float(best_event.get("onset_compactness"), 0.0)
                attack_ratio = _safe_float(best_event.get("attack_ratio"), 0.0)
                if (
                    onset_span_frames > int(args.max_pitchclass_support_onset_span_frames)
                    or onset_compactness < float(args.min_pitchclass_support_compactness)
                    or attack_ratio < float(args.min_pitchclass_support_attack_ratio)
                ):
                    field_support_kind = "foreign_field_support"
            support_kind_counter[field_support_kind] = support_kind_counter.get(field_support_kind, 0) + 1
            if field_support_kind in allowed_support_kinds:
                event_id = str(best_event.get("event_group_id", ""))
                consumed_event_ids.add(event_id)
                attached_event_ids.append(event_id)
                attached_event_count = _safe_int(best_event.get("entity_count"), 0)
                field_support = True
                field_start_frame = str(best_event.get("start_frame", ""))
                field_end_frame = str(best_event.get("end_frame", ""))
                attached_event_dominants.append(str(best_event.get("dominant_note_token", "")).strip())

        fused_rows.append(
            {
                "fused_event_id": next_id,
                "event_kind": "notechain_backbone",
                "start_frame": _safe_int(notechain.get("start_frame"), 0),
                "end_frame": _safe_int(notechain.get("end_frame"), 0),
                "main_note_token": str(notechain.get("dominant_note_token", notechain.get("anchor_note_token", ""))).strip(),
                "backbone_proto_id": str(notechain.get("proto_exciter_id", "")),
                "field_support_attached": int(field_support),
                "field_support_start_frame": field_start_frame,
                "field_support_end_frame": field_end_frame,
                "field_support_entity_count": attached_event_count,
                "field_support_kind": field_support_kind,
                "field_support_dominants_json": json.dumps(attached_event_dominants, ensure_ascii=False),
                "member_event_group_ids_json": json.dumps(attached_event_ids, ensure_ascii=False),
            }
        )
        next_id += 1

    for event_group in event_rows:
        event_id = str(event_group.get("event_group_id", ""))
        if event_id in consumed_event_ids:
            continue
        event_start = _safe_int(event_group.get("start_frame"), 0)
        event_end = _safe_int(event_group.get("end_frame"), event_start)
        nearby_notechain_rows = [
            row for row in notechain_rows
            if not (
                _safe_int(row.get("start_frame"), 0) > event_end + int(args.field_claim_lookahead_frames)
                or _safe_int(row.get("end_frame"), _safe_int(row.get("start_frame"), 0)) < event_start - int(args.field_claim_lookback_frames)
            )
        ]
        event_kind = _standalone_event_kind(
            event_group,
            int(args.max_standalone_onset_span_frames),
            float(args.min_standalone_compactness),
            float(args.min_standalone_attack_ratio),
            float(args.min_standalone_attack_compactness),
            nearby_notechain_rows,
            int(args.field_claim_lookback_frames),
            int(args.field_claim_lookahead_frames),
        )
        fused_rows.append(
            {
                "fused_event_id": next_id,
                "event_kind": event_kind,
                "start_frame": _safe_int(event_group.get("start_frame"), 0),
                "end_frame": _safe_int(event_group.get("end_frame"), 0),
                "main_note_token": str(event_group.get("dominant_note_token", "")).strip(),
                "backbone_proto_id": "",
                "field_support_attached": 1,
                "field_support_start_frame": str(event_group.get("start_frame", "")),
                "field_support_end_frame": str(event_group.get("end_frame", "")),
                "field_support_entity_count": _safe_int(event_group.get("entity_count"), 0),
                "field_support_kind": "event_field_only",
                "field_support_dominants_json": json.dumps([str(event_group.get("dominant_note_token", "")).strip()], ensure_ascii=False),
                "member_event_group_ids_json": json.dumps([event_id], ensure_ascii=False),
            }
        )
        next_id += 1

    fused_rows.sort(key=lambda r: (_safe_int(r.get("start_frame"), 0), _safe_int(r.get("end_frame"), 0), _safe_int(r.get("fused_event_id"), 0)))

    out_csv = Path(args.out_fused_events_csv)
    out_summary = Path(args.out_summary_txt)
    out_meta = Path(args.out_meta_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "fused_event_id",
        "event_kind",
        "start_frame",
        "end_frame",
        "main_note_token",
        "backbone_proto_id",
        "field_support_attached",
        "field_support_start_frame",
        "field_support_end_frame",
        "field_support_entity_count",
        "field_support_kind",
        "field_support_dominants_json",
        "member_event_group_ids_json",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(fused_rows)

    backbone_count = sum(1 for row in fused_rows if str(row.get("event_kind", "")) == "notechain_backbone")
    field_only_count = sum(1 for row in fused_rows if str(row.get("event_kind", "")) == "event_field_only")
    ambient_residue_count = sum(1 for row in fused_rows if str(row.get("event_kind", "")) == "ambient_field_residue")
    attached_count = sum(_safe_int(row.get("field_support_attached"), 0) for row in fused_rows if str(row.get("event_kind", "")) == "notechain_backbone")

    summary_lines = [
        "CROSS BRANCH EVENT FUSION",
        "=" * 72,
        f"notechain_backbone_events : {backbone_count}",
        f"event_field_only_events   : {field_only_count}",
        f"ambient_field_residue     : {ambient_residue_count}",
        f"fused_event_count         : {len(fused_rows)}",
        f"musical_event_count       : {backbone_count + field_only_count}",
        f"attached_field_support    : {attached_count}",
    ]
    for key in ("exact_onset_support", "pitchclass_onset_support", "foreign_field_support"):
        summary_lines.append(f"{key:<24}: {support_kind_counter.get(key, 0)}")
    out_summary.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    meta = {
        "stage": "cross_branch_event_fusion",
        "inputs": {
            "notechain_chains_csv": args.notechain_chains_csv,
            "event_field_groups_csv": args.event_field_groups_csv,
        },
        "parameters": {
            "onset_window_frames": int(args.onset_window_frames),
            "min_fusion_score": float(args.min_fusion_score),
            "max_pitchclass_support_onset_span_frames": int(args.max_pitchclass_support_onset_span_frames),
            "min_pitchclass_support_compactness": float(args.min_pitchclass_support_compactness),
            "min_pitchclass_support_attack_ratio": float(args.min_pitchclass_support_attack_ratio),
            "max_standalone_onset_span_frames": int(args.max_standalone_onset_span_frames),
            "min_standalone_compactness": float(args.min_standalone_compactness),
            "min_standalone_attack_ratio": float(args.min_standalone_attack_ratio),
            "min_standalone_attack_compactness": float(args.min_standalone_attack_compactness),
            "field_claim_lookback_frames": int(args.field_claim_lookback_frames),
            "field_claim_lookahead_frames": int(args.field_claim_lookahead_frames),
            "allowed_support_kinds": sorted(allowed_support_kinds),
        },
        "result": {
            "notechain_backbone_events": backbone_count,
            "event_field_only_events": field_only_count,
            "ambient_field_residue": ambient_residue_count,
            "fused_event_count": len(fused_rows),
            "musical_event_count": backbone_count + field_only_count,
            "attached_field_support": attached_count,
            "support_kind_counter": support_kind_counter,
        },
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
