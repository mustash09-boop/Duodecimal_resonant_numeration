# -*- coding: ascii -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def _safe_int(value: str | None, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _safe_float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return default


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _normalize_token(token: str | None) -> str:
    return str(token or "").strip()


def _token_octave_pitchclass(token: str | None) -> tuple[str, str]:
    text = _normalize_token(token)
    if not text:
        return "", ""
    if "." in text:
        left, right = text.split(".", 1)
        right = right.replace("'", "").replace("-", "").strip()
        return left.strip(), right.strip()
    cleaned = text.replace("'", "").replace("-", "").strip()
    return "", cleaned


def _pitchclass(token: str | None) -> str:
    return _token_octave_pitchclass(token)[1]


def _same_pitchclass(left: str | None, right: str | None) -> bool:
    l_pc = _pitchclass(left)
    r_pc = _pitchclass(right)
    return bool(l_pc and r_pc and l_pc == r_pc)


def _overlap_len(a0: int, a1: int, b0: int, b1: int) -> int:
    lo = max(a0, b0)
    hi = min(a1, b1)
    return max(0, hi - lo + 1)


def _build_time_buckets(rows: list[dict[str, str]], start_key: str, bucket_size: int = 24) -> dict[int, list[dict[str, str]]]:
    buckets: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        bucket = _safe_int(row.get(start_key)) // bucket_size
        buckets.setdefault(bucket, []).append(row)
    return buckets


def _best_match(
    bucket_index: dict[int, list[dict[str, str]]],
    start_key: str,
    end_key: str,
    note_keys: list[str],
    event_start: int,
    event_end: int,
    event_note: str,
    top_k: int = 8,
    bucket_size: int = 24,
    slack_frames: int = 36,
) -> list[dict[str, str]]:
    ranked: list[tuple[float, dict[str, str]]] = []
    start_bucket = max(0, (event_start - slack_frames) // bucket_size)
    end_bucket = (event_end + slack_frames) // bucket_size
    candidates: list[dict[str, str]] = []
    seen_ids: set[int] = set()
    for bucket in range(start_bucket, end_bucket + 1):
        for row in bucket_index.get(bucket, []):
            row_id = id(row)
            if row_id in seen_ids:
                continue
            seen_ids.add(row_id)
            candidates.append(row)
    for row in candidates:
        row_start = _safe_int(row.get(start_key))
        row_end = _safe_int(row.get(end_key))
        overlap = _overlap_len(event_start, event_end, row_start, row_end)
        if overlap <= 0 and abs(row_start - event_start) > slack_frames and abs(row_end - event_end) > slack_frames:
            continue
        note_bonus = 0.0
        for key in note_keys:
            token = row.get(key, "")
            if _normalize_token(token) == event_note and event_note:
                note_bonus = max(note_bonus, 1.0)
            elif _same_pitchclass(token, event_note):
                note_bonus = max(note_bonus, 0.55)
        time_proximity = 1.0 / (1.0 + abs(row_start - event_start))
        score = overlap + note_bonus + 0.25 * time_proximity
        if score > 0.0:
            ranked.append((score, row))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [row for _score, row in ranked[:top_k]]


def _build_parent_links(rows: list[dict[str, str]], max_gap: int = 24) -> tuple[dict[int, int], dict[int, list[int]], Counter[str]]:
    ordered = sorted(rows, key=lambda r: (_safe_int(r.get("birth_frame")), _safe_int(r.get("end_frame")), _safe_int(r.get("merged_event_id"))))
    parent_of: dict[int, int] = {}
    children_of: dict[int, list[int]] = {}
    relation_counter: Counter[str] = Counter()

    for idx, row in enumerate(ordered):
        event_id = _safe_int(row.get("merged_event_id"))
        start = _safe_int(row.get("birth_frame"))
        note = _normalize_token(row.get("candidate_note"))
        best_parent = 0
        best_score = -1.0
        best_relation = ""
        for prev in ordered[:idx]:
            prev_id = _safe_int(prev.get("merged_event_id"))
            prev_start = _safe_int(prev.get("birth_frame"))
            prev_end = _safe_int(prev.get("end_frame"))
            if prev_end > start:
                continue
            gap = start - prev_end
            if gap > max_gap:
                continue
            prev_note = _normalize_token(prev.get("candidate_note"))
            relation = ""
            note_score = 0.0
            if prev_note == note and note:
                relation = "SAME_NOTE_CONTINUATION"
                note_score = 1.0
            elif _same_pitchclass(prev_note, note):
                relation = "SAME_PITCHCLASS_CONTINUATION"
                note_score = 0.72
            else:
                prev_event_type = str(prev.get("initial_event_type", "")).strip()
                this_event_type = str(row.get("initial_event_type", "")).strip()
                if prev_event_type == "DISCRETE_PIANO_EXCITATION" and this_event_type in {"BODY_RETURN_EVENT", "MIXED_EVENT"}:
                    relation = "ATTACK_TO_BODY_CHILD"
                    note_score = 0.48
                else:
                    continue
            time_score = 1.0 / (1.0 + gap)
            total = note_score + 0.35 * time_score + 0.05 * (1.0 / (1.0 + abs(prev_start - start)))
            if total > best_score:
                best_score = total
                best_parent = prev_id
                best_relation = relation
        if best_parent:
            parent_of[event_id] = best_parent
            children_of.setdefault(best_parent, []).append(event_id)
            relation_counter[best_relation] += 1
    return parent_of, children_of, relation_counter


def _concurrent_neighbors(rows: list[dict[str, str]], max_neighbors: int) -> dict[int, list[int]]:
    ordered = sorted(rows, key=lambda r: (_safe_int(r.get("birth_frame")), _safe_int(r.get("end_frame")), _safe_int(r.get("merged_event_id"))))
    out: dict[int, list[int]] = {}
    for row in ordered:
        event_id = _safe_int(row.get("merged_event_id"))
        start = _safe_int(row.get("birth_frame"))
        neighbors: list[int] = []
        for other in ordered:
            other_id = _safe_int(other.get("merged_event_id"))
            if other_id == event_id:
                continue
            o_start = _safe_int(other.get("birth_frame"))
            o_end = _safe_int(other.get("end_frame"))
            if o_start <= start <= o_end:
                neighbors.append(other_id)
        neighbors.sort()
        out[event_id] = neighbors[:max_neighbors]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a full pre-passport event census over the whole file: events, slots, raw/chain/field/sustain attachments, concurrency, and parent-child links.")
    ap.add_argument("--event_slots_csv", required=True)
    ap.add_argument("--proto_exciters_csv", required=True)
    ap.add_argument("--primary_note_chains_csv", required=True)
    ap.add_argument("--event_field_entities_csv", required=True)
    ap.add_argument("--event_field_frames_csv", required=True)
    ap.add_argument("--controlled_sustain_chains_csv", required=True)
    ap.add_argument("--max_parallel_checks", type=int, default=10)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_summary_txt", required=True)
    ap.add_argument("--out_meta_json", required=True)
    args = ap.parse_args()

    event_rows = _load_csv(Path(args.event_slots_csv))
    proto_rows = _load_csv(Path(args.proto_exciters_csv))
    chain_rows = _load_csv(Path(args.primary_note_chains_csv))
    entity_rows = _load_csv(Path(args.event_field_entities_csv))
    field_frame_rows = _load_csv(Path(args.event_field_frames_csv))
    sustain_rows = _load_csv(Path(args.controlled_sustain_chains_csv))

    proto_buckets = _build_time_buckets(proto_rows, "start_frame")
    chain_buckets = _build_time_buckets(chain_rows, "chain_start_frame")
    entity_buckets = _build_time_buckets(entity_rows, "start_frame")
    sustain_buckets = _build_time_buckets(sustain_rows, "start_frame")

    parent_of, children_of, relation_counter = _build_parent_links(event_rows)
    neighbor_index = _concurrent_neighbors(event_rows, max_neighbors=int(args.max_parallel_checks))

    frame_rows_by_proto: dict[int, list[dict[str, str]]] = {}
    for row in field_frame_rows:
        proto_id = _safe_int(row.get("proto_exciter_id"))
        frame_rows_by_proto.setdefault(proto_id, []).append(row)

    out_rows: list[dict[str, str]] = []
    matched_proto_count = 0
    matched_chain_count = 0
    matched_entity_count = 0
    matched_sustain_count = 0
    event_type_counter: Counter[str] = Counter()

    for row in event_rows:
        event_id = _safe_int(row.get("merged_event_id"))
        event_note = _normalize_token(row.get("candidate_note"))
        start = _safe_int(row.get("birth_frame"))
        end = _safe_int(row.get("end_frame"))

        proto_matches = _best_match(
            proto_buckets,
            start_key="start_frame",
            end_key="end_frame",
            note_keys=["coarse_note", "peak_note_token", "dominant_note_token"],
            event_start=start,
            event_end=end,
            event_note=event_note,
            top_k=int(args.max_parallel_checks),
        )
        chain_matches = _best_match(
            chain_buckets,
            start_key="chain_start_frame",
            end_key="chain_end_frame",
            note_keys=["coarse_note", "dominant_note_token"],
            event_start=start,
            event_end=end,
            event_note=event_note,
            top_k=int(args.max_parallel_checks),
        )
        entity_matches = _best_match(
            entity_buckets,
            start_key="start_frame",
            end_key="end_frame",
            note_keys=["anchor_note_token", "dominant_note_token"],
            event_start=start,
            event_end=end,
            event_note=event_note,
            top_k=int(args.max_parallel_checks),
        )
        sustain_matches = _best_match(
            sustain_buckets,
            start_key="start_frame",
            end_key="end_frame",
            note_keys=["coarse_note", "anchor_note_token", "dominant_note_token"],
            event_start=start,
            event_end=end,
            event_note=event_note,
            top_k=int(args.max_parallel_checks),
        )

        best_proto = proto_matches[0] if proto_matches else {}
        best_chain = chain_matches[0] if chain_matches else {}
        best_entity = entity_matches[0] if entity_matches else {}
        best_sustain = sustain_matches[0] if sustain_matches else {}

        if best_proto:
            matched_proto_count += 1
        if best_chain:
            matched_chain_count += 1
        if best_entity:
            matched_entity_count += 1
        if best_sustain:
            matched_sustain_count += 1

        proto_id = _safe_int(best_entity.get("proto_exciter_id") or best_sustain.get("proto_exciter_id") or best_chain.get("proto_exciter_id") or best_proto.get("proto_exciter_id"))
        frame_rows = frame_rows_by_proto.get(proto_id, [])
        anchor_match_values = [_safe_float(r.get("anchor_match_ratio")) for r in frame_rows]
        field_strength_values = [_safe_float(r.get("field_strength")) for r in frame_rows]
        field_diversity_values = [_safe_float(r.get("field_diversity")) for r in frame_rows]
        dominant_scores = [_safe_float(r.get("dominant_score")) for r in frame_rows]

        parent_event_id = parent_of.get(event_id, 0)
        parent_note = ""
        if parent_event_id:
            for src in event_rows:
                if _safe_int(src.get("merged_event_id")) == parent_event_id:
                    parent_note = _normalize_token(src.get("candidate_note"))
                    break
        relation_kind = ""
        if parent_event_id:
            if parent_note == event_note and event_note:
                relation_kind = "SAME_NOTE_CONTINUATION"
            elif _same_pitchclass(parent_note, event_note):
                relation_kind = "SAME_PITCHCLASS_CONTINUATION"
            else:
                relation_kind = "ATTACK_TO_BODY_CHILD"

        child_ids = children_of.get(event_id, [])
        event_type = str(row.get("initial_event_type", "")).strip()
        event_type_counter[event_type] += 1

        out = dict(row)
        out["raw_attached_proto_ids_json"] = json.dumps([_safe_int(r.get("proto_exciter_id")) for r in proto_matches], ensure_ascii=False)
        out["chain_attached_proto_ids_json"] = json.dumps([_safe_int(r.get("proto_exciter_id")) for r in chain_matches], ensure_ascii=False)
        out["field_attached_proto_ids_json"] = json.dumps([_safe_int(r.get("proto_exciter_id")) for r in entity_matches], ensure_ascii=False)
        out["sustain_attached_proto_ids_json"] = json.dumps([_safe_int(r.get("proto_exciter_id")) for r in sustain_matches], ensure_ascii=False)
        out["best_proto_peak_hz"] = f"{_safe_float(best_proto.get('peak_frequency_hz')):.9f}" if best_proto else ""
        out["best_proto_peak_seed_score"] = f"{_safe_float(best_proto.get('peak_seed_score')):.9f}" if best_proto else ""
        out["best_proto_max_energy"] = f"{_safe_float(best_proto.get('max_energy')):.9f}" if best_proto else ""
        out["best_proto_mean_rise"] = f"{_safe_float(best_proto.get('mean_rise')):.9f}" if best_proto else ""
        out["best_proto_mean_continuation"] = f"{_safe_float(best_proto.get('mean_continuation')):.9f}" if best_proto else ""
        out["best_proto_seed_count"] = str(_safe_int(best_proto.get("seed_count"))) if best_proto else ""
        out["best_chain_mode"] = str(best_chain.get("chain_mode", "")).strip() if best_chain else ""
        out["best_chain_mean_score"] = f"{_safe_float(best_chain.get('mean_chain_score')):.9f}" if best_chain else ""
        out["best_chain_bridge_resistance"] = f"{_safe_float(best_chain.get('bridge_resistance')):.9f}" if best_chain else ""
        out["best_chain_exact_coarse_frames"] = str(_safe_int(best_chain.get("exact_coarse_frames"))) if best_chain else ""
        out["best_chain_pitchclass_frames"] = str(_safe_int(best_chain.get("pitchclass_frames"))) if best_chain else ""
        out["best_chain_same_octave_frames"] = str(_safe_int(best_chain.get("same_octave_frames"))) if best_chain else ""
        out["best_field_mean_anchor_match_ratio"] = f"{_safe_float(best_entity.get('mean_anchor_match_ratio')):.9f}" if best_entity else ""
        out["best_field_mean_strength"] = f"{_safe_float(best_entity.get('mean_field_strength')):.9f}" if best_entity else ""
        out["best_field_max_diversity"] = f"{_safe_float(best_entity.get('max_field_diversity')):.9f}" if best_entity else ""
        out["best_field_phase_counts_json"] = str(best_entity.get("phase_counts_json", "")).strip() if best_entity else ""
        out["best_sustain_exact_anchor_frames"] = str(_safe_int(best_sustain.get("exact_anchor_frames"))) if best_sustain else ""
        out["best_sustain_added_rows"] = str(_safe_int(best_sustain.get("added_rows"))) if best_sustain else ""
        out["best_sustain_transfer_rows"] = str(_safe_int(best_sustain.get("transfer_rows"))) if best_sustain else ""
        out["field_frame_mean_anchor_match_ratio"] = f"{(sum(anchor_match_values) / len(anchor_match_values)):.9f}" if anchor_match_values else ""
        out["field_frame_mean_strength"] = f"{(sum(field_strength_values) / len(field_strength_values)):.9f}" if field_strength_values else ""
        out["field_frame_max_diversity"] = f"{max(field_diversity_values):.9f}" if field_diversity_values else ""
        out["field_frame_max_dominant_score"] = f"{max(dominant_scores):.9f}" if dominant_scores else ""
        out["concurrent_neighbor_event_ids_json"] = json.dumps(neighbor_index.get(event_id, []), ensure_ascii=False)
        out["parent_event_id"] = str(parent_event_id) if parent_event_id else ""
        out["parent_relation_kind"] = relation_kind
        out["child_event_ids_json"] = json.dumps(child_ids, ensure_ascii=False)
        out["child_count"] = str(len(child_ids))
        out_rows.append(out)

    fieldnames = list(event_rows[0].keys()) + [
        "raw_attached_proto_ids_json",
        "chain_attached_proto_ids_json",
        "field_attached_proto_ids_json",
        "sustain_attached_proto_ids_json",
        "best_proto_peak_hz",
        "best_proto_peak_seed_score",
        "best_proto_max_energy",
        "best_proto_mean_rise",
        "best_proto_mean_continuation",
        "best_proto_seed_count",
        "best_chain_mode",
        "best_chain_mean_score",
        "best_chain_bridge_resistance",
        "best_chain_exact_coarse_frames",
        "best_chain_pitchclass_frames",
        "best_chain_same_octave_frames",
        "best_field_mean_anchor_match_ratio",
        "best_field_mean_strength",
        "best_field_max_diversity",
        "best_field_phase_counts_json",
        "best_sustain_exact_anchor_frames",
        "best_sustain_added_rows",
        "best_sustain_transfer_rows",
        "field_frame_mean_anchor_match_ratio",
        "field_frame_mean_strength",
        "field_frame_max_diversity",
        "field_frame_max_dominant_score",
        "concurrent_neighbor_event_ids_json",
        "parent_event_id",
        "parent_relation_kind",
        "child_event_ids_json",
        "child_count",
    ]
    with Path(args.out_csv).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    summary_lines = [
        "EVENT FULL CYCLE CENSUS",
        "=" * 72,
        f"input_events: {len(event_rows)}",
        f"max_parallel_checks: {int(args.max_parallel_checks)}",
        f"events_with_proto_match: {matched_proto_count}",
        f"events_with_chain_match: {matched_chain_count}",
        f"events_with_field_match: {matched_entity_count}",
        f"events_with_sustain_match: {matched_sustain_count}",
        "",
        "event_type_counts:",
    ]
    for key, value in event_type_counter.most_common():
        summary_lines.append(f"  {key}: {value}")
    summary_lines.extend(["", "parent_relation_counts:"])
    for key, value in relation_counter.most_common():
        summary_lines.append(f"  {key}: {value}")
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "input_events": len(event_rows),
                "max_parallel_checks": int(args.max_parallel_checks),
                "events_with_proto_match": matched_proto_count,
                "events_with_chain_match": matched_chain_count,
                "events_with_field_match": matched_entity_count,
                "events_with_sustain_match": matched_sustain_count,
                "event_type_counts": dict(event_type_counter),
                "parent_relation_counts": dict(relation_counter),
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
