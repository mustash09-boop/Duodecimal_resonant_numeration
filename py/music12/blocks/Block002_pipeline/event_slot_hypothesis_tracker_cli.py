# -*- coding: ascii -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
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


def _index_rows(path: Path, key: str) -> dict[int, dict[str, str]]:
    return {_safe_int(row.get(key)): row for row in _load_csv(path)}


def _split_set(raw: str | None) -> set[str]:
    raw = str(raw or "").strip()
    if not raw:
        return set()
    return {part.strip() for part in raw.split() if part.strip()}


@dataclass
class SlotState:
    slot_index: int
    current_end_frame: int = -1


def _initial_hypothesis(
    event_row: dict[str, str],
    role_row: dict[str, str],
    layered_row: dict[str, str],
    shared_row: dict[str, str] | None,
) -> tuple[str, str, str, str]:
    candidate_note = str(event_row.get("candidate_note", "")).strip()
    same_note_ratio = _safe_float(event_row.get("same_note_ratio"))
    birth_count = _safe_int(event_row.get("birth_count"))
    re_excitation_count = _safe_int(event_row.get("re_excitation_count"))
    active_body_count = _safe_int(event_row.get("active_body_count"))
    response_trace_count = _safe_int(event_row.get("response_trace_count"))
    decay_trace_count = _safe_int(event_row.get("decay_trace_count"))

    dominant = str(role_row.get("dominant_instrument", "")).strip()
    attack_owner = str(role_row.get("attack_owner", "")).strip()
    sustain_owner = str(role_row.get("sustain_owner", "")).strip()
    body_owner = str(role_row.get("body_owner", "")).strip()
    field_owner = str(role_row.get("field_owner", "")).strip()
    role_pattern = str(role_row.get("role_pattern", "")).strip()
    acoustic_cause = str(role_row.get("acoustic_cause_class", "")).strip()
    residual_class = str(role_row.get("residual_fragmentation_class", "")).strip()

    support_layered = _split_set(layered_row.get("support_instruments"))
    support_role = _split_set(role_row.get("support_owners"))
    shared_mode = str((shared_row or {}).get("shared_mode", "")).strip()
    ownership_mode = str((shared_row or {}).get("ownership_mode", "")).strip()

    harmonic_guess = "UNRESOLVED_HARMONIC_STRUCTURE"
    resonance_guess = "UNRESOLVED_RESONANCE_STRUCTURE"
    instrument_hint = "CHECK_ALL_PASSPORTS_LATER"
    event_type = "MIXED_EVENT"

    if same_note_ratio >= 0.95 and not response_trace_count and not decay_trace_count:
        harmonic_guess = "STABLE_SINGLE_NOTE_CHAIN"
    elif same_note_ratio >= 0.75:
        harmonic_guess = "MOSTLY_STABLE_NOTE_CHAIN"
    elif candidate_note:
        harmonic_guess = "MIGRATING_OR_SHARED_NOTE_IDENTITY"

    if role_pattern == "PIANO_ATTACK_EVENT" or attack_owner == "piano":
        event_type = "DISCRETE_PIANO_EXCITATION"
        instrument_hint = "COMPARE_ALL_PASSPORTS_ATTACK_FIRST"
        resonance_guess = "PIANO_ATTACK_PLUS_LOCAL_BODY"
    elif acoustic_cause == "LIKELY_TRUE_REEXCITATION":
        event_type = "TRUE_REEXCITATION_EVENT"
        resonance_guess = "LOCAL_REEXCITATION_WITH_SHORT_BODY"
    elif acoustic_cause == "LIKELY_INSTRUMENT_BODY_RETURN" or body_owner:
        event_type = "BODY_RETURN_EVENT"
        resonance_guess = "BODY_CONTINUATION_OR_RETURN"
    elif "HALL" in acoustic_cause or field_owner:
        event_type = "FIELD_TRACE_EVENT"
        resonance_guess = "FIELD_OR_HALL_CONTINUATION"

    if sustain_owner in {"violin", "cello"} or body_owner in {"violin", "cello"}:
        instrument_hint = "COMPARE_ALL_PASSPORTS_SUSTAIN_BODY_FIRST"
        resonance_guess = "SUSTAINED_STRING_LIKE_RESONANCE"
    elif sustain_owner == "piano":
        instrument_hint = "COMPARE_ALL_PASSPORTS_SUSTAIN_BODY_FIRST"
    elif dominant == "organ":
        instrument_hint = "COMPARE_ALL_PASSPORTS_AND_DEFER_FIELD_SPECIFICS"

    if "cello" in support_layered or "violin" in support_layered or "cello" in support_role or "violin" in support_role:
        resonance_guess = "SHARED_STRING_SUPPORT_OR_OVERLAP"
    if shared_mode or ownership_mode:
        resonance_guess = "EXPLICIT_SHARED_OWNERSHIP_STRUCTURE"

    if active_body_count >= max(3, birth_count + re_excitation_count):
        resonance_guess = "LONG_BODY_CONTINUATION"
    if response_trace_count or decay_trace_count:
        resonance_guess = "TAIL_OR_RESPONSE_CONTINUATION"
    if residual_class == "VERY_SHORT_EVENT":
        event_type = "VERY_SHORT_EVENT"

    return event_type, harmonic_guess, resonance_guess, instrument_hint


def _late_refinement_target(
    role_row: dict[str, str],
    layered_row: dict[str, str],
) -> str:
    attack_owner = str(role_row.get("attack_owner", "")).strip()
    sustain_owner = str(role_row.get("sustain_owner", "")).strip()
    body_owner = str(role_row.get("body_owner", "")).strip()
    dominant = str(layered_row.get("dominant_instrument", "")).strip()
    support_layered = _split_set(layered_row.get("support_instruments"))

    if attack_owner == "piano" or sustain_owner == "piano" or dominant == "piano":
        if "cello" in support_layered or "violin" in support_layered:
            return "ALL_PASSPORTS_BACKBONE_THEN_SHARED_REFINEMENT"
        return "ALL_PASSPORTS_KEYBOARD_LIKE_REFINEMENT"
    if sustain_owner == "violin" or body_owner == "violin":
        return "ALL_PASSPORTS_UPPER_SUSTAIN_REFINEMENT"
    if sustain_owner == "cello" or body_owner == "cello":
        return "ALL_PASSPORTS_LOWER_SUSTAIN_REFINEMENT"
    if dominant == "organ":
        return "ALL_PASSPORTS_FIELD_OR_SHARED_CHECK"
    return "ALL_PASSPORTS_GENERAL_REFINEMENT"


def _assign_slots(event_rows: list[dict[str, str]], max_slots: int) -> tuple[list[dict[str, str]], int]:
    slots = [SlotState(slot_index=i + 1) for i in range(max_slots)]
    augmented: list[dict[str, str]] = []
    max_concurrency = 0

    for idx, row in enumerate(event_rows):
        start = _safe_int(row.get("birth_frame"))
        end = _safe_int(row.get("end_frame"))
        overlap = 0
        for prev in augmented:
            prev_start = _safe_int(prev.get("birth_frame"))
            prev_end = _safe_int(prev.get("end_frame"))
            if not (prev_end < start or end < prev_start):
                overlap += 1
        concurrency = overlap + 1
        max_concurrency = max(max_concurrency, concurrency)

        chosen_slot = None
        for slot in slots:
            if slot.current_end_frame < start:
                chosen_slot = slot
                break
        if chosen_slot is None:
            chosen_slot = min(slots, key=lambda s: s.current_end_frame)
        chosen_slot.current_end_frame = max(chosen_slot.current_end_frame, end)

        out = dict(row)
        out["event_slot_index"] = str(chosen_slot.slot_index)
        out["active_overlap_count_at_birth"] = str(overlap)
        out["concurrency_count_at_birth"] = str(concurrency)
        out["slot_assignment_mode"] = "FREE_SLOT" if overlap < max_slots else "REUSED_SLOT_UNDER_PRESSURE"
        augmented.append(out)
    return augmented, max_concurrency


def main() -> None:
    ap = argparse.ArgumentParser(description="Track up to N parallel event slots and keep early harmonic/resonance hypotheses before late passport refinement.")
    ap.add_argument("--events_csv", required=True)
    ap.add_argument("--role_map_csv", required=True)
    ap.add_argument("--layered_csv", required=True)
    ap.add_argument("--shared_guard_csv", default="")
    ap.add_argument("--max_parallel_events", type=int, default=8)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_summary_txt", required=True)
    ap.add_argument("--out_meta_json", required=True)
    args = ap.parse_args()

    event_rows = _load_csv(Path(args.events_csv))
    role_index = _index_rows(Path(args.role_map_csv), "merged_event_id")
    layered_index = _index_rows(Path(args.layered_csv), "merged_event_id")
    shared_index = _index_rows(Path(args.shared_guard_csv), "merged_event_id") if str(args.shared_guard_csv).strip() else {}

    event_rows = sorted(event_rows, key=lambda r: (_safe_int(r.get("birth_frame")), _safe_int(r.get("end_frame")), _safe_int(r.get("merged_event_id"))))
    slotted_rows, max_concurrency = _assign_slots(event_rows, max_slots=int(args.max_parallel_events))

    out_rows: list[dict[str, str]] = []
    event_type_counter: Counter[str] = Counter()
    harmonic_counter: Counter[str] = Counter()
    resonance_counter: Counter[str] = Counter()
    refinement_counter: Counter[str] = Counter()
    slot_counter: Counter[str] = Counter()

    for row in slotted_rows:
        event_id = _safe_int(row.get("merged_event_id"))
        role_row = role_index.get(event_id, {})
        layered_row = layered_index.get(event_id, {})
        shared_row = shared_index.get(event_id)
        event_type, harmonic_guess, resonance_guess, instrument_hint = _initial_hypothesis(row, role_row, layered_row, shared_row)
        refinement_target = _late_refinement_target(role_row, layered_row)

        out = dict(row)
        out["initial_event_type"] = event_type
        out["initial_harmonic_hypothesis"] = harmonic_guess
        out["initial_resonance_hypothesis"] = resonance_guess
        out["initial_instrument_hint"] = instrument_hint
        out["late_refinement_target"] = refinement_target
        out["attack_owner"] = str(role_row.get("attack_owner", "")).strip()
        out["sustain_owner"] = str(role_row.get("sustain_owner", "")).strip()
        out["body_owner"] = str(role_row.get("body_owner", "")).strip()
        out["field_owner"] = str(role_row.get("field_owner", "")).strip()
        out["role_pattern"] = str(role_row.get("role_pattern", "")).strip()
        out["acoustic_cause_class"] = str(role_row.get("acoustic_cause_class", "")).strip()
        out["shared_mode"] = str((shared_row or {}).get("shared_mode", "")).strip()
        out["ownership_mode"] = str((shared_row or {}).get("ownership_mode", "")).strip()
        out["dominant_instrument"] = str(layered_row.get("dominant_instrument", "")).strip()
        out["support_instruments"] = str(layered_row.get("support_instruments", "")).strip()

        out_rows.append(out)
        event_type_counter[event_type] += 1
        harmonic_counter[harmonic_guess] += 1
        resonance_counter[resonance_guess] += 1
        refinement_counter[refinement_target] += 1
        slot_counter[out["event_slot_index"]] += 1

    fieldnames = [
        "merged_event_id",
        "event_slot_index",
        "slot_assignment_mode",
        "active_overlap_count_at_birth",
        "concurrency_count_at_birth",
        "candidate_note",
        "birth_frame",
        "end_frame",
        "duration_frames",
        "frame_count",
        "mean_score",
        "max_score",
        "birth_score",
        "final_score",
        "same_note_ratio",
        "re_excitation_count",
        "active_body_count",
        "response_trace_count",
        "decay_trace_count",
        "lifecycle_kind",
        "initial_event_type",
        "initial_harmonic_hypothesis",
        "initial_resonance_hypothesis",
        "initial_instrument_hint",
        "late_refinement_target",
        "dominant_instrument",
        "support_instruments",
        "attack_owner",
        "sustain_owner",
        "body_owner",
        "field_owner",
        "role_pattern",
        "acoustic_cause_class",
        "shared_mode",
        "ownership_mode",
    ]
    with Path(args.out_csv).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    summary_lines = [
        "EVENT SLOT HYPOTHESIS TRACKER",
        "=" * 72,
        f"input_events: {len(event_rows)}",
        f"max_parallel_events_limit: {int(args.max_parallel_events)}",
        f"observed_max_concurrency: {max_concurrency}",
        "",
        "event_type_counts:",
    ]
    for key, value in event_type_counter.most_common():
        summary_lines.append(f"  {key}: {value}")
    summary_lines.extend(["", "harmonic_hypothesis_counts:"])
    for key, value in harmonic_counter.most_common():
        summary_lines.append(f"  {key}: {value}")
    summary_lines.extend(["", "resonance_hypothesis_counts:"])
    for key, value in resonance_counter.most_common():
        summary_lines.append(f"  {key}: {value}")
    summary_lines.extend(["", "late_refinement_targets:"])
    for key, value in refinement_counter.most_common():
        summary_lines.append(f"  {key}: {value}")
    summary_lines.extend(["", "slot_usage_counts:"])
    for key, value in sorted(slot_counter.items(), key=lambda kv: int(kv[0])):
        summary_lines.append(f"  slot_{key}: {value}")
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "input_events": len(event_rows),
                "max_parallel_events_limit": int(args.max_parallel_events),
                "observed_max_concurrency": max_concurrency,
                "event_type_counts": dict(event_type_counter),
                "harmonic_hypothesis_counts": dict(harmonic_counter),
                "resonance_hypothesis_counts": dict(resonance_counter),
                "late_refinement_targets": dict(refinement_counter),
                "slot_usage_counts": dict(slot_counter),
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
