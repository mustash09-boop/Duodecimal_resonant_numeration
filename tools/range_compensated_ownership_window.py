from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ALPHABET12 = "0123456789AB"


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


def _octave_value(note: str) -> int | None:
    try:
        octave_raw = _normalize_note(note).split(".", 1)[0]
    except Exception:
        return None
    if not octave_raw:
        return None
    value = 0
    try:
        for ch in octave_raw:
            value = value * 12 + ALPHABET12.index(ch)
        return value
    except Exception:
        return None


def _range_mode(note: str, profile: str) -> str:
    octave = _octave_value(note)
    if octave is None:
        return "unknown"
    if profile == "bach_working_edges":
        if octave <= 7:
            return "low"
        if octave == 8:
            return "low_edge"
        if octave == 9:
            return "core_mid"
        if octave == 10:
            return "high_edge"
        return "high"

    if octave <= 7:
        return "low"
    if octave >= 11:
        return "high"
    return "mid"


def _json_dict(value: Any) -> dict[str, int]:
    try:
        raw = json.loads(str(value or "{}"))
        if isinstance(raw, dict):
            out: dict[str, int] = {}
            for k, v in raw.items():
                out[str(k).strip()] = _safe_int(v, 0)
            return out
    except Exception:
        return {}
    return {}


def _candidate_compensation(
    note: str,
    role_counts: dict[str, int],
    source_counts: dict[str, int],
) -> tuple[float, str, str]:
    raise RuntimeError("profile-aware wrapper must be used")


def _candidate_compensation_with_profile(
    note: str,
    role_counts: dict[str, int],
    source_counts: dict[str, int],
    profile: str,
) -> tuple[float, str, str]:
    mode = _range_mode(note, profile)
    if mode == "unknown":
        return 0.0, "unknown_register", mode

    birth_backbone = role_counts.get("birth_backbone", 0)
    local_birth = role_counts.get("local_birth", 0)
    exact_support = role_counts.get("exact_support", 0)
    pitchclass_support = role_counts.get("pitchclass_support", 0)
    field_birth = role_counts.get("field_birth", 0)
    previous_tail = role_counts.get("previous_tail", 0)
    future_birth = role_counts.get("future_birth", 0)
    ambient_residue = role_counts.get("ambient_residue", 0)

    birth_core = birth_backbone + local_birth + exact_support
    support_ladder = exact_support + pitchclass_support
    tail_load = previous_tail + ambient_residue
    role_diversity = sum(1 for v in role_counts.values() if v > 0)
    source_diversity = sum(1 for v in source_counts.values() if v > 0)

    if mode == "low":
        bonus = (
            min(birth_core, 3) * 0.14
            + min(support_ladder, 3) * 0.09
            + min(role_diversity, 4) * 0.03
            + min(source_diversity, 3) * 0.02
            + min(field_birth, 2) * 0.02
            - min(tail_load, 3) * 0.05
            - min(future_birth, 2) * 0.02
        )
        return bonus, "low_register_compensation", mode

    if mode == "low_edge":
        bonus = (
            min(birth_core, 3) * 0.12
            + min(support_ladder, 3) * 0.08
            + min(role_diversity, 4) * 0.03
            + min(source_diversity, 3) * 0.02
            + min(field_birth, 2) * 0.03
            - min(tail_load, 3) * 0.045
            - min(future_birth, 2) * 0.02
        )
        return bonus, "low_edge_compensation", mode

    if mode == "core_mid":
        bonus = (
            min(birth_core, 3) * 0.10
            + min(support_ladder, 3) * 0.08
            + min(role_diversity, 4) * 0.02
            + min(source_diversity, 3) * 0.015
            - min(tail_load, 3) * 0.035
            - min(future_birth, 2) * 0.02
        )
        return bonus, "core_mid_compensation", mode

    if mode == "high_edge":
        bonus = (
            min(local_birth + exact_support, 3) * 0.12
            + min(support_ladder, 3) * 0.14
            + min(role_diversity, 4) * 0.025
            + min(source_diversity, 3) * 0.025
            + min(field_birth, 2) * 0.04
            - min(previous_tail, 2) * 0.025
            - min(ambient_residue, 2) * 0.025
            - min(future_birth, 2) * 0.02
        )
        return bonus, "high_edge_compensation", mode

    if mode == "high":
        bonus = (
            min(local_birth + exact_support, 3) * 0.11
            + min(support_ladder, 3) * 0.13
            + min(role_diversity, 4) * 0.02
            + min(source_diversity, 3) * 0.03
            + min(field_birth, 2) * 0.04
            - min(previous_tail, 2) * 0.03
            - min(ambient_residue, 2) * 0.03
            - min(future_birth, 2) * 0.02
        )
        return bonus, "high_register_compensation", mode

    bonus = (
        min(birth_core, 3) * 0.12
        + min(support_ladder, 3) * 0.10
        + min(role_diversity, 4) * 0.025
        + min(source_diversity, 3) * 0.02
        + min(field_birth, 2) * 0.02
        - min(tail_load, 3) * 0.04
        - min(future_birth, 2) * 0.02
    )
    return bonus, "mid_register_compensation", mode


def _classify_topk(note: str, candidates: list[str], k: int) -> str:
    norm_note = _normalize_note(note)
    note_pc = _pitch_class(norm_note)
    norm_candidates = [_normalize_note(x) for x in candidates[:k] if _normalize_note(x)]
    if norm_note in norm_candidates:
        return "EXACT"
    if any(_pitch_class(x) == note_pc for x in norm_candidates):
        return "PITCHCLASS"
    if norm_candidates:
        return "WRONG"
    return "EMPTY"


def _nearest_group(
    start_frame: int,
    group_rows: list[dict[str, Any]],
    window_frames: int,
) -> dict[str, Any] | None:
    nearby = [
        row
        for row in group_rows
        if abs(_safe_int(row.get("anchor_frame"), 0) - start_frame) <= window_frames
    ]
    nearby.sort(key=lambda r: abs(_safe_int(r.get("anchor_frame"), 0) - start_frame))
    return nearby[0] if nearby else None


def _coverage_summary(
    midi_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    window_frames: int,
    profile: str,
) -> dict[str, Any]:
    topk_counter: dict[str, Counter[str]] = {
        "top1": Counter(),
        "top3": Counter(),
        "top5": Counter(),
    }
    range_counter: dict[str, Counter[str]] = defaultdict(Counter)

    for midi in midi_rows:
        note = _normalize_note(midi.get("expected_note_token", midi.get("note_token", "")))
        start_frame = _safe_int(midi.get("start_frame60"), 0)
        mode = _range_mode(note, profile)
        group = _nearest_group(start_frame, group_rows, window_frames)
        candidates = []
        if group:
            try:
                candidates = json.loads(str(group.get("candidate_notes_json", "[]")))
                if not isinstance(candidates, list):
                    candidates = []
            except Exception:
                candidates = []
        for name, k in (("top1", 1), ("top3", 3), ("top5", 5)):
            status = _classify_topk(note, candidates, k)
            topk_counter[name][status] += 1
            if name == "top5":
                range_counter[mode][status] += 1

    return {
        "topk_counter": {k: dict(v) for k, v in topk_counter.items()},
        "range_counter_top5": {k: dict(v) for k, v in range_counter.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Apply low/high register compensation to the early ownership window without changing the note-recognition pipeline files."
    )
    ap.add_argument("--ownership-window-candidates-csv", required=True)
    ap.add_argument("--ownership-window-groups-csv", required=True)
    ap.add_argument("--midi-events-csv", required=True)
    ap.add_argument("--out-candidates-csv", required=True)
    ap.add_argument("--out-groups-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--match-window-frames", type=int, default=5)
    ap.add_argument(
        "--profile",
        choices=["classic", "bach_working_edges"],
        default="classic",
    )
    args = ap.parse_args()

    candidate_rows = _load_csv(Path(args.ownership_window_candidates_csv))
    group_rows = _load_csv(Path(args.ownership_window_groups_csv))
    midi_rows = _load_csv(Path(args.midi_events_csv))

    original_coverage = _coverage_summary(
        midi_rows,
        group_rows,
        int(args.match_window_frames),
        str(args.profile),
    )

    grouped_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        grouped_candidates[str(row.get("onset_group_id", "")).strip()].append(row)

    reranked_candidate_rows: list[dict[str, Any]] = []
    reranked_group_rows: list[dict[str, Any]] = []
    compensation_counter: Counter[str] = Counter()
    range_counter: Counter[str] = Counter()
    changed_top_counter = 0

    for group in group_rows:
        onset_group_id = str(group.get("onset_group_id", "")).strip()
        rows = grouped_candidates.get(onset_group_id, [])
        enriched: list[dict[str, Any]] = []
        for row in rows:
            note = _normalize_note(row.get("note_token", ""))
            role_counts = _json_dict(row.get("role_counts_json", "{}"))
            source_counts = _json_dict(row.get("source_counts_json", "{}"))
            base_weight = _safe_float(row.get("support_weight"), 0.0)
            bonus, basis, mode = _candidate_compensation_with_profile(
                note,
                role_counts,
                source_counts,
                str(args.profile),
            )
            compensation_counter[basis] += 1
            range_counter[mode] += 1

            new_row = dict(row)
            new_row["note_token"] = note
            new_row["base_support_weight"] = f"{base_weight:.9f}"
            new_row["compensation_bonus"] = f"{bonus:.9f}"
            new_row["adjusted_support_weight"] = f"{(base_weight + bonus):.9f}"
            new_row["range_mode"] = mode
            new_row["compensation_basis"] = basis
            enriched.append(new_row)

        original_top = _normalize_note(group.get("top_note_token", ""))
        enriched.sort(
            key=lambda r: (
                -_safe_float(r.get("adjusted_support_weight"), 0.0),
                -_safe_float(r.get("base_support_weight"), 0.0),
                r.get("note_token", ""),
            )
        )
        if enriched and _normalize_note(enriched[0].get("note_token", "")) != original_top:
            changed_top_counter += 1

        candidate_notes = []
        for rank, row in enumerate(enriched, start=1):
            row["candidate_rank"] = str(rank)
            candidate_notes.append(str(row.get("note_token", "")))
            reranked_candidate_rows.append(row)

        reranked_group = dict(group)
        reranked_group["candidate_count"] = str(len(enriched))
        reranked_group["candidate_notes_json"] = json.dumps(candidate_notes, ensure_ascii=False)
        reranked_group["top_note_token"] = candidate_notes[0] if candidate_notes else ""
        reranked_group["top_support_weight"] = enriched[0]["adjusted_support_weight"] if enriched else "0.000000000"
        reranked_group_rows.append(reranked_group)

    reranked_coverage = _coverage_summary(
        midi_rows,
        reranked_group_rows,
        int(args.match_window_frames),
        str(args.profile),
    )

    out_candidates = Path(args.out_candidates_csv)
    out_groups = Path(args.out_groups_csv)
    out_summary = Path(args.out_summary_txt)
    out_meta = Path(args.out_meta_json)
    out_candidates.parent.mkdir(parents=True, exist_ok=True)

    if reranked_candidate_rows:
        with out_candidates.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(reranked_candidate_rows[0].keys()))
            w.writeheader()
            w.writerows(reranked_candidate_rows)

    if reranked_group_rows:
        with out_groups.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(reranked_group_rows[0].keys()))
            w.writeheader()
            w.writerows(reranked_group_rows)

    lines = [
        "RANGE COMPENSATED OWNERSHIP WINDOW",
        "=" * 72,
        f"profile                        : {args.profile}",
        f"onset_group_count              : {len(reranked_group_rows)}",
        f"candidate_row_count            : {len(reranked_candidate_rows)}",
        f"changed_top_candidates         : {changed_top_counter}",
        "",
        "COMPENSATION BASIS COUNTS",
        "-" * 72,
    ]
    for key in sorted(compensation_counter):
        lines.append(f"{key:30s}: {compensation_counter[key]}")
    lines.extend(["", "RANGE MODE COUNTS", "-" * 72])
    for key in sorted(range_counter):
        lines.append(f"{key:30s}: {range_counter[key]}")

    lines.extend(["", "TOP-K BEFORE", "-" * 72])
    for key in ("top1", "top3", "top5"):
        lines.append(f"{key}")
        for status in ("EXACT", "PITCHCLASS", "WRONG", "EMPTY"):
            lines.append(
                f"  {status:28s}: {original_coverage['topk_counter'][key].get(status, 0)}"
            )

    lines.extend(["", "TOP-K AFTER", "-" * 72])
    for key in ("top1", "top3", "top5"):
        lines.append(f"{key}")
        for status in ("EXACT", "PITCHCLASS", "WRONG", "EMPTY"):
            lines.append(
                f"  {status:28s}: {reranked_coverage['topk_counter'][key].get(status, 0)}"
            )

    lines.extend(["", "TOP-5 RANGE COVERAGE AFTER", "-" * 72])
    for mode in sorted(reranked_coverage["range_counter_top5"]):
        lines.append(f"range_mode={mode}")
        for status in ("EXACT", "PITCHCLASS", "WRONG", "EMPTY"):
            lines.append(
                f"  {status:28s}: {reranked_coverage['range_counter_top5'][mode].get(status, 0)}"
            )

    out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    meta = {
        "stage": "range_compensated_ownership_window",
        "inputs": {
            "ownership_window_candidates_csv": args.ownership_window_candidates_csv,
            "ownership_window_groups_csv": args.ownership_window_groups_csv,
            "midi_events_csv": args.midi_events_csv,
        },
        "parameters": {
            "match_window_frames": int(args.match_window_frames),
            "profile": str(args.profile),
        },
        "result": {
            "changed_top_candidates": changed_top_counter,
            "compensation_counter": dict(compensation_counter),
            "range_counter": dict(range_counter),
            "original_coverage": original_coverage,
            "reranked_coverage": reranked_coverage,
        },
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
