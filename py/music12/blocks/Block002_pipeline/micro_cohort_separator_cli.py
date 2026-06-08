# -*- coding: ascii -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_progress(path_str: str, payload: dict[str, Any]) -> None:
    if not str(path_str).strip():
        return
    Path(path_str).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _normalize_note(note: str) -> str:
    s = str(note or "").strip()
    if not s:
        return ""
    if "'" in s:
        return s.split("'", 1)[0] + "'-"
    if s.endswith("-"):
        return s[:-1] + "'-"
    return s + "'-"


def _pitchclass(note: str) -> str:
    token = _normalize_note(note)
    if "." not in token:
        return ""
    return token.split(".", 1)[1].split("'", 1)[0].strip()


def _octave(note: str) -> str:
    token = _normalize_note(note)
    if "." not in token:
        return ""
    return token.split(".", 1)[0].strip()


def _token_to_abs_degree(token: str) -> int | None:
    alphabet = "123456789ABC"
    try:
        token = str(token).strip().upper()
        octave_raw, rest = token.split(".", 1)
        degree_raw = rest.split("'", 1)[0]
        octave_value = 0
        for ch in octave_raw:
            if ch not in alphabet:
                return None
            octave_value = octave_value * 12 + (alphabet.index(ch) + 1)
        if degree_raw not in alphabet:
            return None
        return octave_value * 12 + alphabet.index(degree_raw)
    except Exception:
        return None


def _is_octave_related(left: str, right: str) -> bool:
    la = _token_to_abs_degree(left)
    rb = _token_to_abs_degree(right)
    if la is None or rb is None:
        return False
    return abs(la - rb) == 12


def _parse_pairs(value: str) -> list[list[Any]]:
    try:
        loaded = json.loads(str(value or "").strip() or "[]")
        return loaded if isinstance(loaded, list) else []
    except Exception:
        return []


def _top_probe_keys(value: str) -> set[str]:
    out: set[str] = set()
    for item in _parse_pairs(value):
        if isinstance(item, list) and item:
            out.add(str(item[0]))
    return out


def _probe_overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Separate simultaneous micro cohorts: identify overlapping sibling chain-groups that coexist in time and should not be collapsed into one life."
    )
    ap.add_argument("--coalesced-chains-csv", required=True)
    ap.add_argument("--out-cohorts-csv", required=True)
    ap.add_argument("--out-cohort-members-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--progress-json", default="")
    ap.add_argument("--max-start-delta-frames", type=int, default=4)
    ap.add_argument("--min-overlap-frames", type=int, default=4)
    ap.add_argument("--min-probe-overlap-ratio", type=float, default=0.12)
    args = ap.parse_args()

    _write_progress(
        args.progress_json,
        {
            "status": "running",
            "phase": "loading_inputs",
            "processed_groups": 0,
            "total_groups": 0,
            "cohort_count": 0,
        },
    )

    rows = _load_csv(Path(args.coalesced_chains_csv))
    rows.sort(key=lambda row: (_safe_int(row.get("start_frame"), 0), _safe_int(row.get("end_frame"), 0), _safe_int(row.get("coalesced_group_id"), 0)))
    total = len(rows)

    adjacency: dict[int, set[int]] = defaultdict(set)
    row_by_id: dict[int, dict[str, Any]] = {_safe_int(row.get("coalesced_group_id"), 0): row for row in rows}

    _write_progress(
        args.progress_json,
        {
            "status": "running",
            "phase": "building_adjacency",
            "processed_groups": 0,
            "total_groups": total,
            "cohort_count": 0,
        },
    )

    for idx, left in enumerate(rows):
        left_id = _safe_int(left.get("coalesced_group_id"), 0)
        left_start = _safe_int(left.get("start_frame"), 0)
        left_end = _safe_int(left.get("end_frame"), 0)
        left_coarse = str(left.get("anchor_coarse_note", "")).strip()
        left_micro = str(left.get("anchor_micro_note_token", "")).strip()
        left_pc = _pitchclass(left_micro)
        left_probes = _top_probe_keys(left.get("dominant_probes_json", ""))

        for right in rows[idx + 1:]:
            right_start = _safe_int(right.get("start_frame"), 0)
            if right_start - left_start > int(args.max_start_delta_frames) and right_start > left_end:
                break

            right_id = _safe_int(right.get("coalesced_group_id"), 0)
            right_end = _safe_int(right.get("end_frame"), 0)
            overlap_frames = min(left_end, right_end) - max(left_start, right_start) + 1
            if overlap_frames < int(args.min_overlap_frames):
                continue

            right_coarse = str(right.get("anchor_coarse_note", "")).strip()
            right_micro = str(right.get("anchor_micro_note_token", "")).strip()
            right_pc = _pitchclass(right_micro)
            if not left_coarse or not right_coarse:
                continue

            same_coarse = left_coarse == right_coarse
            same_pitchclass = bool(left_pc and right_pc and left_pc == right_pc)
            octave_related = _is_octave_related(left_micro, right_micro) or _is_octave_related(left_coarse, right_coarse)
            probe_overlap = _probe_overlap_ratio(left_probes, _top_probe_keys(right.get("dominant_probes_json", "")))

            if same_coarse or octave_related or (same_pitchclass and probe_overlap >= float(args.min_probe_overlap_ratio)):
                adjacency[left_id].add(right_id)
                adjacency[right_id].add(left_id)

        if (idx + 1) % 512 == 0 or (idx + 1) == total:
            _write_progress(
                args.progress_json,
                {
                    "status": "running",
                    "phase": "building_adjacency",
                    "processed_groups": idx + 1,
                    "total_groups": total,
                    "cohort_count": 0,
                },
            )

    _write_progress(
        args.progress_json,
        {
            "status": "running",
            "phase": "extracting_cohorts",
            "processed_groups": total,
            "total_groups": total,
            "cohort_count": 0,
        },
    )

    seen: set[int] = set()
    cohort_rows: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    cohort_kind_counter: Counter[str] = Counter()
    next_cohort_id = 1

    for group_id in row_by_id:
        if group_id in seen:
            continue
        stack = [group_id]
        component: list[int] = []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.append(current)
            for neighbor in adjacency.get(current, set()):
                if neighbor not in seen:
                    stack.append(neighbor)

        component_rows = [row_by_id[cid] for cid in sorted(component)]
        coarse_counter: Counter[str] = Counter()
        status_counter: Counter[str] = Counter()
        start_frame = min(_safe_int(row.get("start_frame"), 0) for row in component_rows)
        end_frame = max(_safe_int(row.get("end_frame"), 0) for row in component_rows)
        max_overlap_size = len(component_rows)
        octave_relations = 0
        probe_overlap_values: list[float] = []
        micro_tokens: list[str] = []
        for i, row in enumerate(component_rows):
            micro = str(row.get("anchor_micro_note_token", "")).strip()
            micro_tokens.append(micro)
            coarse_counter[str(row.get("anchor_coarse_note", "")).strip()] += 1
            status_counter[str(row.get("coalesced_status", "")).strip()] += 1
            left_probes = _top_probe_keys(row.get("dominant_probes_json", ""))
            for other in component_rows[i + 1:]:
                other_micro = str(other.get("anchor_micro_note_token", "")).strip()
                if _is_octave_related(micro, other_micro):
                    octave_relations += 1
                probe_overlap_values.append(_probe_overlap_ratio(left_probes, _top_probe_keys(other.get("dominant_probes_json", ""))))

        if len(component_rows) == 1:
            cohort_kind = "SINGLETON_COHORT"
        elif octave_relations > 0:
            cohort_kind = "OCTAVE_SIMULTANEOUS_COHORT"
        elif len(coarse_counter) == 1:
            cohort_kind = "SAME_COARSE_SIMULTANEOUS_COHORT"
        else:
            cohort_kind = "SHARED_FIELD_SIMULTANEOUS_COHORT"
        cohort_kind_counter[cohort_kind] += 1

        cohort_id = next_cohort_id
        next_cohort_id += 1

        cohort_rows.append(
            {
                "cohort_id": cohort_id,
                "source_mode": "MICRO_SIMULTANEOUS_COHORT",
                "start_frame": start_frame,
                "end_frame": end_frame,
                "duration_frames": end_frame - start_frame + 1,
                "member_count": len(component_rows),
                "cohort_kind": cohort_kind,
                "dominant_coarse_notes_json": json.dumps(coarse_counter.most_common(8), ensure_ascii=False),
                "member_status_counts_json": json.dumps(dict(status_counter), ensure_ascii=False),
                "mean_probe_overlap_ratio": f"{(sum(probe_overlap_values) / len(probe_overlap_values)) if probe_overlap_values else 0.0:.9f}",
                "octave_relation_count": octave_relations,
                "member_group_ids_json": json.dumps(component, ensure_ascii=False),
                "member_micro_tokens_json": json.dumps(micro_tokens, ensure_ascii=False),
            }
        )

        for row in component_rows:
            member_rows.append(
                {
                    "cohort_id": cohort_id,
                    "coalesced_group_id": _safe_int(row.get("coalesced_group_id"), 0),
                    "anchor_micro_note_token": str(row.get("anchor_micro_note_token", "")).strip(),
                    "anchor_coarse_note": str(row.get("anchor_coarse_note", "")).strip(),
                    "start_frame": _safe_int(row.get("start_frame"), 0),
                    "end_frame": _safe_int(row.get("end_frame"), 0),
                    "coalesced_structure_class": str(row.get("coalesced_structure_class", "")).strip(),
                    "coalesced_status": str(row.get("coalesced_status", "")).strip(),
                    "observation_frame_count": _safe_int(row.get("observation_frame_count"), 0),
                    "chain_count": _safe_int(row.get("chain_count"), 0),
                }
            )

    cohort_fields = [
        "cohort_id",
        "source_mode",
        "start_frame",
        "end_frame",
        "duration_frames",
        "member_count",
        "cohort_kind",
        "dominant_coarse_notes_json",
        "member_status_counts_json",
        "mean_probe_overlap_ratio",
        "octave_relation_count",
        "member_group_ids_json",
        "member_micro_tokens_json",
    ]
    member_fields = [
        "cohort_id",
        "coalesced_group_id",
        "anchor_micro_note_token",
        "anchor_coarse_note",
        "start_frame",
        "end_frame",
        "coalesced_structure_class",
        "coalesced_status",
        "observation_frame_count",
        "chain_count",
    ]

    out_cohorts = Path(args.out_cohorts_csv)
    out_cohorts.parent.mkdir(parents=True, exist_ok=True)
    with out_cohorts.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cohort_fields)
        writer.writeheader()
        for row in cohort_rows:
            writer.writerow({key: row.get(key, "") for key in cohort_fields})

    out_members = Path(args.out_cohort_members_csv)
    with out_members.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=member_fields)
        writer.writeheader()
        for row in member_rows:
            writer.writerow({key: row.get(key, "") for key in member_fields})

    summary_lines = [
        "MICRO COHORT SEPARATOR",
        "=" * 72,
        "source_mode               : MICRO_SIMULTANEOUS_COHORT",
        f"input_group_rows          : {len(rows)}",
        f"cohort_count              : {len(cohort_rows)}",
        f"cohort_member_rows        : {len(member_rows)}",
        f"max_start_delta_frames    : {int(args.max_start_delta_frames)}",
        f"min_overlap_frames        : {int(args.min_overlap_frames)}",
        f"min_probe_overlap_ratio   : {float(args.min_probe_overlap_ratio):.3f}",
        "",
        "cohort_kind_counts:",
    ]
    for key, value in cohort_kind_counter.most_common():
        summary_lines.append(f"  {key}: {value}")
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "stage": "micro_cohort_separator",
                "source_mode": "MICRO_SIMULTANEOUS_COHORT",
                "inputs": {
                    "coalesced_chains_csv": args.coalesced_chains_csv,
                },
                "parameters": {
                    "max_start_delta_frames": int(args.max_start_delta_frames),
                    "min_overlap_frames": int(args.min_overlap_frames),
                    "min_probe_overlap_ratio": float(args.min_probe_overlap_ratio),
                },
                "result": {
                    "input_group_rows": len(rows),
                    "cohort_count": len(cohort_rows),
                    "cohort_member_rows": len(member_rows),
                    "cohort_kind_counts": dict(cohort_kind_counter),
                },
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    _write_progress(
        args.progress_json,
        {
            "status": "done",
            "phase": "complete",
            "processed_groups": total,
            "total_groups": total,
            "cohort_count": len(cohort_rows),
        },
    )


if __name__ == "__main__":
    main()
