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


def _parse_json_list(value: str) -> list[Any]:
    try:
        loaded = json.loads(str(value or "").strip() or "[]")
        return loaded if isinstance(loaded, list) else []
    except Exception:
        return []


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _overlap_count(left: set[Any], right: set[Any]) -> int:
    if not left or not right:
        return 0
    return len(left & right)


def _event_link_score(
    *,
    prev_event: dict[str, Any],
    event: dict[str, Any],
    max_gap_frames: int,
) -> tuple[float, str]:
    prev_start = _safe_int(prev_event.get("start_frame"), 0)
    prev_end = _safe_int(prev_event.get("end_frame"), 0)
    start = _safe_int(event.get("start_frame"), 0)
    end = _safe_int(event.get("end_frame"), 0)
    if start <= prev_start:
        return -1.0, "non_forward_time"
    gap = start - prev_end
    if gap < 0 or gap > max_gap_frames:
        return -1.0, "gap_out_of_range"

    event_note = str(event.get("primary_micro_note_token", "")).strip()
    prev_note = str(prev_event.get("primary_micro_note_token", "")).strip()
    event_coarse = str(event.get("primary_coarse_note", "")).strip()
    prev_coarse = str(prev_event.get("primary_coarse_note", "")).strip()
    event_pc = _pitchclass(event_note)
    prev_pc = _pitchclass(prev_note)
    relation = str(event.get("parent_relation_kind", "")).strip()
    event_parent = _safe_int(event.get("parent_event_id"), 0)
    prev_id = _safe_int(prev_event.get("event_id"), 0)

    score = 0.0
    reasons: list[str] = []

    if event_parent and event_parent == prev_id:
        score += 2.8
        reasons.append("parent_link")

    if event_note and prev_note and event_note == prev_note:
        score += 3.0
        reasons.append("exact_micro")
    elif event_coarse and prev_coarse and event_coarse == prev_coarse:
        score += 1.8
        reasons.append("same_coarse")
    elif event_pc and prev_pc and event_pc == prev_pc:
        score += 0.9
        reasons.append("same_pitchclass")

    if relation == "EXACT_MICRO_CONTINUATION":
        score += 2.2
        reasons.append("relation_exact")
    elif relation == "COARSE_OVERLAY_CONTINUATION":
        score += 1.2
        reasons.append("relation_coarse")
    elif relation == "NEAR_MICRO_DRIFT_CONTINUATION":
        score += 1.6
        reasons.append("relation_drift")
    elif relation == "PITCHCLASS_CONTINUATION":
        score += 0.8
        reasons.append("relation_pc")

    prev_freq = _safe_float(prev_event.get("mean_frequency_hz"), 0.0)
    event_freq = _safe_float(event.get("mean_frequency_hz"), 0.0)
    if prev_freq > 0.0 and event_freq > 0.0:
        rel_freq = abs(event_freq - prev_freq) / max(prev_freq, 1e-9)
        if rel_freq <= 0.0025:
            score += 1.2
            reasons.append("tight_freq")
        elif rel_freq <= 0.0060:
            score += 0.6
            reasons.append("near_freq")

    prev_probes = {int(x[0]) for x in _parse_json_list(prev_event.get("top_probes_json", "")) if isinstance(x, list) and x}
    event_probes = {int(x[0]) for x in _parse_json_list(event.get("top_probes_json", "")) if isinstance(x, list) and x}
    probe_overlap = _overlap_count(prev_probes, event_probes)
    if probe_overlap > 0:
        score += min(1.0, probe_overlap * 0.20)
        reasons.append(f"probe_overlap_{probe_overlap}")

    prev_suffixes = {str(x[0]) for x in _parse_json_list(prev_event.get("top_micro_suffixes_json", "")) if isinstance(x, list) and x}
    event_suffixes = {str(x[0]) for x in _parse_json_list(event.get("top_micro_suffixes_json", "")) if isinstance(x, list) and x}
    suffix_overlap = _overlap_count(prev_suffixes, event_suffixes)
    if suffix_overlap > 0:
        score += min(0.8, suffix_overlap * 0.15)
        reasons.append(f"suffix_overlap_{suffix_overlap}")

    score += max(0.0, 0.60 - 0.15 * gap)
    if _safe_int(event.get("frame_count"), 0) >= 4:
        score += 0.25
    if _safe_int(prev_event.get("frame_count"), 0) >= 4:
        score += 0.10

    return score, "|".join(reasons) if reasons else "weak"


def _classify_chain(
    *,
    exact_counter: Counter[str],
    coarse_counter: Counter[str],
    suffix_counter: Counter[str],
    event_count: int,
    frame_count: int,
    branching_max: int,
    resonance_counter: Counter[str],
) -> tuple[str, str]:
    top_exact_count = exact_counter.most_common(1)[0][1] if exact_counter else 0
    top_coarse_count = coarse_counter.most_common(1)[0][1] if coarse_counter else 0
    exact_share = top_exact_count / max(frame_count, 1)
    coarse_share = top_coarse_count / max(frame_count, 1)
    suffix_div = len(suffix_counter)

    if event_count >= 3 and exact_share >= 0.60 and suffix_div <= 4:
        chain_kind = "EXACT_MICRO_NOTECHAIN"
    elif event_count >= 3 and coarse_share >= 0.70 and suffix_div >= 3:
        chain_kind = "COARSE_ANCHORED_MICRO_DRIFT_CHAIN"
    elif resonance_counter.get("SHARED_MICRO_RESONANCE_FIELD", 0) >= max(2, event_count // 3):
        chain_kind = "SHARED_MICRO_FIELD_CHAIN"
    elif branching_max >= 4:
        chain_kind = "BRANCHED_MICRO_RESPONSE_CHAIN"
    else:
        chain_kind = "LOCAL_MICRO_CHAIN"

    if event_count >= 4 or frame_count >= 8:
        confirmation = "CONFIRMED_CHAIN"
    elif event_count >= 2 or frame_count >= 4:
        confirmation = "PROBABLE_CHAIN"
    else:
        confirmation = "WEAK_CHAIN"
    return chain_kind, confirmation


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build confirmed micro-note chains from micro-event observations without collapsing early to coarse_note."
    )
    ap.add_argument("--micro-events-csv", required=True)
    ap.add_argument("--micro-frame-slots-csv", required=True)
    ap.add_argument("--out-chains-csv", required=True)
    ap.add_argument("--out-chain-frames-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--progress-json", default="")
    ap.add_argument("--max-gap-frames", type=int, default=4)
    ap.add_argument("--min-link-score", type=float, default=3.60)
    args = ap.parse_args()

    _write_progress(
        args.progress_json,
        {
            "status": "running",
            "phase": "loading_inputs",
            "processed_events": 0,
            "total_events": 0,
            "built_chain_count": 0,
        },
    )

    event_rows = _load_csv(Path(args.micro_events_csv))
    frame_rows = _load_csv(Path(args.micro_frame_slots_csv))
    event_rows.sort(key=lambda row: (_safe_int(row.get("start_frame"), 0), _safe_int(row.get("end_frame"), 0), _safe_int(row.get("event_id"), 0)))
    total_events = len(event_rows)

    _write_progress(
        args.progress_json,
        {
            "status": "running",
            "phase": "linking_events",
            "processed_events": 0,
            "total_events": total_events,
            "built_chain_count": 0,
        },
    )

    chain_tail_by_id: dict[int, dict[str, Any]] = {}
    chain_event_ids: dict[int, list[int]] = defaultdict(list)
    event_to_chain: dict[int, int] = {}
    event_link_reason: dict[int, str] = {}
    next_chain_id = 1

    active_chain_ids: list[int] = []
    event_by_id: dict[int, dict[str, Any]] = {}

    for idx, row in enumerate(event_rows, start=1):
        event_id = _safe_int(row.get("event_id"), 0)
        event_by_id[event_id] = row
        start_frame = _safe_int(row.get("start_frame"), 0)

        keep_chain_ids: list[int] = []
        for chain_id in active_chain_ids:
            tail = chain_tail_by_id[chain_id]
            if start_frame - _safe_int(tail.get("end_frame"), 0) <= int(args.max_gap_frames):
                keep_chain_ids.append(chain_id)
        active_chain_ids = keep_chain_ids

        best_chain_id = 0
        best_score = -1.0
        best_reason = ""
        for chain_id in active_chain_ids:
            tail = chain_tail_by_id[chain_id]
            score, reason = _event_link_score(
                prev_event=tail,
                event=row,
                max_gap_frames=int(args.max_gap_frames),
            )
            if score > best_score:
                best_score = score
                best_chain_id = chain_id
                best_reason = reason

        if best_chain_id and best_score >= float(args.min_link_score):
            chain_id = best_chain_id
            event_link_reason[event_id] = best_reason
        else:
            chain_id = next_chain_id
            next_chain_id += 1
            event_link_reason[event_id] = "new_chain"
            active_chain_ids.append(chain_id)

        event_to_chain[event_id] = chain_id
        chain_event_ids[chain_id].append(event_id)
        chain_tail_by_id[chain_id] = row

        if idx % 512 == 0 or idx == total_events:
            _write_progress(
                args.progress_json,
                {
                    "status": "running",
                    "phase": "linking_events",
                    "processed_events": idx,
                    "total_events": total_events,
                    "built_chain_count": next_chain_id - 1,
                },
            )

    _write_progress(
        args.progress_json,
        {
            "status": "running",
            "phase": "assembling_chains",
            "processed_events": total_events,
            "total_events": total_events,
            "built_chain_count": len(chain_event_ids),
        },
    )

    frame_rows_by_event: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        frame_rows_by_event[_safe_int(row.get("event_id"), 0)].append(row)

    chain_rows: list[dict[str, Any]] = []
    chain_frame_rows: list[dict[str, Any]] = []
    chain_kind_counter: Counter[str] = Counter()
    confirmation_counter: Counter[str] = Counter()

    for chain_id in sorted(chain_event_ids):
        event_ids = chain_event_ids[chain_id]
        rows = [event_by_id[event_id] for event_id in event_ids if event_id in event_by_id]
        if not rows:
            continue
        rows.sort(key=lambda row: (_safe_int(row.get("start_frame"), 0), _safe_int(row.get("end_frame"), 0), _safe_int(row.get("event_id"), 0)))

        exact_counter: Counter[str] = Counter()
        coarse_counter: Counter[str] = Counter()
        suffix_counter: Counter[str] = Counter()
        probe_counter: Counter[str] = Counter()
        resonance_counter: Counter[str] = Counter()
        event_structure_counter: Counter[str] = Counter()
        freq_values: list[float] = []
        energy_values: list[float] = []
        observation_path: list[str] = []
        child_counts: list[int] = []
        accepted_links: list[str] = []
        chain_frame_count = 0
        start_frame = min(_safe_int(row.get("start_frame"), 0) for row in rows)
        end_frame = max(_safe_int(row.get("end_frame"), 0) for row in rows)
        start_sec = min(_safe_float(row.get("start_time_sec"), 0.0) for row in rows)
        end_sec = max(_safe_float(row.get("end_time_sec"), 0.0) for row in rows)

        for row in rows:
            accepted_links.append(event_link_reason.get(_safe_int(row.get("event_id"), 0), ""))
            resonance_counter[str(row.get("resonance_structure_class", "")).strip()] += 1
            event_structure_counter[str(row.get("event_structure_class", "")).strip()] += 1
            child_counts.append(_safe_int(row.get("child_count"), 0))
            for item in _parse_json_list(row.get("top_micro_note_hypotheses_json", "")):
                if isinstance(item, list) and len(item) >= 2:
                    exact_counter[str(item[0])] += _safe_int(item[1], 0)
            for item in _parse_json_list(row.get("top_coarse_note_hypotheses_json", "")):
                if isinstance(item, list) and len(item) >= 2:
                    coarse_counter[str(item[0])] += _safe_int(item[1], 0)
            for item in _parse_json_list(row.get("top_micro_suffixes_json", "")):
                if isinstance(item, list) and len(item) >= 2:
                    suffix_counter[str(item[0])] += _safe_int(item[1], 0)
            for item in _parse_json_list(row.get("top_probes_json", "")):
                if isinstance(item, list) and len(item) >= 2:
                    probe_counter[str(item[0])] += _safe_int(item[1], 0)
            if row.get("mean_frequency_hz", "") != "":
                freq_values.append(_safe_float(row.get("mean_frequency_hz"), 0.0))
            if row.get("mean_energy", "") != "":
                energy_values.append(_safe_float(row.get("mean_energy"), 0.0))

            event_id = _safe_int(row.get("event_id"), 0)
            for frow in frame_rows_by_event.get(event_id, []):
                chain_frame_count += 1
                observed = str(frow.get("note_token", "")).strip()
                observation_path.append(observed)
                chain_frame_rows.append(
                    {
                        "chain_id": chain_id,
                        "frame_index": _safe_int(frow.get("frame_index"), 0),
                        "time_sec": frow.get("time_sec", ""),
                        "event_id": event_id,
                        "slot_index": _safe_int(frow.get("slot_index"), 0),
                        "observation_rank_in_frame": _safe_int(frow.get("rank_in_frame"), 0),
                        "observed_micro_note_token": observed,
                        "coarse_note_overlay": str(frow.get("coarse_note_overlay", "")).strip(),
                        "micro_suffix": str(frow.get("micro_suffix", "")).strip(),
                        "probe_index": _safe_int(frow.get("probe_index"), 0),
                        "frequency_hz": frow.get("frequency_hz", ""),
                        "energy": frow.get("energy", ""),
                        "rise": frow.get("rise", ""),
                        "continuation": frow.get("continuation", ""),
                        "octave_partner_count": _safe_int(frow.get("octave_partner_count"), 0),
                        "observation_kind": "micro_chain_observation",
                    }
                )

        anchor_micro = exact_counter.most_common(1)[0][0] if exact_counter else ""
        anchor_coarse = coarse_counter.most_common(1)[0][0] if coarse_counter else ""
        chain_kind, confirmation_level = _classify_chain(
            exact_counter=exact_counter,
            coarse_counter=coarse_counter,
            suffix_counter=suffix_counter,
            event_count=len(rows),
            frame_count=chain_frame_count,
            branching_max=max(child_counts) if child_counts else 0,
            resonance_counter=resonance_counter,
        )
        chain_kind_counter[chain_kind] += 1
        confirmation_counter[confirmation_level] += 1

        chain_rows.append(
            {
                "chain_id": chain_id,
                "source_mode": "MICRO_EVENT_TO_CHAIN_CONFIRMED",
                "start_frame": start_frame,
                "end_frame": end_frame,
                "duration_frames": end_frame - start_frame + 1,
                "start_time_sec": f"{start_sec:.9f}",
                "end_time_sec": f"{end_sec:.9f}",
                "event_count": len(rows),
                "observation_frame_count": chain_frame_count,
                "anchor_micro_note_token": anchor_micro,
                "anchor_coarse_note": anchor_coarse,
                "micro_note_hypothesis_count": len(exact_counter),
                "coarse_note_hypothesis_count": len(coarse_counter),
                "micro_suffix_diversity": len(suffix_counter),
                "probe_diversity": len(probe_counter),
                "mean_frequency_hz": f"{_mean(freq_values):.9f}",
                "micro_frequency_span_hz": f"{(max(freq_values) - min(freq_values)) if freq_values else 0.0:.9f}",
                "mean_energy": f"{_mean(energy_values):.9f}",
                "max_child_branching": max(child_counts) if child_counts else 0,
                "chain_structure_class": chain_kind,
                "confirmation_level": confirmation_level,
                "top_micro_note_hypotheses_json": json.dumps(exact_counter.most_common(16), ensure_ascii=False),
                "top_coarse_note_hypotheses_json": json.dumps(coarse_counter.most_common(12), ensure_ascii=False),
                "dominant_micro_suffixes_json": json.dumps(suffix_counter.most_common(16), ensure_ascii=False),
                "dominant_probes_json": json.dumps(probe_counter.most_common(16), ensure_ascii=False),
                "accepted_link_kinds_json": json.dumps(accepted_links, ensure_ascii=False),
                "resonance_structure_counts_json": json.dumps(dict(resonance_counter), ensure_ascii=False),
                "event_structure_counts_json": json.dumps(dict(event_structure_counter), ensure_ascii=False),
                "source_event_ids_json": json.dumps(event_ids, ensure_ascii=False),
                "observed_micro_path_json": json.dumps(observation_path, ensure_ascii=False),
            }
        )

    chain_fields = [
        "chain_id",
        "source_mode",
        "start_frame",
        "end_frame",
        "duration_frames",
        "start_time_sec",
        "end_time_sec",
        "event_count",
        "observation_frame_count",
        "anchor_micro_note_token",
        "anchor_coarse_note",
        "micro_note_hypothesis_count",
        "coarse_note_hypothesis_count",
        "micro_suffix_diversity",
        "probe_diversity",
        "mean_frequency_hz",
        "micro_frequency_span_hz",
        "mean_energy",
        "max_child_branching",
        "chain_structure_class",
        "confirmation_level",
        "top_micro_note_hypotheses_json",
        "top_coarse_note_hypotheses_json",
        "dominant_micro_suffixes_json",
        "dominant_probes_json",
        "accepted_link_kinds_json",
        "resonance_structure_counts_json",
        "event_structure_counts_json",
        "source_event_ids_json",
        "observed_micro_path_json",
    ]
    chain_frame_fields = [
        "chain_id",
        "frame_index",
        "time_sec",
        "event_id",
        "slot_index",
        "observation_rank_in_frame",
        "observed_micro_note_token",
        "coarse_note_overlay",
        "micro_suffix",
        "probe_index",
        "frequency_hz",
        "energy",
        "rise",
        "continuation",
        "octave_partner_count",
        "observation_kind",
    ]

    out_chain_csv = Path(args.out_chains_csv)
    out_chain_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_chain_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=chain_fields)
        writer.writeheader()
        for row in chain_rows:
            writer.writerow({key: row.get(key, "") for key in chain_fields})

    out_chain_frames_csv = Path(args.out_chain_frames_csv)
    with out_chain_frames_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=chain_frame_fields)
        writer.writeheader()
        for row in chain_frame_rows:
            writer.writerow({key: row.get(key, "") for key in chain_frame_fields})

    summary_lines = [
        "MICRO NOTECHAIN BUILDER",
        "=" * 72,
        "source_mode               : MICRO_EVENT_TO_CHAIN_CONFIRMED",
        f"input_event_rows          : {len(event_rows)}",
        f"input_frame_slot_rows     : {len(frame_rows)}",
        f"built_chain_count         : {len(chain_rows)}",
        f"built_chain_frame_rows    : {len(chain_frame_rows)}",
        f"max_gap_frames            : {int(args.max_gap_frames)}",
        f"min_link_score            : {float(args.min_link_score):.3f}",
        "",
        "chain_structure_counts:",
    ]
    for key, value in chain_kind_counter.most_common():
        summary_lines.append(f"  {key}: {value}")
    summary_lines.extend(["", "confirmation_counts:"])
    for key, value in confirmation_counter.most_common():
        summary_lines.append(f"  {key}: {value}")
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "stage": "micro_notechain_builder",
                "source_mode": "MICRO_EVENT_TO_CHAIN_CONFIRMED",
                "inputs": {
                    "micro_events_csv": args.micro_events_csv,
                    "micro_frame_slots_csv": args.micro_frame_slots_csv,
                },
                "parameters": {
                    "max_gap_frames": int(args.max_gap_frames),
                    "min_link_score": float(args.min_link_score),
                },
                "result": {
                    "input_event_rows": len(event_rows),
                    "input_frame_slot_rows": len(frame_rows),
                    "built_chain_count": len(chain_rows),
                    "built_chain_frame_rows": len(chain_frame_rows),
                    "chain_structure_counts": dict(chain_kind_counter),
                    "confirmation_counts": dict(confirmation_counter),
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
            "processed_events": total_events,
            "total_events": total_events,
            "built_chain_count": len(chain_rows),
        },
    )


if __name__ == "__main__":
    main()
