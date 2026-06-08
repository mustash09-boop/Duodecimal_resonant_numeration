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


def _parse_pairs(value: str) -> list[list[Any]]:
    try:
        loaded = json.loads(str(value or "").strip() or "[]")
        return loaded if isinstance(loaded, list) else []
    except Exception:
        return []


def _top_keys_from_pairs(value: str) -> set[str]:
    out: set[str] = set()
    for item in _parse_pairs(value):
        if isinstance(item, list) and item:
            out.add(str(item[0]))
    return out


def _overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _coalesce_score(
    *,
    prev_row: dict[str, Any],
    next_row: dict[str, Any],
    max_gap_frames: int,
) -> tuple[float, str]:
    prev_end = _safe_int(prev_row.get("end_frame"), 0)
    next_start = _safe_int(next_row.get("start_frame"), 0)
    gap = next_start - prev_end
    # A coalescer may only stitch forward-in-time continuation.
    # Same-frame siblings belong to a simultaneous cohort, not to one chain life.
    if gap <= 0 or gap > max_gap_frames:
        return -1.0, "gap_out_of_range"

    prev_coarse = str(prev_row.get("anchor_coarse_note", "")).strip()
    next_coarse = str(next_row.get("anchor_coarse_note", "")).strip()
    if not prev_coarse or not next_coarse or prev_coarse != next_coarse:
        return -1.0, "different_coarse"

    prev_micro = str(prev_row.get("anchor_micro_note_token", "")).strip()
    next_micro = str(next_row.get("anchor_micro_note_token", "")).strip()
    prev_freq = _safe_float(prev_row.get("mean_frequency_hz"), 0.0)
    next_freq = _safe_float(next_row.get("mean_frequency_hz"), 0.0)
    prev_chain_kind = str(prev_row.get("chain_structure_class", "")).strip()
    next_chain_kind = str(next_row.get("chain_structure_class", "")).strip()
    prev_confirm = str(prev_row.get("confirmation_level", "")).strip()
    next_confirm = str(next_row.get("confirmation_level", "")).strip()

    prev_probe_keys = _top_keys_from_pairs(prev_row.get("dominant_probes_json", ""))
    next_probe_keys = _top_keys_from_pairs(next_row.get("dominant_probes_json", ""))
    prev_suffix_keys = _top_keys_from_pairs(prev_row.get("dominant_micro_suffixes_json", ""))
    next_suffix_keys = _top_keys_from_pairs(next_row.get("dominant_micro_suffixes_json", ""))
    probe_overlap = _overlap_ratio(prev_probe_keys, next_probe_keys)
    suffix_overlap = _overlap_ratio(prev_suffix_keys, next_suffix_keys)

    score = 0.0
    reasons: list[str] = []

    if prev_micro and next_micro and prev_micro == next_micro:
        score += 3.0
        reasons.append("exact_micro")

    if prev_freq > 0.0 and next_freq > 0.0:
        rel_freq = abs(next_freq - prev_freq) / max(prev_freq, 1e-9)
        if rel_freq <= 0.0015:
            score += 2.0
            reasons.append("tight_freq")
        elif rel_freq <= 0.0035:
            score += 1.0
            reasons.append("near_freq")

    if probe_overlap > 0.0:
        score += probe_overlap * 3.0
        reasons.append(f"probe_{probe_overlap:.2f}")
    if suffix_overlap > 0.0:
        score += suffix_overlap * 2.2
        reasons.append(f"suffix_{suffix_overlap:.2f}")

    if prev_chain_kind == "EXACT_MICRO_NOTECHAIN" and next_chain_kind == "EXACT_MICRO_NOTECHAIN":
        score += 1.0
        reasons.append("both_exact")
    if "COARSE_ANCHORED_MICRO_DRIFT_CHAIN" in {prev_chain_kind, next_chain_kind}:
        score += 0.35
        reasons.append("coarse_drift_context")
    if prev_confirm == "CONFIRMED_CHAIN" or next_confirm == "CONFIRMED_CHAIN":
        score += 0.25
        reasons.append("confirmed_side")

    score += max(0.0, 0.75 - 0.12 * gap)
    return score, "|".join(reasons) if reasons else "weak"


def _coalesced_class(class_counter: Counter[str], confirm_counter: Counter[str], chain_count: int) -> tuple[str, str]:
    top_class = class_counter.most_common(1)[0][0] if class_counter else "LOCAL_MICRO_CHAIN"
    if chain_count >= 3 and top_class in {"COARSE_ANCHORED_MICRO_DRIFT_CHAIN", "EXACT_MICRO_NOTECHAIN"}:
        out_class = "COALESCED_NOTECHAIN_BACKBONE"
    elif chain_count >= 2 and class_counter.get("BRANCHED_MICRO_RESPONSE_CHAIN", 0) >= max(1, chain_count // 2):
        out_class = "COALESCED_BRANCH_RESPONSE"
    elif chain_count >= 2 and class_counter.get("COARSE_ANCHORED_MICRO_DRIFT_CHAIN", 0) >= max(1, chain_count // 2):
        out_class = "COALESCED_COARSE_DRIFT_BACKBONE"
    else:
        out_class = "LOCAL_CHAIN_GROUP"

    if confirm_counter.get("CONFIRMED_CHAIN", 0) >= max(1, chain_count // 2):
        status = "COALESCED_CONFIRMED"
    elif confirm_counter.get("PROBABLE_CHAIN", 0) >= 1 or chain_count >= 2:
        status = "COALESCED_PROBABLE"
    else:
        status = "COALESCED_WEAK"
    return out_class, status


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Coalesce adjacent compatible micro-chains into longer living chain groups without erasing microshift identity."
    )
    ap.add_argument("--chains-csv", required=True)
    ap.add_argument("--chain-frames-csv", required=True)
    ap.add_argument("--out-coalesced-chains-csv", required=True)
    ap.add_argument("--out-coalesced-chain-frames-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--progress-json", default="")
    ap.add_argument("--max-gap-frames", type=int, default=8)
    ap.add_argument("--min-coalesce-score", type=float, default=2.75)
    args = ap.parse_args()

    _write_progress(
        args.progress_json,
        {
            "status": "running",
            "phase": "loading_inputs",
            "processed_chains": 0,
            "total_chains": 0,
            "coalesced_group_count": 0,
        },
    )

    chain_rows = _load_csv(Path(args.chains_csv))
    frame_rows = _load_csv(Path(args.chain_frames_csv))
    total_chains = len(chain_rows)
    chain_rows.sort(key=lambda row: (_safe_int(row.get("start_frame"), 0), _safe_int(row.get("end_frame"), 0), _safe_int(row.get("chain_id"), 0)))

    _write_progress(
        args.progress_json,
        {
            "status": "running",
            "phase": "coalescing",
            "processed_chains": 0,
            "total_chains": total_chains,
            "coalesced_group_count": 0,
        },
    )

    group_of_chain: dict[int, int] = {}
    group_chain_ids: dict[int, list[int]] = defaultdict(list)
    next_group_id = 1
    active_group_tail: dict[int, dict[str, Any]] = {}
    active_group_ids: list[int] = []

    for idx, row in enumerate(chain_rows, start=1):
        chain_id = _safe_int(row.get("chain_id"), 0)
        start_frame = _safe_int(row.get("start_frame"), 0)

        keep_groups: list[int] = []
        for group_id in active_group_ids:
            tail = active_group_tail[group_id]
            if start_frame - _safe_int(tail.get("end_frame"), 0) <= int(args.max_gap_frames):
                keep_groups.append(group_id)
        active_group_ids = keep_groups

        best_group_id = 0
        best_score = -1.0
        for group_id in active_group_ids:
            tail = active_group_tail[group_id]
            score, _reason = _coalesce_score(
                prev_row=tail,
                next_row=row,
                max_gap_frames=int(args.max_gap_frames),
            )
            if score > best_score:
                best_score = score
                best_group_id = group_id

        if best_group_id and best_score >= float(args.min_coalesce_score):
            group_id = best_group_id
        else:
            group_id = next_group_id
            next_group_id += 1
            active_group_ids.append(group_id)

        group_of_chain[chain_id] = group_id
        group_chain_ids[group_id].append(chain_id)
        active_group_tail[group_id] = row

        if idx % 512 == 0 or idx == total_chains:
            _write_progress(
                args.progress_json,
                {
                    "status": "running",
                    "phase": "coalescing",
                    "processed_chains": idx,
                    "total_chains": total_chains,
                    "coalesced_group_count": len(group_chain_ids),
                },
            )

    chain_by_id: dict[int, dict[str, Any]] = {_safe_int(row.get("chain_id"), 0): row for row in chain_rows}
    frames_by_chain: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        frames_by_chain[_safe_int(row.get("chain_id"), 0)].append(row)

    coalesced_rows: list[dict[str, Any]] = []
    coalesced_frame_rows: list[dict[str, Any]] = []
    coalesced_class_counter: Counter[str] = Counter()
    coalesced_status_counter: Counter[str] = Counter()

    for group_id in sorted(group_chain_ids):
        chain_ids = group_chain_ids[group_id]
        rows = [chain_by_id[chain_id] for chain_id in chain_ids if chain_id in chain_by_id]
        if not rows:
            continue
        rows.sort(key=lambda row: (_safe_int(row.get("start_frame"), 0), _safe_int(row.get("end_frame"), 0), _safe_int(row.get("chain_id"), 0)))

        class_counter: Counter[str] = Counter()
        confirm_counter: Counter[str] = Counter()
        exact_counter: Counter[str] = Counter()
        coarse_counter: Counter[str] = Counter()
        suffix_counter: Counter[str] = Counter()
        probe_counter: Counter[str] = Counter()
        start_frame = min(_safe_int(row.get("start_frame"), 0) for row in rows)
        end_frame = max(_safe_int(row.get("end_frame"), 0) for row in rows)
        start_sec = min(_safe_float(row.get("start_time_sec"), 0.0) for row in rows)
        end_sec = max(_safe_float(row.get("end_time_sec"), 0.0) for row in rows)
        observation_frame_count = 0
        event_count_sum = 0
        mean_freqs: list[float] = []
        mean_energies: list[float] = []
        child_branching_values: list[int] = []
        observed_path: list[str] = []

        for row in rows:
            class_counter[str(row.get("chain_structure_class", "")).strip()] += 1
            confirm_counter[str(row.get("confirmation_level", "")).strip()] += 1
            event_count_sum += _safe_int(row.get("event_count"), 0)
            child_branching_values.append(_safe_int(row.get("max_child_branching"), 0))
            mean_freqs.append(_safe_float(row.get("mean_frequency_hz"), 0.0))
            mean_energies.append(_safe_float(row.get("mean_energy"), 0.0))
            for item in _parse_pairs(row.get("top_micro_note_hypotheses_json", "")):
                if isinstance(item, list) and len(item) >= 2:
                    exact_counter[str(item[0])] += _safe_int(item[1], 0)
            for item in _parse_pairs(row.get("top_coarse_note_hypotheses_json", "")):
                if isinstance(item, list) and len(item) >= 2:
                    coarse_counter[str(item[0])] += _safe_int(item[1], 0)
            for item in _parse_pairs(row.get("dominant_micro_suffixes_json", "")):
                if isinstance(item, list) and len(item) >= 2:
                    suffix_counter[str(item[0])] += _safe_int(item[1], 0)
            for item in _parse_pairs(row.get("dominant_probes_json", "")):
                if isinstance(item, list) and len(item) >= 2:
                    probe_counter[str(item[0])] += _safe_int(item[1], 0)
            for frow in frames_by_chain.get(_safe_int(row.get("chain_id"), 0), []):
                observation_frame_count += 1
                observed_path.append(str(frow.get("observed_micro_note_token", "")).strip())
                coalesced_frame_rows.append(
                    {
                        "coalesced_group_id": group_id,
                        "chain_id": _safe_int(row.get("chain_id"), 0),
                        "frame_index": _safe_int(frow.get("frame_index"), 0),
                        "time_sec": frow.get("time_sec", ""),
                        "event_id": _safe_int(frow.get("event_id"), 0),
                        "slot_index": _safe_int(frow.get("slot_index"), 0),
                        "observed_micro_note_token": str(frow.get("observed_micro_note_token", "")).strip(),
                        "coarse_note_overlay": str(frow.get("coarse_note_overlay", "")).strip(),
                        "micro_suffix": str(frow.get("micro_suffix", "")).strip(),
                        "probe_index": _safe_int(frow.get("probe_index"), 0),
                        "frequency_hz": frow.get("frequency_hz", ""),
                        "energy": frow.get("energy", ""),
                        "rise": frow.get("rise", ""),
                        "continuation": frow.get("continuation", ""),
                        "observation_kind": "coalesced_micro_chain_observation",
                    }
                )

        out_class, out_status = _coalesced_class(class_counter, confirm_counter, len(rows))
        coalesced_class_counter[out_class] += 1
        coalesced_status_counter[out_status] += 1

        coalesced_rows.append(
            {
                "coalesced_group_id": group_id,
                "source_mode": "MICRO_CHAIN_COALESCED",
                "start_frame": start_frame,
                "end_frame": end_frame,
                "duration_frames": end_frame - start_frame + 1,
                "start_time_sec": f"{start_sec:.9f}",
                "end_time_sec": f"{end_sec:.9f}",
                "chain_count": len(rows),
                "summed_event_count": event_count_sum,
                "observation_frame_count": observation_frame_count,
                "anchor_micro_note_token": exact_counter.most_common(1)[0][0] if exact_counter else "",
                "anchor_coarse_note": coarse_counter.most_common(1)[0][0] if coarse_counter else "",
                "micro_note_hypothesis_count": len(exact_counter),
                "coarse_note_hypothesis_count": len(coarse_counter),
                "micro_suffix_diversity": len(suffix_counter),
                "probe_diversity": len(probe_counter),
                "mean_frequency_hz": f"{_mean(mean_freqs):.9f}",
                "mean_energy": f"{_mean(mean_energies):.9f}",
                "max_child_branching": max(child_branching_values) if child_branching_values else 0,
                "coalesced_structure_class": out_class,
                "coalesced_status": out_status,
                "top_micro_note_hypotheses_json": json.dumps(exact_counter.most_common(16), ensure_ascii=False),
                "top_coarse_note_hypotheses_json": json.dumps(coarse_counter.most_common(12), ensure_ascii=False),
                "dominant_micro_suffixes_json": json.dumps(suffix_counter.most_common(16), ensure_ascii=False),
                "dominant_probes_json": json.dumps(probe_counter.most_common(16), ensure_ascii=False),
                "source_chain_ids_json": json.dumps(chain_ids, ensure_ascii=False),
                "source_chain_structure_counts_json": json.dumps(dict(class_counter), ensure_ascii=False),
                "source_confirmation_counts_json": json.dumps(dict(confirm_counter), ensure_ascii=False),
                "observed_micro_path_json": json.dumps(observed_path, ensure_ascii=False),
            }
        )

    group_fields = [
        "coalesced_group_id",
        "source_mode",
        "start_frame",
        "end_frame",
        "duration_frames",
        "start_time_sec",
        "end_time_sec",
        "chain_count",
        "summed_event_count",
        "observation_frame_count",
        "anchor_micro_note_token",
        "anchor_coarse_note",
        "micro_note_hypothesis_count",
        "coarse_note_hypothesis_count",
        "micro_suffix_diversity",
        "probe_diversity",
        "mean_frequency_hz",
        "mean_energy",
        "max_child_branching",
        "coalesced_structure_class",
        "coalesced_status",
        "top_micro_note_hypotheses_json",
        "top_coarse_note_hypotheses_json",
        "dominant_micro_suffixes_json",
        "dominant_probes_json",
        "source_chain_ids_json",
        "source_chain_structure_counts_json",
        "source_confirmation_counts_json",
        "observed_micro_path_json",
    ]
    frame_fields = [
        "coalesced_group_id",
        "chain_id",
        "frame_index",
        "time_sec",
        "event_id",
        "slot_index",
        "observed_micro_note_token",
        "coarse_note_overlay",
        "micro_suffix",
        "probe_index",
        "frequency_hz",
        "energy",
        "rise",
        "continuation",
        "observation_kind",
    ]

    out_group_csv = Path(args.out_coalesced_chains_csv)
    out_group_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_group_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=group_fields)
        writer.writeheader()
        for row in coalesced_rows:
            writer.writerow({key: row.get(key, "") for key in group_fields})

    out_frames_csv = Path(args.out_coalesced_chain_frames_csv)
    with out_frames_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=frame_fields)
        writer.writeheader()
        for row in coalesced_frame_rows:
            writer.writerow({key: row.get(key, "") for key in frame_fields})

    summary_lines = [
        "MICRO CHAIN COALESCER",
        "=" * 72,
        "source_mode               : MICRO_CHAIN_COALESCED",
        f"input_chain_rows          : {len(chain_rows)}",
        f"input_chain_frame_rows    : {len(frame_rows)}",
        f"coalesced_group_count     : {len(coalesced_rows)}",
        f"coalesced_frame_rows      : {len(coalesced_frame_rows)}",
        f"max_gap_frames            : {int(args.max_gap_frames)}",
        f"min_coalesce_score        : {float(args.min_coalesce_score):.3f}",
        "",
        "coalesced_structure_counts:",
    ]
    for key, value in coalesced_class_counter.most_common():
        summary_lines.append(f"  {key}: {value}")
    summary_lines.extend(["", "coalesced_status_counts:"])
    for key, value in coalesced_status_counter.most_common():
        summary_lines.append(f"  {key}: {value}")
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "stage": "micro_chain_coalescer",
                "source_mode": "MICRO_CHAIN_COALESCED",
                "inputs": {
                    "chains_csv": args.chains_csv,
                    "chain_frames_csv": args.chain_frames_csv,
                },
                "parameters": {
                    "max_gap_frames": int(args.max_gap_frames),
                    "min_coalesce_score": float(args.min_coalesce_score),
                },
                "result": {
                    "input_chain_rows": len(chain_rows),
                    "input_chain_frame_rows": len(frame_rows),
                    "coalesced_group_count": len(coalesced_rows),
                    "coalesced_frame_rows": len(coalesced_frame_rows),
                    "coalesced_structure_counts": dict(coalesced_class_counter),
                    "coalesced_status_counts": dict(coalesced_status_counter),
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
            "processed_chains": total_chains,
            "total_chains": total_chains,
            "coalesced_group_count": len(coalesced_rows),
        },
    )


if __name__ == "__main__":
    main()
