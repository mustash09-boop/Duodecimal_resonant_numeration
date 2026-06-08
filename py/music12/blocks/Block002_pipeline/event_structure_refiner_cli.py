# -*- coding: ascii -*-
# REJECTED BRANCH NOTICE:
# This module was an exploratory coarse-averaging refiner.
# It collapses microshifts too early by building event identity around coarse_note
# and weighted average frequency. For the Block001 rebuild this is forbidden.
# Keep the file only as a rejected reference. Do not use it as part of the new pipeline.
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from music12.blocks.Block002_pipeline.resonance_candidate_inference_core import (
    load_coords_csv,
    load_matrix_csv_memmap,
    load_times_csv,
)


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


def _octave_token(note: str) -> str:
    token = _normalize_note(note)
    if "." not in token:
        return ""
    return token.split(".", 1)[0].strip()


def _same_pitchclass(left: str, right: str) -> bool:
    l_pc = _pitchclass(left)
    r_pc = _pitchclass(right)
    return bool(l_pc and r_pc and l_pc == r_pc)


def _is_octave_related(left: str, right: str) -> bool:
    return bool(left and right and left != right and _same_pitchclass(left, right))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _top_run_length(frames: list[int]) -> tuple[int, int, int]:
    if not frames:
        return 0, 0, 0
    ordered = sorted(frames)
    best_start = ordered[0]
    best_end = ordered[0]
    best_len = 1
    cur_start = ordered[0]
    prev = ordered[0]
    for frame in ordered[1:]:
        if frame == prev + 1:
            prev = frame
            continue
        cur_len = prev - cur_start + 1
        if cur_len > best_len:
            best_start = cur_start
            best_end = prev
            best_len = cur_len
        cur_start = frame
        prev = frame
    cur_len = prev - cur_start + 1
    if cur_len > best_len:
        best_start = cur_start
        best_end = prev
        best_len = cur_len
    return best_start, best_end, best_len


def _write_progress(path_str: str, payload: dict[str, Any]) -> None:
    if not str(path_str).strip():
        return
    Path(path_str).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


@dataclass
class FrameCandidate:
    note_token: str
    coarse_note: str
    weighted_hz: float
    total_energy: float
    max_energy: float
    energy_share: float
    probe_indices: list[int]
    harmonic_hits: dict[int, float]
    harmonic_support_ratio: float
    harmonic_5_support: float
    harmonic_7_support: float
    octave_partner_count: int


@dataclass
class EventAccumulator:
    event_id: int
    slot_index: int
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    last_seen_frame: int
    last_note: str
    frame_count: int = 0
    peak_frame: int = 0
    peak_sec: float = 0.0
    peak_energy: float = 0.0
    total_energy: float = 0.0
    first_energy: float = 0.0
    final_energy: float = 0.0
    note_counter: Counter[str] = field(default_factory=Counter)
    pitchclass_counter: Counter[str] = field(default_factory=Counter)
    harmonic_presence_counter: Counter[int] = field(default_factory=Counter)
    harmonic_energy_counter: Counter[int] = field(default_factory=Counter)
    octave_partner_frames: int = 0
    concurrent_sum: int = 0
    concurrent_max: int = 0
    weighted_hz_sum: float = 0.0
    support_ratio_sum: float = 0.0
    h5_sum: float = 0.0
    h7_sum: float = 0.0
    frame_notes: list[str] = field(default_factory=list)

    def add(self, *, frame_index: int, time_sec: float, candidate: FrameCandidate, concurrency: int) -> None:
        if self.frame_count == 0:
            self.first_energy = candidate.total_energy
            self.peak_frame = frame_index
            self.peak_sec = time_sec
            self.peak_energy = candidate.max_energy
        self.end_frame = frame_index
        self.end_sec = time_sec
        self.last_seen_frame = frame_index
        self.last_note = candidate.coarse_note
        self.frame_count += 1
        self.total_energy += candidate.total_energy
        self.final_energy = candidate.total_energy
        self.note_counter[candidate.coarse_note] += 1
        pc = _pitchclass(candidate.coarse_note)
        if pc:
            self.pitchclass_counter[pc] += 1
        for harmonic_index, harmonic_energy in candidate.harmonic_hits.items():
            if harmonic_energy > 0.0:
                self.harmonic_presence_counter[harmonic_index] += 1
                self.harmonic_energy_counter[harmonic_index] += harmonic_energy
        if candidate.octave_partner_count > 0:
            self.octave_partner_frames += 1
        self.concurrent_sum += concurrency
        self.concurrent_max = max(self.concurrent_max, concurrency)
        self.weighted_hz_sum += candidate.weighted_hz
        self.support_ratio_sum += candidate.harmonic_support_ratio
        self.h5_sum += candidate.harmonic_5_support
        self.h7_sum += candidate.harmonic_7_support
        self.frame_notes.append(candidate.coarse_note)
        if candidate.max_energy > self.peak_energy:
            self.peak_energy = candidate.max_energy
            self.peak_frame = frame_index
            self.peak_sec = time_sec


def _build_frame_candidates(
    *,
    frame_probe_indices: np.ndarray,
    frame_probe_values: np.ndarray,
    coarse_notes: list[str],
    note_tokens: list[str],
    probe_freqs: np.ndarray,
    top_notes_per_frame: int,
    harmonic_tolerance_ratio: float,
    candidate_energy_floor: float,
) -> list[FrameCandidate]:
    positive_mask = frame_probe_values > float(candidate_energy_floor)
    if not positive_mask.any():
        return []

    use_indices = frame_probe_indices[positive_mask]
    use_values = frame_probe_values[positive_mask]
    total_energy = float(use_values.sum())
    if total_energy <= 0.0:
        return []

    grouped: dict[str, dict[str, Any]] = {}
    for probe_index, energy in zip(use_indices.tolist(), use_values.tolist()):
        coarse = coarse_notes[probe_index]
        if not coarse:
            continue
        bucket = grouped.setdefault(
            coarse,
            {
                "note_token": note_tokens[probe_index],
                "coarse_note": coarse,
                "energy_sum": 0.0,
                "max_energy": 0.0,
                "weighted_hz_sum": 0.0,
                "probe_indices": [],
            },
        )
        bucket["energy_sum"] += float(energy)
        bucket["max_energy"] = max(float(bucket["max_energy"]), float(energy))
        bucket["weighted_hz_sum"] += float(energy) * float(probe_freqs[probe_index])
        bucket["probe_indices"].append(int(probe_index))

    ranked = sorted(grouped.values(), key=lambda row: (-float(row["energy_sum"]), row["coarse_note"]))
    top_ranked = ranked[:top_notes_per_frame]

    candidate_notes = {str(row["coarse_note"]): row for row in top_ranked}
    out: list[FrameCandidate] = []
    probe_lookup = {int(idx): float(val) for idx, val in zip(use_indices.tolist(), use_values.tolist())}

    for row in top_ranked:
        coarse = str(row["coarse_note"])
        root_hz = float(row["weighted_hz_sum"]) / max(float(row["energy_sum"]), 1e-9)
        harmonic_hits: dict[int, float] = {}
        harmonic_energy_sum = 0.0
        for harmonic_index in (2, 3, 4, 5, 6, 7, 8):
            expected_hz = root_hz * float(harmonic_index)
            best_hit = 0.0
            for probe_index, energy in probe_lookup.items():
                probe_hz = float(probe_freqs[probe_index])
                rel = abs(probe_hz - expected_hz) / max(expected_hz, 1e-9)
                if rel <= harmonic_tolerance_ratio:
                    best_hit = max(best_hit, float(energy))
            harmonic_hits[harmonic_index] = best_hit
            harmonic_energy_sum += best_hit

        octave_partner_count = 0
        for other_note in candidate_notes.keys():
            if other_note == coarse:
                continue
            if _is_octave_related(other_note, coarse):
                octave_partner_count += 1

        energy_sum = float(row["energy_sum"])
        out.append(
            FrameCandidate(
                note_token=str(row["note_token"]),
                coarse_note=coarse,
                weighted_hz=root_hz,
                total_energy=energy_sum,
                max_energy=float(row["max_energy"]),
                energy_share=energy_sum / max(total_energy, 1e-9),
                probe_indices=list(row["probe_indices"]),
                harmonic_hits=harmonic_hits,
                harmonic_support_ratio=harmonic_energy_sum / max(energy_sum, 1e-9),
                harmonic_5_support=float(harmonic_hits.get(5, 0.0)) / max(energy_sum, 1e-9),
                harmonic_7_support=float(harmonic_hits.get(7, 0.0)) / max(energy_sum, 1e-9),
                octave_partner_count=octave_partner_count,
            )
        )

    out.sort(key=lambda item: (-item.total_energy, item.coarse_note))
    return out


def _match_score(event: EventAccumulator, candidate: FrameCandidate, frame_index: int, allowed_gap_frames: int) -> float:
    gap = frame_index - event.last_seen_frame
    if gap < 0 or gap > allowed_gap_frames:
        return -1.0

    score = 0.0
    if candidate.coarse_note == event.last_note:
        score += 4.0
    elif _same_pitchclass(candidate.coarse_note, event.last_note):
        score += 2.2
    if _is_octave_related(candidate.coarse_note, event.last_note):
        score += 1.3
    hz_ref = event.weighted_hz_sum / max(event.frame_count, 1)
    rel = abs(candidate.weighted_hz - hz_ref) / max(hz_ref, 1e-9)
    score += max(0.0, 1.25 - min(rel, 1.25))
    score += max(0.0, 0.6 - 0.2 * gap)
    score += 0.35 * candidate.harmonic_support_ratio
    score += 0.15 * candidate.energy_share
    return score


def _assign_slot(active_events: list[EventAccumulator], max_parallel_slots: int) -> int:
    used = {event.slot_index for event in active_events}
    for slot_index in range(1, max_parallel_slots + 1):
        if slot_index not in used:
            return slot_index
    return max_parallel_slots


def _finalize_event(
    event: EventAccumulator,
    *,
    parent_event_id: int,
    parent_relation_kind: str,
    child_ids: list[int],
) -> dict[str, Any]:
    duration_frames = event.end_frame - event.start_frame + 1
    primary_note, primary_count = event.note_counter.most_common(1)[0] if event.note_counter else ("", 0)
    note_share = primary_count / max(event.frame_count, 1)
    weighted_hz = event.weighted_hz_sum / max(event.frame_count, 1)
    harmonic_support_ratio = event.support_ratio_sum / max(event.frame_count, 1)
    h5_support = event.h5_sum / max(event.frame_count, 1)
    h7_support = event.h7_sum / max(event.frame_count, 1)
    octave_partner_ratio = event.octave_partner_frames / max(event.frame_count, 1)
    pitchclass_diversity = len(event.pitchclass_counter)
    top_harmonics = sorted(event.harmonic_presence_counter.items(), key=lambda kv: (-kv[1], kv[0]))
    top_notes = sorted(event.note_counter.items(), key=lambda kv: (-kv[1], kv[0]))
    run_start, run_end, run_len = _top_run_length(list(range(event.start_frame, event.end_frame + 1)))

    if note_share >= 0.72 and harmonic_support_ratio >= 0.40:
        event_structure_class = "STABLE_SINGLE_NOTE_EVENT"
    elif octave_partner_ratio >= 0.25:
        event_structure_class = "OCTAVE_LINKED_EVENT"
    elif pitchclass_diversity >= 3 or note_share < 0.42:
        event_structure_class = "SHARED_COMPLEX_EVENT"
    elif duration_frames <= 2 and event.peak_energy >= max(event.first_energy, event.final_energy):
        event_structure_class = "VERY_SHORT_ATTACK_EVENT"
    else:
        event_structure_class = "MIXED_LOCAL_EVENT"

    if duration_frames >= 10 and harmonic_support_ratio >= 0.30:
        resonance_structure_class = "SUSTAINED_RESONANCE_CHAIN"
    elif parent_relation_kind == "SAME_NOTE_CONTINUATION":
        resonance_structure_class = "CHAIN_CONTINUATION"
    elif parent_relation_kind == "ATTACK_TO_BODY_CHILD":
        resonance_structure_class = "BODY_RETURN_OR_SECONDARY_RESPONSE"
    elif pitchclass_diversity >= 3:
        resonance_structure_class = "SHARED_RESONANCE_FIELD"
    elif h5_support >= 0.10 or h7_support >= 0.10:
        resonance_structure_class = "H57_COLORED_RESONANCE_CHAIN"
    else:
        resonance_structure_class = "LOCAL_ATTACK_BODY"

    return {
        "event_id": event.event_id,
        "slot_index": event.slot_index,
        "source_mode": "RAW_PROBE_NEW_PIPELINE",
        "start_frame": event.start_frame,
        "end_frame": event.end_frame,
        "duration_frames": duration_frames,
        "start_time_sec": f"{event.start_sec:.9f}",
        "end_time_sec": f"{event.end_sec:.9f}",
        "peak_frame": event.peak_frame,
        "peak_time_sec": f"{event.peak_sec:.9f}",
        "peak_energy": f"{event.peak_energy:.9f}",
        "first_energy": f"{event.first_energy:.9f}",
        "final_energy": f"{event.final_energy:.9f}",
        "mean_energy": f"{event.total_energy / max(event.frame_count, 1):.9f}",
        "frame_count": event.frame_count,
        "primary_note_token": primary_note,
        "primary_pitchclass": _pitchclass(primary_note),
        "weighted_frequency_hz": f"{weighted_hz:.9f}",
        "primary_note_share": f"{note_share:.9f}",
        "note_hypothesis_count": len(event.note_counter),
        "top_note_hypotheses_json": json.dumps(top_notes[:8], ensure_ascii=False),
        "pitchclass_diversity": pitchclass_diversity,
        "harmonic_hit_count": len([k for k, v in event.harmonic_presence_counter.items() if v > 0]),
        "harmonic_hits_json": json.dumps(top_harmonics[:8], ensure_ascii=False),
        "harmonic_support_ratio": f"{harmonic_support_ratio:.9f}",
        "harmonic_5_support": f"{h5_support:.9f}",
        "harmonic_7_support": f"{h7_support:.9f}",
        "harmonic_5_7_signature": f"{(h5_support + h7_support):.9f}",
        "octave_partner_count": event.octave_partner_frames,
        "octave_partner_ratio": f"{octave_partner_ratio:.9f}",
        "concurrent_mean": f"{(event.concurrent_sum / max(event.frame_count, 1)):.9f}",
        "concurrent_max": event.concurrent_max,
        "event_structure_class": event_structure_class,
        "resonance_structure_class": resonance_structure_class,
        "parent_event_id": parent_event_id if parent_event_id else "",
        "parent_relation_kind": parent_relation_kind,
        "child_event_ids_json": json.dumps(child_ids, ensure_ascii=False),
        "child_count": len(child_ids),
        "best_contiguous_run_start": run_start,
        "best_contiguous_run_end": run_end,
        "best_contiguous_run_len": run_len,
    }


def _build_parent_child(rows: list[dict[str, Any]], max_gap_frames: int = 12) -> tuple[dict[int, int], dict[int, list[int]], Counter[str]]:
    ordered = sorted(rows, key=lambda row: (int(row["start_frame"]), int(row["end_frame"]), int(row["event_id"])))
    parent_of: dict[int, int] = {}
    children_of: dict[int, list[int]] = {}
    relation_counter: Counter[str] = Counter()
    for idx, row in enumerate(ordered):
        event_id = int(row["event_id"])
        start = int(row["start_frame"])
        primary_note = str(row["primary_note_token"])
        weighted_hz = _safe_float(row.get("weighted_frequency_hz"), 0.0)
        best_parent = 0
        best_relation = ""
        best_score = -1.0
        for prev in ordered[:idx]:
            prev_id = int(prev["event_id"])
            prev_end = int(prev["end_frame"])
            prev_note = str(prev["primary_note_token"])
            prev_hz = _safe_float(prev.get("weighted_frequency_hz"), 0.0)
            gap = start - prev_end
            if gap < 0 or gap > max_gap_frames:
                continue
            relation = ""
            score = 0.0
            if primary_note and prev_note and primary_note == prev_note:
                relation = "SAME_NOTE_CONTINUATION"
                score = 3.0
            elif _same_pitchclass(primary_note, prev_note):
                relation = "SAME_PITCHCLASS_CONTINUATION"
                score = 2.0
            elif _is_octave_related(primary_note, prev_note):
                relation = "OCTAVE_COMPANION_CONTINUATION"
                score = 1.6
            elif prev_hz and weighted_hz and weighted_hz < prev_hz and gap <= 4:
                relation = "ATTACK_TO_BODY_CHILD"
                score = 1.1
            else:
                continue
            score += max(0.0, 0.75 - 0.05 * gap)
            if score > best_score:
                best_score = score
                best_parent = prev_id
                best_relation = relation
        if best_parent:
            parent_of[event_id] = best_parent
            children_of.setdefault(best_parent, []).append(event_id)
            relation_counter[best_relation] += 1
    return parent_of, children_of, relation_counter


def main() -> None:
    raise SystemExit(
        "Rejected branch: event_structure_refiner_cli averages away microshifts too early. "
        "Use a future microshift-preserving refiner instead."
    )

    ap = argparse.ArgumentParser(description="New raw-driven event structure refiner for Block001/Block002: build early event slots, note hypotheses, harmonic structure, resonance structure, and parent-child links directly from probe data before passports.")
    ap.add_argument("--probe-matrix-csv", required=True)
    ap.add_argument("--probe-times-csv", required=True)
    ap.add_argument("--probe-coords-csv", required=True)
    ap.add_argument("--out-events-csv", required=True)
    ap.add_argument("--out-frame-slots-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--progress-json", default="")
    ap.add_argument("--matrix-cache-dir", default="")
    ap.add_argument("--frame-chunk-size", type=int, default=128)
    ap.add_argument("--top-probes-per-frame", type=int, default=24)
    ap.add_argument("--top-notes-per-frame", type=int, default=10)
    ap.add_argument("--max-parallel-slots", type=int, default=10)
    ap.add_argument("--energy-threshold", type=float, default=0.003)
    ap.add_argument("--candidate-energy-floor", type=float, default=0.010)
    ap.add_argument("--start-energy-share-threshold", type=float, default=0.055)
    ap.add_argument("--allowed-gap-frames", type=int, default=2)
    ap.add_argument("--harmonic-tolerance-ratio", type=float, default=0.03)
    args = ap.parse_args()

    matrix, matrix_cache_info = load_matrix_csv_memmap(
        args.probe_matrix_csv,
        cache_dir=(Path(args.matrix_cache_dir) if str(args.matrix_cache_dir).strip() else None),
    )
    times = load_times_csv(args.probe_times_csv)
    coords = load_coords_csv(args.probe_coords_csv)

    probe_count, frame_count = int(matrix.shape[0]), int(matrix.shape[1])
    usable_probe_count = min(probe_count, len(coords))
    note_tokens = [coords[i].note_token for i in range(usable_probe_count)]
    coarse_notes = [_normalize_note(note_tokens[i]) for i in range(usable_probe_count)]
    probe_freqs = np.asarray([coords[i].frequency_hz for i in range(usable_probe_count)], dtype=np.float32)

    _write_progress(
        args.progress_json,
        {
            "status": "running",
            "phase": "initialized",
            "processed_frames": 0,
            "total_frames": frame_count,
            "usable_probe_count": usable_probe_count,
            "active_event_count": 0,
            "finalized_event_count": 0,
        },
    )

    out_frame_rows: list[dict[str, Any]] = []
    active_events: list[EventAccumulator] = []
    finalized_events: list[EventAccumulator] = []
    next_event_id = 1

    def finalize_stale(current_frame: int) -> None:
        stale: list[EventAccumulator] = []
        keep: list[EventAccumulator] = []
        for event in active_events:
            if current_frame - event.last_seen_frame > int(args.allowed_gap_frames):
                stale.append(event)
            else:
                keep.append(event)
        active_events[:] = keep
        finalized_events.extend(stale)

    chunk_size = max(1, int(args.frame_chunk_size))
    top_probe_count = min(max(1, int(args.top_probes_per_frame)), usable_probe_count)

    for chunk_start in range(0, frame_count, chunk_size):
        chunk_end = min(frame_count, chunk_start + chunk_size)
        chunk = np.asarray(matrix[:usable_probe_count, chunk_start:chunk_end], dtype=np.float32)
        top_idx = np.argpartition(chunk, -top_probe_count, axis=0)[-top_probe_count:, :]
        top_vals = np.take_along_axis(chunk, top_idx, axis=0)
        sort_order = np.argsort(top_vals, axis=0)[::-1, :]
        sorted_idx = np.take_along_axis(top_idx, sort_order, axis=0)
        sorted_vals = np.take_along_axis(top_vals, sort_order, axis=0)

        for local_col in range(chunk_end - chunk_start):
            frame_index = chunk_start + local_col
            time_sec = float(times[frame_index]) if frame_index < len(times) else 0.0
            frame_probe_indices = sorted_idx[:, local_col]
            frame_probe_values = sorted_vals[:, local_col]
            frame_candidates = _build_frame_candidates(
                frame_probe_indices=frame_probe_indices,
                frame_probe_values=frame_probe_values,
                coarse_notes=coarse_notes,
                note_tokens=note_tokens,
                probe_freqs=probe_freqs,
                top_notes_per_frame=int(args.top_notes_per_frame),
                harmonic_tolerance_ratio=float(args.harmonic_tolerance_ratio),
                candidate_energy_floor=float(args.candidate_energy_floor),
            )
            frame_candidates = [
                candidate
                for candidate in frame_candidates
                if candidate.total_energy >= float(args.energy_threshold)
                and candidate.energy_share >= float(args.start_energy_share_threshold)
            ]
            frame_candidates = frame_candidates[: int(args.max_parallel_slots)]

            finalize_stale(frame_index)

            used_event_ids: set[int] = set()
            concurrency = len(frame_candidates)
            for rank, candidate in enumerate(frame_candidates, start=1):
                best_event = None
                best_score = -1.0
                for event in active_events:
                    if event.event_id in used_event_ids:
                        continue
                    score = _match_score(event, candidate, frame_index, int(args.allowed_gap_frames))
                    if score > best_score:
                        best_score = score
                        best_event = event
                if best_event is not None and best_score >= 1.35:
                    event = best_event
                else:
                    slot_index = _assign_slot(active_events, int(args.max_parallel_slots))
                    event = EventAccumulator(
                        event_id=next_event_id,
                        slot_index=slot_index,
                        start_frame=frame_index,
                        end_frame=frame_index,
                        start_sec=time_sec,
                        end_sec=time_sec,
                        last_seen_frame=frame_index,
                        last_note=candidate.coarse_note,
                    )
                    active_events.append(event)
                    next_event_id += 1
                event.add(frame_index=frame_index, time_sec=time_sec, candidate=candidate, concurrency=concurrency)
                used_event_ids.add(event.event_id)

                out_frame_rows.append(
                    {
                        "frame_index": frame_index,
                        "time_sec": f"{time_sec:.9f}",
                        "rank_in_frame": rank,
                        "event_id": event.event_id,
                        "slot_index": event.slot_index,
                        "coarse_note": candidate.coarse_note,
                        "weighted_hz": f"{candidate.weighted_hz:.9f}",
                        "total_energy": f"{candidate.total_energy:.9f}",
                        "max_energy": f"{candidate.max_energy:.9f}",
                        "energy_share": f"{candidate.energy_share:.9f}",
                        "harmonic_support_ratio": f"{candidate.harmonic_support_ratio:.9f}",
                        "harmonic_5_support": f"{candidate.harmonic_5_support:.9f}",
                        "harmonic_7_support": f"{candidate.harmonic_7_support:.9f}",
                        "octave_partner_count": candidate.octave_partner_count,
                    }
                )

        _write_progress(
            args.progress_json,
            {
                "status": "running",
                "phase": "frame_chunks",
                "processed_frames": chunk_end,
                "total_frames": frame_count,
                "active_event_count": len(active_events),
                "finalized_event_count": len(finalized_events),
            },
        )

    finalized_events.extend(active_events)
    active_events = []

    base_rows = []
    for event in finalized_events:
        base_rows.append(
            {
                "event_id": event.event_id,
                "slot_index": event.slot_index,
                "start_frame": event.start_frame,
                "end_frame": event.end_frame,
                "primary_note_token": event.note_counter.most_common(1)[0][0] if event.note_counter else "",
                "weighted_frequency_hz": f"{(event.weighted_hz_sum / max(event.frame_count, 1)):.9f}",
            }
        )
    parent_of, children_of, relation_counter = _build_parent_child(base_rows)

    out_event_rows: list[dict[str, Any]] = []
    event_class_counter: Counter[str] = Counter()
    resonance_class_counter: Counter[str] = Counter()
    for event in sorted(finalized_events, key=lambda item: (item.start_frame, item.end_frame, item.event_id)):
        row = _finalize_event(
            event,
            parent_event_id=parent_of.get(event.event_id, 0),
            parent_relation_kind=next(
                (
                    kind
                    for kind, parent_id in [(k, parent_of.get(event.event_id, 0)) for k in relation_counter.keys()]
                    if parent_id
                ),
                "",
            ),
            child_ids=children_of.get(event.event_id, []),
        )
        # recompute exact relation kind from parent rows
        if row["parent_event_id"]:
            parent_row = next((r for r in base_rows if int(r["event_id"]) == int(row["parent_event_id"])), None)
            if parent_row is not None:
                parent_note = str(parent_row["primary_note_token"])
                this_note = str(row["primary_note_token"])
                if parent_note == this_note and this_note:
                    row["parent_relation_kind"] = "SAME_NOTE_CONTINUATION"
                elif _same_pitchclass(parent_note, this_note):
                    row["parent_relation_kind"] = "SAME_PITCHCLASS_CONTINUATION"
                elif _is_octave_related(parent_note, this_note):
                    row["parent_relation_kind"] = "OCTAVE_COMPANION_CONTINUATION"
                else:
                    row["parent_relation_kind"] = "ATTACK_TO_BODY_CHILD"

        out_event_rows.append(row)
        event_class_counter[str(row["event_structure_class"])] += 1
        resonance_class_counter[str(row["resonance_structure_class"])] += 1

    event_fields = [
        "event_id",
        "slot_index",
        "source_mode",
        "start_frame",
        "end_frame",
        "duration_frames",
        "start_time_sec",
        "end_time_sec",
        "peak_frame",
        "peak_time_sec",
        "peak_energy",
        "first_energy",
        "final_energy",
        "mean_energy",
        "frame_count",
        "primary_note_token",
        "primary_pitchclass",
        "weighted_frequency_hz",
        "primary_note_share",
        "note_hypothesis_count",
        "top_note_hypotheses_json",
        "pitchclass_diversity",
        "harmonic_hit_count",
        "harmonic_hits_json",
        "harmonic_support_ratio",
        "harmonic_5_support",
        "harmonic_7_support",
        "harmonic_5_7_signature",
        "octave_partner_count",
        "octave_partner_ratio",
        "concurrent_mean",
        "concurrent_max",
        "event_structure_class",
        "resonance_structure_class",
        "parent_event_id",
        "parent_relation_kind",
        "child_event_ids_json",
        "child_count",
        "best_contiguous_run_start",
        "best_contiguous_run_end",
        "best_contiguous_run_len",
    ]
    frame_fields = [
        "frame_index",
        "time_sec",
        "rank_in_frame",
        "event_id",
        "slot_index",
        "coarse_note",
        "weighted_hz",
        "total_energy",
        "max_energy",
        "energy_share",
        "harmonic_support_ratio",
        "harmonic_5_support",
        "harmonic_7_support",
        "octave_partner_count",
    ]

    out_events_csv = Path(args.out_events_csv)
    out_events_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_events_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=event_fields)
        writer.writeheader()
        for row in out_event_rows:
            writer.writerow({key: row.get(key, "") for key in event_fields})

    with Path(args.out_frame_slots_csv).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=frame_fields)
        writer.writeheader()
        for row in out_frame_rows:
            writer.writerow({key: row.get(key, "") for key in frame_fields})

    summary_lines = [
        "EVENT STRUCTURE REFINER",
        "=" * 72,
        f"source_mode            : RAW_PROBE_NEW_PIPELINE",
        f"probe_count             : {usable_probe_count}",
        f"frame_count             : {frame_count}",
        f"frame_slot_rows         : {len(out_frame_rows)}",
        f"refined_event_count     : {len(out_event_rows)}",
        f"max_parallel_slots      : {int(args.max_parallel_slots)}",
        f"matrix_cache_reused     : {matrix_cache_info.reused_existing_cache}",
        "",
        "event_structure_counts:",
    ]
    for key, value in event_class_counter.most_common():
        summary_lines.append(f"  {key}: {value}")
    summary_lines.extend(["", "resonance_structure_counts:"])
    for key, value in resonance_class_counter.most_common():
        summary_lines.append(f"  {key}: {value}")
    summary_lines.extend(["", "parent_relation_counts:"])
    for key, value in relation_counter.most_common():
        summary_lines.append(f"  {key}: {value}")
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "stage": "event_structure_refiner",
                "source_mode": "RAW_PROBE_NEW_PIPELINE",
                "inputs": {
                    "probe_matrix_csv": args.probe_matrix_csv,
                    "probe_times_csv": args.probe_times_csv,
                    "probe_coords_csv": args.probe_coords_csv,
                },
                "parameters": {
                    "frame_chunk_size": int(args.frame_chunk_size),
                    "top_probes_per_frame": int(args.top_probes_per_frame),
                    "top_notes_per_frame": int(args.top_notes_per_frame),
                    "max_parallel_slots": int(args.max_parallel_slots),
                    "energy_threshold": float(args.energy_threshold),
                    "candidate_energy_floor": float(args.candidate_energy_floor),
                    "start_energy_share_threshold": float(args.start_energy_share_threshold),
                    "allowed_gap_frames": int(args.allowed_gap_frames),
                    "harmonic_tolerance_ratio": float(args.harmonic_tolerance_ratio),
                },
                "matrix_cache": {
                    "source_csv": matrix_cache_info.source_csv,
                    "cache_dat": matrix_cache_info.cache_dat,
                    "cache_meta_json": matrix_cache_info.cache_meta_json,
                    "shape": list(matrix_cache_info.shape),
                    "dtype": matrix_cache_info.dtype,
                    "source_size_bytes": matrix_cache_info.source_size_bytes,
                    "source_mtime_ns": matrix_cache_info.source_mtime_ns,
                    "reused_existing_cache": matrix_cache_info.reused_existing_cache,
                },
                "result": {
                    "probe_count": usable_probe_count,
                    "frame_count": frame_count,
                    "frame_slot_rows": len(out_frame_rows),
                    "refined_event_count": len(out_event_rows),
                    "event_structure_counts": dict(event_class_counter),
                    "resonance_structure_counts": dict(resonance_class_counter),
                    "parent_relation_counts": dict(relation_counter),
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
            "processed_frames": frame_count,
            "total_frames": frame_count,
            "refined_event_count": len(out_event_rows),
        },
    )


if __name__ == "__main__":
    main()
