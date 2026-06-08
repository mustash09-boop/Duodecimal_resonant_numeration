# -*- coding: ascii -*-
from __future__ import annotations

import argparse
import csv
import heapq
import json
from collections import Counter, defaultdict
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


def _micro_suffix(note_token: str) -> str:
    s = str(note_token or "").strip()
    if "'" not in s:
        return "-"
    return s.split("'", 1)[1] or "-"


def _same_pitchclass(left: str, right: str) -> bool:
    l_pc = _pitchclass(left)
    r_pc = _pitchclass(right)
    return bool(l_pc and r_pc and l_pc == r_pc)


def _write_progress(path_str: str, payload: dict[str, Any]) -> None:
    if not str(path_str).strip():
        return
    Path(path_str).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass
class MicroFrameCandidate:
    probe_index: int
    note_token: str
    coarse_note: str
    micro_suffix: str
    frequency_hz: float
    energy: float
    rise: float
    continuation: float
    local_rank: int


@dataclass
class MicroEventAccumulator:
    event_id: int
    slot_index: int
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    last_seen_frame: int
    last_probe_index: int
    last_note_token: str
    last_frequency_hz: float
    frame_count: int = 0
    peak_frame: int = 0
    peak_sec: float = 0.0
    peak_energy: float = 0.0
    total_energy: float = 0.0
    first_energy: float = 0.0
    final_energy: float = 0.0
    exact_note_counter: Counter[str] = field(default_factory=Counter)
    coarse_note_counter: Counter[str] = field(default_factory=Counter)
    micro_suffix_counter: Counter[str] = field(default_factory=Counter)
    probe_counter: Counter[int] = field(default_factory=Counter)
    rise_values: list[float] = field(default_factory=list)
    continuation_values: list[float] = field(default_factory=list)
    frequency_values: list[float] = field(default_factory=list)
    energy_values: list[float] = field(default_factory=list)
    frame_path: list[str] = field(default_factory=list)
    concurrent_sum: int = 0
    concurrent_max: int = 0
    octave_partner_frames: int = 0

    def add(self, *, frame_index: int, time_sec: float, candidate: MicroFrameCandidate, concurrency: int, octave_partner_count: int) -> None:
        if self.frame_count == 0:
            self.first_energy = candidate.energy
            self.peak_frame = frame_index
            self.peak_sec = time_sec
            self.peak_energy = candidate.energy
        self.end_frame = frame_index
        self.end_sec = time_sec
        self.last_seen_frame = frame_index
        self.last_probe_index = candidate.probe_index
        self.last_note_token = candidate.note_token
        self.last_frequency_hz = candidate.frequency_hz
        self.frame_count += 1
        self.total_energy += candidate.energy
        self.final_energy = candidate.energy
        self.exact_note_counter[candidate.note_token] += 1
        self.coarse_note_counter[candidate.coarse_note] += 1
        self.micro_suffix_counter[candidate.micro_suffix] += 1
        self.probe_counter[candidate.probe_index] += 1
        self.rise_values.append(candidate.rise)
        self.continuation_values.append(candidate.continuation)
        self.frequency_values.append(candidate.frequency_hz)
        self.energy_values.append(candidate.energy)
        self.frame_path.append(candidate.note_token)
        self.concurrent_sum += concurrency
        self.concurrent_max = max(self.concurrent_max, concurrency)
        if octave_partner_count > 0:
            self.octave_partner_frames += 1
        if candidate.energy > self.peak_energy:
            self.peak_energy = candidate.energy
            self.peak_frame = frame_index
            self.peak_sec = time_sec


def _build_frame_micro_candidates(
    *,
    frame_probe_indices: np.ndarray,
    frame_probe_values: np.ndarray,
    note_tokens: list[str],
    coarse_notes: list[str],
    probe_freqs: np.ndarray,
    energy_threshold: float,
    top_micro_per_frame: int,
) -> list[MicroFrameCandidate]:
    positive_mask = frame_probe_values >= float(energy_threshold)
    if not positive_mask.any():
        return []

    probe_indices = frame_probe_indices[positive_mask]
    energies = frame_probe_values[positive_mask]
    ranked_pairs = sorted(
        zip(probe_indices.tolist(), energies.tolist()),
        key=lambda pair: (-float(pair[1]), int(pair[0])),
    )[:top_micro_per_frame]

    out: list[MicroFrameCandidate] = []
    energy_array = np.asarray([float(v) for _idx, v in ranked_pairs], dtype=np.float32)
    local_mean = float(energy_array.mean()) if energy_array.size else 0.0

    for rank, (probe_index, energy) in enumerate(ranked_pairs, start=1):
        probe_index = int(probe_index)
        energy = float(energy)
        note_token = str(note_tokens[probe_index]).strip()
        coarse_note = str(coarse_notes[probe_index]).strip()
        freq_hz = float(probe_freqs[probe_index])
        rise = max(0.0, energy - local_mean)
        continuation = min(1.0, energy / max(local_mean, 1e-9)) if local_mean > 0.0 else 0.0
        out.append(
            MicroFrameCandidate(
                probe_index=probe_index,
                note_token=note_token,
                coarse_note=coarse_note,
                micro_suffix=_micro_suffix(note_token),
                frequency_hz=freq_hz,
                energy=energy,
                rise=rise,
                continuation=continuation,
                local_rank=rank,
            )
        )
    return out


def _match_score(event: MicroEventAccumulator, candidate: MicroFrameCandidate, frame_index: int, max_gap_frames: int, rel_freq_tolerance: float) -> float:
    gap = frame_index - event.last_seen_frame
    if gap < 0 or gap > max_gap_frames:
        return -1.0

    score = 0.0
    if candidate.probe_index == event.last_probe_index:
        score += 5.0
    if candidate.note_token == event.last_note_token:
        score += 3.5
    elif candidate.coarse_note == _normalize_note(event.last_note_token):
        score += 1.5
    elif _same_pitchclass(candidate.note_token, event.last_note_token):
        score += 0.75

    rel_freq = abs(candidate.frequency_hz - event.last_frequency_hz) / max(event.last_frequency_hz, 1e-9)
    if rel_freq <= rel_freq_tolerance:
        score += 2.0 * (1.0 - rel_freq / max(rel_freq_tolerance, 1e-9))
    elif rel_freq <= rel_freq_tolerance * 3.0:
        score += 0.35

    score += max(0.0, 0.5 - 0.15 * gap)
    score += 0.20 * candidate.continuation
    score += 0.10 * candidate.rise
    return score


def _assign_slot(active_events: list[MicroEventAccumulator], max_parallel_slots: int) -> int:
    used_slots = {event.slot_index for event in active_events}
    for slot_index in range(1, max_parallel_slots + 1):
        if slot_index not in used_slots:
            return slot_index
    return max_parallel_slots


def _build_parent_child(rows: list[dict[str, Any]], max_gap_frames: int = 12) -> tuple[dict[int, int], dict[int, list[int]], Counter[str]]:
    ordered = sorted(rows, key=lambda row: (int(row["start_frame"]), int(row["end_frame"]), int(row["event_id"])))
    parent_of: dict[int, int] = {}
    children_of: dict[int, list[int]] = {}
    relation_counter: Counter[str] = Counter()
    row_by_id: dict[int, dict[str, Any]] = {int(row["event_id"]): row for row in ordered}
    active_ids: set[int] = set()
    active_by_exact: dict[str, set[int]] = defaultdict(set)
    active_by_coarse: dict[str, set[int]] = defaultdict(set)
    active_by_pitchclass: dict[str, set[int]] = defaultdict(set)
    active_by_freq_bucket: dict[int, set[int]] = defaultdict(set)
    expiry_heap: list[tuple[int, int]] = []

    def _freq_bucket(freq_hz: float) -> int:
        return int(round(freq_hz / 2.0))

    def _remove_active(event_id: int) -> None:
        if event_id not in active_ids:
            return
        active_ids.discard(event_id)
        old = row_by_id[event_id]
        old_exact = str(old["primary_micro_note_token"])
        old_coarse = str(old["primary_coarse_note"])
        old_pitch = _pitchclass(old_exact)
        old_freq = _safe_float(old.get("mean_frequency_hz"), 0.0)
        if old_exact:
            active_by_exact[old_exact].discard(event_id)
        if old_coarse:
            active_by_coarse[old_coarse].discard(event_id)
        if old_pitch:
            active_by_pitchclass[old_pitch].discard(event_id)
        if old_freq > 0.0:
            active_by_freq_bucket[_freq_bucket(old_freq)].discard(event_id)

    for row in ordered:
        event_id = int(row["event_id"])
        start = int(row["start_frame"])
        exact_note = str(row["primary_micro_note_token"])
        coarse_note = str(row["primary_coarse_note"])
        freq_hz = _safe_float(row.get("mean_frequency_hz"), 0.0)
        threshold = start - max_gap_frames
        while expiry_heap and expiry_heap[0][0] < threshold:
            _end_frame, expired_event_id = heapq.heappop(expiry_heap)
            _remove_active(expired_event_id)
        best_parent = 0
        best_kind = ""
        best_score = -1.0
        candidate_ids: set[int] = set()
        if exact_note:
            candidate_ids.update(active_by_exact.get(exact_note, set()))
        if coarse_note:
            candidate_ids.update(active_by_coarse.get(coarse_note, set()))
        pitch = _pitchclass(exact_note)
        if pitch:
            candidate_ids.update(active_by_pitchclass.get(pitch, set()))
        if freq_hz > 0.0:
            base_bucket = _freq_bucket(freq_hz)
            for bucket in range(base_bucket - 1, base_bucket + 2):
                candidate_ids.update(active_by_freq_bucket.get(bucket, set()))
        if not candidate_ids:
            candidate_ids.update(active_ids)
        for prev_id in candidate_ids:
            prev = row_by_id[prev_id]
            prev_end = int(prev["end_frame"])
            gap = start - prev_end
            if gap < 0 or gap > max_gap_frames:
                continue
            prev_exact = str(prev["primary_micro_note_token"])
            prev_coarse = str(prev["primary_coarse_note"])
            prev_freq = _safe_float(prev.get("mean_frequency_hz"), 0.0)
            kind = ""
            score = 0.0
            if exact_note and prev_exact and exact_note == prev_exact:
                kind = "EXACT_MICRO_CONTINUATION"
                score = 4.0
            elif coarse_note and prev_coarse and coarse_note == prev_coarse:
                kind = "COARSE_OVERLAY_CONTINUATION"
                score = 2.5
            elif _same_pitchclass(exact_note, prev_exact):
                kind = "PITCHCLASS_CONTINUATION"
                score = 1.8
            elif prev_freq and freq_hz and abs(freq_hz - prev_freq) / max(prev_freq, 1e-9) <= 0.006:
                kind = "NEAR_MICRO_DRIFT_CONTINUATION"
                score = 2.2
            else:
                continue
            score += max(0.0, 0.8 - 0.05 * gap)
            if score > best_score:
                best_score = score
                best_parent = prev_id
                best_kind = kind
        if best_parent:
            parent_of[event_id] = best_parent
            children_of.setdefault(best_parent, []).append(event_id)
            relation_counter[best_kind] += 1
        active_ids.add(event_id)
        if exact_note:
            active_by_exact[exact_note].add(event_id)
        if coarse_note:
            active_by_coarse[coarse_note].add(event_id)
        if pitch:
            active_by_pitchclass[pitch].add(event_id)
        if freq_hz > 0.0:
            active_by_freq_bucket[_freq_bucket(freq_hz)].add(event_id)
        heapq.heappush(expiry_heap, (int(row["end_frame"]), event_id))
    return parent_of, children_of, relation_counter


def _finalize_event(
    event: MicroEventAccumulator,
    *,
    parent_event_id: int,
    parent_relation_kind: str,
    child_ids: list[int],
) -> dict[str, Any]:
    duration_frames = event.end_frame - event.start_frame + 1
    primary_micro_note, primary_micro_count = event.exact_note_counter.most_common(1)[0] if event.exact_note_counter else ("", 0)
    primary_coarse_note, primary_coarse_count = event.coarse_note_counter.most_common(1)[0] if event.coarse_note_counter else ("", 0)
    primary_micro_share = primary_micro_count / max(event.frame_count, 1)
    primary_coarse_share = primary_coarse_count / max(event.frame_count, 1)
    mean_frequency_hz = _mean(event.frequency_values)
    micro_frequency_span_hz = (max(event.frequency_values) - min(event.frequency_values)) if event.frequency_values else 0.0
    micro_diversity = len(event.micro_suffix_counter)
    probe_diversity = len(event.probe_counter)
    octave_partner_ratio = event.octave_partner_frames / max(event.frame_count, 1)
    mean_rise = _mean(event.rise_values)
    mean_continuation = _mean(event.continuation_values)

    if primary_micro_share >= 0.80 and probe_diversity <= 2:
        event_structure_class = "MICRO_STABLE_EVENT"
    elif primary_coarse_share >= 0.75 and micro_diversity >= 2:
        event_structure_class = "MICRO_DRIFT_WITHIN_COARSE_EVENT"
    elif octave_partner_ratio >= 0.20:
        event_structure_class = "MICRO_OCTAVE_LINKED_EVENT"
    elif micro_diversity >= 4 or probe_diversity >= 5:
        event_structure_class = "MICRO_SHARED_COMPLEX_EVENT"
    else:
        event_structure_class = "MICRO_LOCAL_EVENT"

    if parent_relation_kind == "EXACT_MICRO_CONTINUATION":
        resonance_structure_class = "MICRO_CHAIN_CONTINUATION"
    elif parent_relation_kind == "NEAR_MICRO_DRIFT_CONTINUATION":
        resonance_structure_class = "MICRO_DRIFT_CHAIN"
    elif parent_relation_kind == "COARSE_OVERLAY_CONTINUATION":
        resonance_structure_class = "COARSE_CONTINUATION_WITH_PRESERVED_MICROSHIFT"
    elif duration_frames >= 8 and primary_micro_share >= 0.55:
        resonance_structure_class = "SUSTAINED_MICRO_RESONANCE"
    elif micro_diversity >= 4:
        resonance_structure_class = "SHARED_MICRO_RESONANCE_FIELD"
    else:
        resonance_structure_class = "LOCAL_MICRO_ATTACK_BODY"

    return {
        "event_id": event.event_id,
        "slot_index": event.slot_index,
        "source_mode": "RAW_PROBE_MICROSHIFT_PRESERVED",
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
        "primary_micro_note_token": primary_micro_note,
        "primary_coarse_note": primary_coarse_note,
        "primary_micro_share": f"{primary_micro_share:.9f}",
        "primary_coarse_share": f"{primary_coarse_share:.9f}",
        "micro_note_hypothesis_count": len(event.exact_note_counter),
        "coarse_note_hypothesis_count": len(event.coarse_note_counter),
        "micro_suffix_diversity": micro_diversity,
        "probe_diversity": probe_diversity,
        "top_micro_note_hypotheses_json": json.dumps(event.exact_note_counter.most_common(12), ensure_ascii=False),
        "top_coarse_note_hypotheses_json": json.dumps(event.coarse_note_counter.most_common(8), ensure_ascii=False),
        "micro_path_json": json.dumps(event.frame_path, ensure_ascii=False),
        "top_micro_suffixes_json": json.dumps(event.micro_suffix_counter.most_common(12), ensure_ascii=False),
        "top_probes_json": json.dumps(event.probe_counter.most_common(12), ensure_ascii=False),
        "mean_frequency_hz": f"{mean_frequency_hz:.9f}",
        "micro_frequency_span_hz": f"{micro_frequency_span_hz:.9f}",
        "mean_rise": f"{mean_rise:.9f}",
        "mean_continuation": f"{mean_continuation:.9f}",
        "octave_partner_count": event.octave_partner_frames,
        "octave_partner_ratio": f"{octave_partner_ratio:.9f}",
        "concurrent_mean": f"{event.concurrent_sum / max(event.frame_count, 1):.9f}",
        "concurrent_max": event.concurrent_max,
        "event_structure_class": event_structure_class,
        "resonance_structure_class": resonance_structure_class,
        "parent_event_id": parent_event_id if parent_event_id else "",
        "parent_relation_kind": parent_relation_kind,
        "child_event_ids_json": json.dumps(child_ids, ensure_ascii=False),
        "child_count": len(child_ids),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Microshift-preserving raw event structure refiner: build event slots from exact probe/note micro-life without collapsing early to coarse_note.")
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
    ap.add_argument("--top-micro-per-frame", type=int, default=10)
    ap.add_argument("--max-parallel-slots", type=int, default=10)
    ap.add_argument("--energy-threshold", type=float, default=0.010)
    ap.add_argument("--allowed-gap-frames", type=int, default=2)
    ap.add_argument("--relative-frequency-tolerance", type=float, default=0.004)
    ap.add_argument("--match-threshold", type=float, default=2.20)
    args = ap.parse_args()

    _write_progress(
        args.progress_json,
        {
            "status": "running",
            "phase": "loading_inputs",
            "processed_frames": 0,
            "total_frames": 0,
            "active_event_count": 0,
            "finalized_event_count": 0,
        },
    )

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
    active_events: list[MicroEventAccumulator] = []
    finalized_events: list[MicroEventAccumulator] = []
    next_event_id = 1

    def finalize_stale(current_frame: int) -> None:
        stale: list[MicroEventAccumulator] = []
        keep: list[MicroEventAccumulator] = []
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
            frame_candidates = _build_frame_micro_candidates(
                frame_probe_indices=frame_probe_indices,
                frame_probe_values=frame_probe_values,
                note_tokens=note_tokens,
                coarse_notes=coarse_notes,
                probe_freqs=probe_freqs,
                energy_threshold=float(args.energy_threshold),
                top_micro_per_frame=int(args.top_micro_per_frame),
            )

            finalize_stale(frame_index)

            concurrency = len(frame_candidates)
            used_event_ids: set[int] = set()

            for rank, candidate in enumerate(frame_candidates, start=1):
                octave_partner_count = sum(
                    1
                    for other in frame_candidates
                    if other.note_token != candidate.note_token and _same_pitchclass(other.note_token, candidate.note_token)
                )
                best_event = None
                best_score = -1.0
                for event in active_events:
                    if event.event_id in used_event_ids:
                        continue
                    score = _match_score(
                        event,
                        candidate,
                        frame_index,
                        int(args.allowed_gap_frames),
                        float(args.relative_frequency_tolerance),
                    )
                    if score > best_score:
                        best_score = score
                        best_event = event
                if best_event is not None and best_score >= float(args.match_threshold):
                    event = best_event
                else:
                    slot_index = _assign_slot(active_events, int(args.max_parallel_slots))
                    event = MicroEventAccumulator(
                        event_id=next_event_id,
                        slot_index=slot_index,
                        start_frame=frame_index,
                        end_frame=frame_index,
                        start_sec=time_sec,
                        end_sec=time_sec,
                        last_seen_frame=frame_index,
                        last_probe_index=candidate.probe_index,
                        last_note_token=candidate.note_token,
                        last_frequency_hz=candidate.frequency_hz,
                    )
                    active_events.append(event)
                    next_event_id += 1

                event.add(
                    frame_index=frame_index,
                    time_sec=time_sec,
                    candidate=candidate,
                    concurrency=concurrency,
                    octave_partner_count=octave_partner_count,
                )
                used_event_ids.add(event.event_id)

                out_frame_rows.append(
                    {
                        "frame_index": frame_index,
                        "time_sec": f"{time_sec:.9f}",
                        "rank_in_frame": rank,
                        "event_id": event.event_id,
                        "slot_index": event.slot_index,
                        "probe_index": candidate.probe_index,
                        "note_token": candidate.note_token,
                        "coarse_note_overlay": candidate.coarse_note,
                        "micro_suffix": candidate.micro_suffix,
                        "frequency_hz": f"{candidate.frequency_hz:.9f}",
                        "energy": f"{candidate.energy:.9f}",
                        "rise": f"{candidate.rise:.9f}",
                        "continuation": f"{candidate.continuation:.9f}",
                        "octave_partner_count": octave_partner_count,
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
        primary_micro_note = event.exact_note_counter.most_common(1)[0][0] if event.exact_note_counter else ""
        primary_coarse_note = event.coarse_note_counter.most_common(1)[0][0] if event.coarse_note_counter else ""
        base_rows.append(
            {
                "event_id": event.event_id,
                "slot_index": event.slot_index,
                "start_frame": event.start_frame,
                "end_frame": event.end_frame,
                "primary_micro_note_token": primary_micro_note,
                "primary_coarse_note": primary_coarse_note,
                "mean_frequency_hz": f"{_mean(event.frequency_values):.9f}",
            }
        )
    _write_progress(
        args.progress_json,
        {
            "status": "running",
            "phase": "parent_child_indexing",
            "processed_frames": frame_count,
            "total_frames": frame_count,
            "active_event_count": len(active_events),
            "finalized_event_count": len(finalized_events),
        },
    )
    parent_of, children_of, relation_counter = _build_parent_child(base_rows)
    base_row_by_id: dict[int, dict[str, Any]] = {int(row["event_id"]): row for row in base_rows}

    out_event_rows: list[dict[str, Any]] = []
    event_class_counter: Counter[str] = Counter()
    resonance_class_counter: Counter[str] = Counter()

    relation_lookup: dict[int, str] = {}
    for row in base_rows:
        event_id = int(row["event_id"])
        parent_id = parent_of.get(event_id, 0)
        if not parent_id:
            relation_lookup[event_id] = ""
            continue
        parent_row = base_row_by_id.get(parent_id)
        if parent_row is None:
            relation_lookup[event_id] = ""
            continue
        child_exact = str(row["primary_micro_note_token"])
        parent_exact = str(parent_row["primary_micro_note_token"])
        child_coarse = str(row["primary_coarse_note"])
        parent_coarse = str(parent_row["primary_coarse_note"])
        child_hz = _safe_float(row.get("mean_frequency_hz"), 0.0)
        parent_hz = _safe_float(parent_row.get("mean_frequency_hz"), 0.0)
        if child_exact and parent_exact and child_exact == parent_exact:
            relation_lookup[event_id] = "EXACT_MICRO_CONTINUATION"
        elif child_coarse and parent_coarse and child_coarse == parent_coarse:
            relation_lookup[event_id] = "COARSE_OVERLAY_CONTINUATION"
        elif _same_pitchclass(child_exact, parent_exact):
            relation_lookup[event_id] = "PITCHCLASS_CONTINUATION"
        elif parent_hz and child_hz and abs(child_hz - parent_hz) / max(parent_hz, 1e-9) <= 0.006:
            relation_lookup[event_id] = "NEAR_MICRO_DRIFT_CONTINUATION"
        else:
            relation_lookup[event_id] = ""

    _write_progress(
        args.progress_json,
        {
            "status": "running",
            "phase": "writing_outputs",
            "processed_frames": frame_count,
            "total_frames": frame_count,
            "active_event_count": len(active_events),
            "finalized_event_count": len(finalized_events),
        },
    )

    for event in sorted(finalized_events, key=lambda item: (item.start_frame, item.end_frame, item.event_id)):
        row = _finalize_event(
            event,
            parent_event_id=parent_of.get(event.event_id, 0),
            parent_relation_kind=relation_lookup.get(event.event_id, ""),
            child_ids=children_of.get(event.event_id, []),
        )
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
        "primary_micro_note_token",
        "primary_coarse_note",
        "primary_micro_share",
        "primary_coarse_share",
        "micro_note_hypothesis_count",
        "coarse_note_hypothesis_count",
        "micro_suffix_diversity",
        "probe_diversity",
        "top_micro_note_hypotheses_json",
        "top_coarse_note_hypotheses_json",
        "micro_path_json",
        "top_micro_suffixes_json",
        "top_probes_json",
        "mean_frequency_hz",
        "micro_frequency_span_hz",
        "mean_rise",
        "mean_continuation",
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
    ]
    frame_fields = [
        "frame_index",
        "time_sec",
        "rank_in_frame",
        "event_id",
        "slot_index",
        "probe_index",
        "note_token",
        "coarse_note_overlay",
        "micro_suffix",
        "frequency_hz",
        "energy",
        "rise",
        "continuation",
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
        "MICRO EVENT STRUCTURE REFINER",
        "=" * 72,
        f"source_mode               : RAW_PROBE_MICROSHIFT_PRESERVED",
        f"probe_count                : {usable_probe_count}",
        f"frame_count                : {frame_count}",
        f"frame_slot_rows            : {len(out_frame_rows)}",
        f"refined_event_count        : {len(out_event_rows)}",
        f"max_parallel_slots         : {int(args.max_parallel_slots)}",
        f"matrix_cache_reused        : {matrix_cache_info.reused_existing_cache}",
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
                "stage": "micro_event_structure_refiner",
                "source_mode": "RAW_PROBE_MICROSHIFT_PRESERVED",
                "inputs": {
                    "probe_matrix_csv": args.probe_matrix_csv,
                    "probe_times_csv": args.probe_times_csv,
                    "probe_coords_csv": args.probe_coords_csv,
                },
                "parameters": {
                    "frame_chunk_size": int(args.frame_chunk_size),
                    "top_probes_per_frame": int(args.top_probes_per_frame),
                    "top_micro_per_frame": int(args.top_micro_per_frame),
                    "max_parallel_slots": int(args.max_parallel_slots),
                    "energy_threshold": float(args.energy_threshold),
                    "allowed_gap_frames": int(args.allowed_gap_frames),
                    "relative_frequency_tolerance": float(args.relative_frequency_tolerance),
                    "match_threshold": float(args.match_threshold),
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
