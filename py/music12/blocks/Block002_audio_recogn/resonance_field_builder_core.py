from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from music12.core.spiral12_geometry import (
    SpiralPosition,
    parse_token_to_spiral,
)


# ============================================================
# DATA
# ============================================================

@dataclass(frozen=True)
class ResonanceCoord:
    probe_index: int
    octave: int
    degree12_symbol: str
    degree12_index0: int
    subdivisions: tuple[str, ...]
    frequency_hz: float
    global_index: int
    delta_radius: int
    delta_vector: tuple[int, ...]

    @property
    def note_token(self) -> str:
        base = f"{self.octave}.{self.degree12_symbol}"
        if not self.subdivisions:
            return base
        subs = "".join(self.subdivisions)
        return f"{base}'{subs}"

    @property
    def spiral(self) -> Optional[SpiralPosition]:
        return parse_token_to_spiral(self.note_token)


@dataclass(frozen=True)
class ResonanceFieldEvent:
    frame_index: int
    time_sec: float
    probe_index: int

    note_token: str
    spiral: SpiralPosition

    octave: int
    degree12_symbol: str
    degree12_index0: int
    subdivisions: tuple[str, ...]
    delta_radius: int
    delta_vector: tuple[int, ...]

    frequency_hz: float
    energy: float


class FastTrajectory:
    def __init__(self, trajectory_id: int, first_event: ResonanceFieldEvent):
        self.trajectory_id = trajectory_id
        self.events = [first_event]

        self.last_time = first_event.time_sec
        self.count = 1
        self.sum_energy = first_event.energy
        self.sum_arc = first_event.spiral.absolute_arc

    def add(self, ev: ResonanceFieldEvent) -> None:
        self.events.append(ev)
        self.last_time = ev.time_sec
        self.count += 1
        self.sum_energy += ev.energy
        self.sum_arc += ev.spiral.absolute_arc

    @property
    def mean_arc(self) -> float:
        return self.sum_arc / max(1, self.count)

    @property
    def time_start_sec(self) -> float:
        return self.events[0].time_sec

    @property
    def time_end_sec(self) -> float:
        return self.last_time

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.time_end_sec - self.time_start_sec)

    @property
    def temporal_density(self) -> float:
        frame_span = self.events[-1].frame_index - self.events[0].frame_index + 1
        return self.count / max(1, frame_span)

    @property
    def dominant_probe_index(self) -> int:
        counts: dict[int, int] = {}
        for ev in self.events:
            counts[ev.probe_index] = counts.get(ev.probe_index, 0) + 1
        return max(counts.items(), key=lambda x: x[1])[0]

    @property
    def dominant_note_token(self) -> str:
        counts: dict[str, int] = {}
        for ev in self.events:
            counts[ev.note_token] = counts.get(ev.note_token, 0) + 1
        return max(counts.items(), key=lambda x: x[1])[0]

    @property
    def mean_energy(self) -> float:
        return self.sum_energy / max(1, self.count)


# ============================================================
# BASIC HELPERS
# ============================================================

def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        s = _safe_str(v)
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        s = _safe_str(v)
        if s == "":
            return default
        return int(s)
    except Exception:
        return default


def _load_json_list(raw: str) -> list[Any]:
    try:
        data = json.loads(raw or "[]")
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def arc_distance(a: SpiralPosition, b: SpiralPosition) -> float:
    return abs(a.absolute_arc - b.absolute_arc)


# ============================================================
# CSV LOADERS
# ============================================================

def load_probe_matrix_csv(path: str | Path) -> np.ndarray:
    path = Path(path)

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return np.zeros((0, 0), dtype=np.float32)

    data_rows = rows[1:]
    matrix: list[list[float]] = []

    for row in data_rows:
        if len(row) <= 1:
            continue
        matrix.append([_safe_float(v, 0.0) for v in row[1:]])

    if not matrix:
        return np.zeros((0, 0), dtype=np.float32)

    return np.asarray(matrix, dtype=np.float32)


def load_probe_times_csv(path: str | Path) -> np.ndarray:
    path = Path(path)

    out: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(_safe_float(row.get("time_seconds", ""), 0.0))

    return np.asarray(out, dtype=np.float32)


def load_probe_coords_delta_csv(path: str | Path) -> list[ResonanceCoord]:
    path = Path(path)
    coords: list[ResonanceCoord] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            probe_index = _safe_int(row.get("probe_index", ""), len(coords))
            octave = _safe_int(row.get("octave", ""), 0)

            degree_raw = _safe_str(row.get("degree12", ""))
            degree12_symbol = degree_raw if degree_raw else "1"

            alphabet = "123456789ABC"
            degree12_index0 = alphabet.find(degree12_symbol.upper())
            if degree12_index0 < 0:
                degree12_index0 = 0

            subdivisions_raw = row.get("subdivisions", "[]")
            parsed_subs = _load_json_list(subdivisions_raw)
            subdivisions = tuple(str(x) for x in parsed_subs)

            frequency_hz = _safe_float(row.get("frequency_hz", ""), 0.0)
            global_index = _safe_int(row.get("global_index", ""), probe_index)
            delta_radius = _safe_int(row.get("delta_radius", ""), 0)

            delta_vector_raw = row.get("delta_vector", "[]")
            parsed_vec = _load_json_list(delta_vector_raw)
            delta_vector = tuple(_safe_int(x, 0) for x in parsed_vec)

            coords.append(
                ResonanceCoord(
                    probe_index=probe_index,
                    octave=octave,
                    degree12_symbol=degree12_symbol,
                    degree12_index0=degree12_index0,
                    subdivisions=subdivisions,
                    frequency_hz=frequency_hz,
                    global_index=global_index,
                    delta_radius=delta_radius,
                    delta_vector=delta_vector,
                )
            )

    return coords


# ============================================================
# FIELD EVENTS
# ============================================================

def build_resonance_field_events(
    *,
    matrix: np.ndarray,
    times: np.ndarray,
    coords: list[ResonanceCoord],
    energy_threshold: float = 0.0,
    top_k_per_frame: int = 0,
) -> list[ResonanceFieldEvent]:
    events: list[ResonanceFieldEvent] = []

    if matrix.size == 0:
        return events

    frame_count = matrix.shape[1]
    probe_count = matrix.shape[0]

    usable_frame_count = min(frame_count, len(times))
    usable_probe_count = min(probe_count, len(coords))

    for frame_index in range(usable_frame_count):
        frame_values = matrix[:usable_probe_count, frame_index]

        candidates: list[tuple[float, int]] = []
        for probe_index, energy in enumerate(frame_values):
            energy = float(energy)
            if energy < energy_threshold:
                continue
            candidates.append((energy, probe_index))

        candidates.sort(key=lambda x: x[0], reverse=True)
        if top_k_per_frame > 0:
            candidates = candidates[:top_k_per_frame]

        for energy, probe_index in candidates:
            coord = coords[probe_index]
            spiral = coord.spiral
            if spiral is None:
                continue

            events.append(
                ResonanceFieldEvent(
                    frame_index=frame_index,
                    time_sec=float(times[frame_index]),
                    probe_index=coord.probe_index,

                    note_token=coord.note_token,
                    spiral=spiral,

                    octave=coord.octave,
                    degree12_symbol=coord.degree12_symbol,
                    degree12_index0=coord.degree12_index0,
                    subdivisions=coord.subdivisions,
                    delta_radius=coord.delta_radius,
                    delta_vector=coord.delta_vector,

                    frequency_hz=coord.frequency_hz,
                    energy=float(energy),
                )
            )

    return events


# ============================================================
# TRAJECTORIES
# ============================================================

def _can_attach(
    event: ResonanceFieldEvent,
    traj: FastTrajectory,
    *,
    max_time_gap_sec: float,
    max_arc_gap: float,
) -> bool:
    dt = event.time_sec - traj.last_time
    if dt < 0:
        return False
    if dt > max_time_gap_sec:
        return False

    mean_arc_pos = SpiralPosition(
        absolute_arc=traj.mean_arc,
        turn_index=0,
        local_step=0.0,
    )

    if arc_distance(event.spiral, mean_arc_pos) > max_arc_gap:
        return False

    return True


def build_resonance_trajectories(
    *,
    events: list[ResonanceFieldEvent],
    max_time_gap_sec: float = 0.05,
    max_arc_gap: float = 0.5,
    min_events_per_trajectory: int = 2,
) -> list[FastTrajectory]:
    if not events:
        return []

    events_sorted = sorted(events, key=lambda e: (e.time_sec, e.probe_index))

    trajectories: list[FastTrajectory] = []
    next_id = 1

    for ev in events_sorted:
        attached = False

        candidate_trajs = sorted(
            trajectories,
            key=lambda t: abs(t.mean_arc - ev.spiral.absolute_arc),
        )

        for traj in candidate_trajs:
            if _can_attach(
                ev,
                traj,
                max_time_gap_sec=max_time_gap_sec,
                max_arc_gap=max_arc_gap,
            ):
                traj.add(ev)
                attached = True
                break

        if not attached:
            trajectories.append(FastTrajectory(next_id, ev))
            next_id += 1

    trajectories = [t for t in trajectories if t.count >= min_events_per_trajectory]
    return trajectories


# ============================================================
# ROW EXPORT
# ============================================================

def field_events_to_rows(events: list[ResonanceFieldEvent]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for e in events:
        rows.append(
            {
                "frame_index": e.frame_index,
                "time_sec": round(e.time_sec, 6),
                "probe_index": e.probe_index,
                "note_token": e.note_token,
                "spiral_arc": round(e.spiral.absolute_arc, 6),
                "spiral_turn_index": e.spiral.turn_index,
                "spiral_local_step": round(e.spiral.local_step, 6),
                "octave": e.octave,
                "degree12_symbol": e.degree12_symbol,
                "degree12_index0": e.degree12_index0,
                "subdivisions": json.dumps(list(e.subdivisions), ensure_ascii=False),
                "delta_radius": e.delta_radius,
                "delta_vector": json.dumps(list(e.delta_vector), ensure_ascii=False),
                "frequency_hz": round(e.frequency_hz, 6),
                "energy": round(e.energy, 9),
            }
        )

    return rows


def trajectories_to_rows(trajectories: list[FastTrajectory]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for t in trajectories:
        rows.append(
            {
                "trajectory_id": t.trajectory_id,
                "dominant_probe_index": t.dominant_probe_index,
                "dominant_note_token": t.dominant_note_token,
                "time_start_sec": round(t.time_start_sec, 6),
                "time_end_sec": round(t.time_end_sec, 6),
                "duration_sec": round(t.duration_sec, 6),
                "event_count": t.count,
                "temporal_density": round(t.temporal_density, 6),
                "mean_spiral_arc": round(t.mean_arc, 6),
                "mean_energy": round(t.mean_energy, 9),
            }
        )

    return rows


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)