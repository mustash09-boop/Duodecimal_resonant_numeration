from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np


DEFAULT_HARMONIC_INDICES = [2, 3, 4, 5, 6, 7, 8]
ALPHABET12 = "123456789ABC"


@dataclass(frozen=True)
class ResonanceCoordRow:
    probe_index: int
    octave: int
    degree12: int
    subdivisions: tuple[int, ...]
    frequency_hz: float
    global_index: int
    note_token: str


@dataclass(frozen=True)
class HarmonicSupport:
    harmonic_index: int
    expected_hz: float
    matched_probe_index: Optional[int]
    matched_hz: Optional[float]
    matched_energy: float
    matched_note: str
    is_hit: bool


@dataclass(frozen=True)
class Candidate:
    probe_index: int
    frequency_hz: float
    note_token: str
    energy: float
    supports: tuple[HarmonicSupport, ...]


@dataclass(frozen=True)
class InferenceResult:
    candidates: List[Candidate]


@dataclass(frozen=True)
class MatrixCacheInfo:
    source_csv: str
    cache_dat: str
    cache_meta_json: str
    shape: tuple[int, int]
    dtype: str
    source_size_bytes: int
    source_mtime_ns: int
    reused_existing_cache: bool


def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _symbol_to_degree12(x) -> int:
    s = str(x).strip().upper()

    if s in ALPHABET12:
        return ALPHABET12.index(s)

    try:
        v = int(s)
        if 0 <= v < 12:
            return v
    except Exception:
        pass

    raise ValueError(f"Invalid degree12 symbol: {x!r}")


def _degree_to_symbol(degree12: int) -> str:
    if 0 <= degree12 < 12:
        return ALPHABET12[degree12]
    raise ValueError(f"Invalid degree12 index: {degree12!r}")


def _parse_octave12(x) -> int:
    s = str(x).strip().upper()

    if not s:
        raise ValueError("Empty octave token")

    value = 0
    for ch in s:
        if ch not in ALPHABET12:
            raise ValueError(f"Invalid octave12 symbol: {x!r}")
        value = value * 12 + (ALPHABET12.index(ch) + 1)

    return value


def _octave_to_token12(n: int) -> str:
    if n <= 0:
        raise ValueError(f"Invalid octave number: {n!r}")

    if n <= 12:
        return ALPHABET12[n - 1]

    digits = []
    x = n

    while x > 0:
        r = x % 12

        if r == 0:
            digits.append("C")
            x = x // 12 - 1
        else:
            digits.append(ALPHABET12[r - 1])
            x = x // 12

    return "".join(reversed(digits))


def _subdivisions_to_tuple(raw: str) -> tuple[int, ...]:
    try:
        data = json.loads(raw or "[]")
        if isinstance(data, list):
            return tuple(int(v) for v in data)
    except Exception:
        pass
    return ()


def _coord_to_note_token(octave: int, degree12: int, subdivisions: tuple[int, ...]) -> str:
    base = f"{_octave_to_token12(octave)}.{_degree_to_symbol(degree12)}"

    if not subdivisions:
        return f"{base}'-"

    if all(v == 0 for v in subdivisions):
        return f"{base}'-"

    raise ValueError(
        "Cannot reconstruct micro note_token from raw subdivisions without i/a semantics. "
        "coords.csv must provide explicit note_token for non-center micro coordinates. "
        f"base={base!r}, subdivisions={subdivisions!r}"
    )


def _matrix_csv_shape(path: Path) -> tuple[int, int]:
    probe_count = 0
    frame_count = 0

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)

        if not header:
            return 0, 0

        frame_count = max(0, len(header) - 1)

        for row in reader:
            if len(row) > 1:
                probe_count += 1

    return probe_count, frame_count


def _default_cache_dir_for_matrix(path: Path) -> Path:
    return path.parent / "_matrix_cache"


def _safe_cache_stem(path: Path) -> str:
    return path.stem.replace(" ", "_").replace(".", "_")


def load_matrix_csv(
    path: str | Path,
    *,
    cache_dir: str | Path | None = None,
    force_rebuild_cache: bool = False,
    dtype: str | np.dtype = "float32",
) -> np.ndarray:
    matrix, _info = load_matrix_csv_memmap(
        path,
        cache_dir=cache_dir,
        force_rebuild_cache=force_rebuild_cache,
        dtype=dtype,
    )
    return matrix


def load_matrix_csv_memmap(
    path: str | Path,
    *,
    cache_dir: str | Path | None = None,
    force_rebuild_cache: bool = False,
    dtype: str | np.dtype = "float32",
) -> tuple[np.memmap, MatrixCacheInfo]:
    path = Path(path).resolve()

    if cache_dir is None:
        cache_dir = _default_cache_dir_for_matrix(path)
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    dtype_np = np.dtype(dtype)
    stem = _safe_cache_stem(path)
    dat_path = cache_dir / f"{stem}__{dtype_np.name}.dat"
    meta_path = cache_dir / f"{stem}__{dtype_np.name}.meta.json"

    src_stat = path.stat()
    source_size_bytes = int(src_stat.st_size)
    source_mtime_ns = int(src_stat.st_mtime_ns)

    probe_count, frame_count = _matrix_csv_shape(path)
    shape = (int(probe_count), int(frame_count))

    cache_ok = False

    if dat_path.exists() and meta_path.exists() and not force_rebuild_cache:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            cache_ok = (
                tuple(meta.get("shape", [])) == shape
                and str(meta.get("dtype", "")) == dtype_np.name
                and int(meta.get("source_size_bytes", -1)) == source_size_bytes
                and int(meta.get("source_mtime_ns", -1)) == source_mtime_ns
                and dat_path.stat().st_size == probe_count * frame_count * dtype_np.itemsize
            )
        except Exception:
            cache_ok = False

    if not cache_ok:
        matrix = np.memmap(dat_path, dtype=dtype_np, mode="w+", shape=shape)

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            _header = next(reader, None)

            row_index = 0
            for row in reader:
                if len(row) <= 1:
                    continue

                values = row[1:]

                if len(values) != frame_count:
                    raise ValueError(
                        f"Matrix row length mismatch at row_index={row_index}: "
                        f"expected {frame_count}, got {len(values)}"
                    )

                matrix[row_index, :] = np.asarray(
                    [_safe_float(v, 0.0) for v in values],
                    dtype=dtype_np,
                )

                row_index += 1

        matrix.flush()

        meta_data = {
            "source_csv": str(path),
            "cache_dat": str(dat_path),
            "shape": list(shape),
            "dtype": dtype_np.name,
            "source_size_bytes": source_size_bytes,
            "source_mtime_ns": source_mtime_ns,
            "note": (
                "Disk-backed matrix cache. Safe to delete; it will be rebuilt "
                "from the source CSV."
            ),
        }
        meta_path.write_text(json.dumps(meta_data, ensure_ascii=False, indent=2), encoding="utf-8")
        reused = False
    else:
        reused = True

    matrix = np.memmap(dat_path, dtype=dtype_np, mode="r", shape=shape)

    info = MatrixCacheInfo(
        source_csv=str(path),
        cache_dat=str(dat_path),
        cache_meta_json=str(meta_path),
        shape=shape,
        dtype=dtype_np.name,
        source_size_bytes=source_size_bytes,
        source_mtime_ns=source_mtime_ns,
        reused_existing_cache=reused,
    )

    return matrix, info


def load_times_csv(path: str | Path) -> np.ndarray:
    path = Path(path)

    times = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(_safe_float(row.get("time_seconds", ""), 0.0))

    return np.asarray(times, dtype=np.float32)


def load_coords_csv(path: str | Path) -> List[ResonanceCoordRow]:
    path = Path(path)
    rows: List[ResonanceCoordRow] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            probe_index = _safe_int(row.get("probe_index", ""), len(rows))
            octave = _parse_octave12(row.get("octave", ""))
            degree12 = _symbol_to_degree12(row.get("degree12", ""))
            subdivisions = _subdivisions_to_tuple(row.get("subdivisions", "[]"))
            frequency_hz = _safe_float(row.get("frequency_hz", ""), 0.0)
            global_index = _safe_int(row.get("global_index", ""), probe_index)

            existing_token = str(row.get("note_token", "")).strip()
            rebuilt_token = _coord_to_note_token(octave, degree12, subdivisions)

            if existing_token:
                note_token = existing_token
            else:
                note_token = rebuilt_token

            rows.append(
                ResonanceCoordRow(
                    probe_index=probe_index,
                    octave=octave,
                    degree12=degree12,
                    subdivisions=subdivisions,
                    frequency_hz=frequency_hz,
                    global_index=global_index,
                    note_token=note_token,
                )
            )

    return rows


def _find_support_for_harmonic(
    *,
    base_freq: float,
    harmonic_index: int,
    coords: List[ResonanceCoordRow],
    coord_freqs: np.ndarray,
    frame_values: np.ndarray,
    tolerance_ratio: float,
    analysis_min_hz: float,
    analysis_max_hz: float,
) -> HarmonicSupport:
    expected_hz = base_freq * harmonic_index

    if expected_hz < analysis_min_hz or expected_hz > analysis_max_hz:
        return HarmonicSupport(
            harmonic_index=harmonic_index,
            expected_hz=expected_hz,
            matched_probe_index=None,
            matched_hz=None,
            matched_energy=0.0,
            matched_note="",
            is_hit=False,
        )

    best_idx: Optional[int] = None
    best_energy = -1.0
    tol = expected_hz * tolerance_ratio
    left_hz = expected_hz - tol
    right_hz = expected_hz + tol
    left_idx = int(np.searchsorted(coord_freqs, left_hz, side="left"))
    right_idx = int(np.searchsorted(coord_freqs, right_hz, side="right"))

    if right_idx > len(frame_values):
        right_idx = len(frame_values)

    for i in range(left_idx, right_idx):
        energy = float(frame_values[i])
        if energy > best_energy:
            best_energy = energy
            best_idx = i

    if best_idx is None:
        return HarmonicSupport(
            harmonic_index=harmonic_index,
            expected_hz=expected_hz,
            matched_probe_index=None,
            matched_hz=None,
            matched_energy=0.0,
            matched_note="",
            is_hit=False,
        )

    coord = coords[best_idx]
    return HarmonicSupport(
        harmonic_index=harmonic_index,
        expected_hz=expected_hz,
        matched_probe_index=coord.probe_index,
        matched_hz=coord.frequency_hz,
        matched_energy=float(frame_values[best_idx]),
        matched_note=coord.note_token,
        is_hit=True,
    )


def _build_candidate(
    *,
    coord: ResonanceCoordRow,
    energy: float,
    coords: List[ResonanceCoordRow],
    coord_freqs: np.ndarray,
    frame_values: np.ndarray,
    tolerance_ratio: float,
    analysis_min_hz: float,
    analysis_max_hz: float,
) -> Candidate:
    supports = []

    for h in DEFAULT_HARMONIC_INDICES:
        supports.append(
            _find_support_for_harmonic(
                base_freq=coord.frequency_hz,
                harmonic_index=h,
                coords=coords,
                coord_freqs=coord_freqs,
                frame_values=frame_values,
                tolerance_ratio=tolerance_ratio,
                analysis_min_hz=analysis_min_hz,
                analysis_max_hz=analysis_max_hz,
            )
        )

    return Candidate(
        probe_index=coord.probe_index,
        frequency_hz=coord.frequency_hz,
        note_token=coord.note_token,
        energy=float(energy),
        supports=tuple(supports),
    )


def scan_frame_candidates(
    *,
    matrix: np.ndarray,
    times: np.ndarray,
    coords: List[ResonanceCoordRow],
    coord_freqs: np.ndarray,
    frame_index: int,
    energy_threshold: float,
    top_n_candidates: int,
    tolerance_ratio: float,
    analysis_min_hz: float,
    analysis_max_hz: float,
    max_polyphonic_candidates: int,
) -> List[Candidate]:
    if matrix.size == 0:
        return []

    if frame_index < 0 or frame_index >= matrix.shape[1]:
        return []

    frame_values = np.asarray(matrix[:, frame_index], dtype=np.float32)
    raw = []
    hit_indices = np.flatnonzero(frame_values >= energy_threshold)

    for i in hit_indices.tolist():
        if i >= len(coords):
            continue

        coord = coords[i]
        if coord.frequency_hz < analysis_min_hz or coord.frequency_hz > analysis_max_hz:
            continue

        raw.append((float(frame_values[i]), i, coord))

    if not raw:
        return []

    raw.sort(key=lambda x: x[0], reverse=True)
    raw = raw[:top_n_candidates]

    candidates = []
    for energy, _, coord in raw:
        candidates.append(
            _build_candidate(
                coord=coord,
                energy=energy,
                coords=coords,
                coord_freqs=coord_freqs,
                frame_values=frame_values,
                tolerance_ratio=tolerance_ratio,
                analysis_min_hz=analysis_min_hz,
                analysis_max_hz=analysis_max_hz,
            )
        )

    return candidates[:max_polyphonic_candidates]


def iter_frame_candidates(
    *,
    matrix: np.ndarray,
    times: np.ndarray,
    coords: List[ResonanceCoordRow],
    energy_threshold: float,
    top_n_candidates: int,
    tolerance_ratio: float,
    analysis_min_hz: float,
    analysis_max_hz: float,
    max_polyphonic_candidates: int,
    start_frame: int = 0,
    stop_frame: int | None = None,
) -> Iterator[tuple[int, float, List[Candidate]]]:
    frame_count = min(matrix.shape[1] if len(matrix.shape) == 2 else 0, len(times))
    start = max(0, int(start_frame))
    stop_exclusive = frame_count if stop_frame is None else min(frame_count, int(stop_frame) + 1)
    coord_freqs = np.asarray([c.frequency_hz for c in coords], dtype=np.float64)

    for frame_index in range(start, stop_exclusive):
        yield (
            frame_index,
            float(times[frame_index]),
            scan_frame_candidates(
                matrix=matrix,
                times=times,
                coords=coords,
                coord_freqs=coord_freqs,
                frame_index=frame_index,
                energy_threshold=energy_threshold,
                top_n_candidates=top_n_candidates,
                tolerance_ratio=tolerance_ratio,
                analysis_min_hz=analysis_min_hz,
                analysis_max_hz=analysis_max_hz,
                max_polyphonic_candidates=max_polyphonic_candidates,
            ),
        )


def infer_candidates(raw_candidates: List[Candidate]) -> InferenceResult:
    if not raw_candidates:
        return InferenceResult(candidates=[])

    sorted_candidates = sorted(
        raw_candidates,
        key=lambda c: c.energy,
        reverse=True,
    )

    return InferenceResult(candidates=sorted_candidates)
