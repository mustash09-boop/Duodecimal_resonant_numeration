from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from music12.core.spiral12_core import (
    BASE12,
    NoteCandidate,
    ResonanceCurve,
    SpiralCoordinate,
    SpiralFrame,
    SpiralPoint,
    cells_per_octave,
    chain_points_into_curves,
    infer_note_candidates_from_curves,
)


# ============================================================
# MUSIC12 SPIRAL ANALYSIS
# ------------------------------------------------------------
# Новый слой анализа поверх spiral12_core.
#
# Принципы:
# 1. Внешнее наблюдение -> внутренняя координата.
# 2. Нет жёсткого потолка 144/1728.
# 3. detail_depth задаёт только глубину детализации:
#       0 -> 12
#       1 -> 144
#       2 -> 1728
#       3 -> 20736
#       ...
# 4. Анализ строится через:
#       observed peaks
#       -> SpiralFrame
#       -> ResonanceCurve
#       -> NoteCandidate
# ============================================================


# ------------------------------------------------------------
# External observation model
# ------------------------------------------------------------

@dataclass(frozen=True)
class ObservedPeak:
    """
    Одна наблюдаемая активность из front-end слоя.

    bin_index:
        индекс внутреннего резонансного узла внутри общей
        линейной развёртки банка проб

    time_index:
        индекс временного кадра

    time_seconds:
        внешняя проекция времени

    energy:
        энергия/амплитуда

    confidence:
        уверенность front-end слоя
    """
    time_index: int
    time_seconds: float
    bin_index: int
    energy: float
    confidence: float = 0.0


@dataclass(frozen=True)
class BinMappingConfig:
    """
    Конфигурация перевода линейного bin_index во внутреннюю
    рекурсивную координату.

    detail_depth:
        глубина детализации внутри октавы:
          0 -> 12
          1 -> 144
          2 -> 1728
          ...

    base_octave:
        от какой октавы считать нулевой bin

    degree_offset:
        сдвиг главной ступени

    subdivision_offsets:
        сдвиг для каждого дополнительного 12-ричного слоя
        детализации
    """
    detail_depth: int = 1
    base_octave: int = 0
    degree_offset: int = 0
    subdivision_offsets: tuple[int, ...] = ()

    def normalized(self) -> "BinMappingConfig":
        if self.detail_depth < 0:
            raise ValueError("detail_depth must be >= 0")

        offsets = list(self.subdivision_offsets)
        if len(offsets) < self.detail_depth:
            offsets.extend([0] * (self.detail_depth - len(offsets)))
        elif len(offsets) > self.detail_depth:
            offsets = offsets[:self.detail_depth]

        return BinMappingConfig(
            detail_depth=self.detail_depth,
            base_octave=self.base_octave,
            degree_offset=self.degree_offset,
            subdivision_offsets=tuple(int(x) for x in offsets),
        )


@dataclass
class SpiralAnalysisResult:
    """
    Полный результат анализа.
    """
    frames: list[SpiralFrame] = field(default_factory=list)
    curves: list[ResonanceCurve] = field(default_factory=list)
    note_candidates: list[NoteCandidate] = field(default_factory=list)

    def top_candidate(self) -> Optional[NoteCandidate]:
        return self.note_candidates[0] if self.note_candidates else None


# ------------------------------------------------------------
# Index <-> recursive coordinate
# ------------------------------------------------------------

def _digits_base12(value: int, ndigits: int) -> tuple[int, ...]:
    """
    Разложение value в ndigits цифр по основанию 12.
    """
    if value < 0:
        raise ValueError("value must be >= 0")
    if ndigits < 0:
        raise ValueError("ndigits must be >= 0")

    digits = [0] * ndigits
    x = int(value)

    for i in range(ndigits - 1, -1, -1):
        x, rem = divmod(x, BASE12)
        digits[i] = rem

    if x != 0:
        raise ValueError("value does not fit in ndigits base12 digits")

    return tuple(digits)


def map_bin_to_spiral_coordinate(
    bin_index: int,
    config: BinMappingConfig,
    energy: float = 0.0,
    confidence: float = 0.0,
) -> SpiralCoordinate:
    """
    Переводит линейный bin_index в рекурсивную внутреннюю координату.

    При detail_depth = d:
      cells_per_octave = 12^(d+1)

    Внутри октавы:
      [degree12][sub0][sub1]...[sub(d-1)]
    """
    cfg = config.normalized()

    if bin_index < 0:
        raise ValueError("bin_index must be >= 0")

    cpo = cells_per_octave(cfg.detail_depth)

    octave_shift, local_bin = divmod(int(bin_index), cpo)

    digits = _digits_base12(local_bin, cfg.detail_depth + 1)
    degree12 = digits[0] + cfg.degree_offset
    subdivisions = list(digits[1:])

    for i in range(len(subdivisions)):
        subdivisions[i] += cfg.subdivision_offsets[i]

    coord = SpiralCoordinate(
        octave=cfg.base_octave + octave_shift,
        degree12=degree12,
        subdivisions=tuple(subdivisions),
        energy=float(energy),
        confidence=float(confidence),
    )

    return coord.normalized()


def observed_peak_to_spiral_point(
    peak: ObservedPeak,
    config: BinMappingConfig,
    resonance_score: Optional[float] = None,
) -> SpiralPoint:
    coord = map_bin_to_spiral_coordinate(
        bin_index=peak.bin_index,
        config=config,
        energy=peak.energy,
        confidence=peak.confidence,
    )

    if resonance_score is None:
        resonance_score = float(peak.energy)

    return SpiralPoint(
        time_index=peak.time_index,
        time_seconds=peak.time_seconds,
        coord=coord,
        amplitude=float(peak.energy),
        resonance_score=float(resonance_score),
    )


# ------------------------------------------------------------
# Time grouping
# ------------------------------------------------------------

def group_points_into_frames(points: Iterable[SpiralPoint]) -> list[SpiralFrame]:
    """
    Группировка точек по временным кадрам.
    """
    bucket: dict[int, SpiralFrame] = {}

    for point in points:
        frame = bucket.get(point.time_index)
        if frame is None:
            frame = SpiralFrame(
                time_index=point.time_index,
                time_seconds=point.time_seconds,
                points=[],
            )
            bucket[point.time_index] = frame
        frame.add_point(point)

    return [bucket[k] for k in sorted(bucket.keys())]


def peaks_to_frames(
    peaks: Iterable[ObservedPeak],
    config: BinMappingConfig,
) -> list[SpiralFrame]:
    points = [
        observed_peak_to_spiral_point(peak=peak, config=config)
        for peak in peaks
    ]
    return group_points_into_frames(points)


# ------------------------------------------------------------
# Main analysis pipeline
# ------------------------------------------------------------

def analyze_peaks(
    peaks: Iterable[ObservedPeak],
    config: Optional[BinMappingConfig] = None,
    affinity_threshold: float = 0.45,
    top_k: int = 8,
    curve_depth: Optional[int] = None,
) -> SpiralAnalysisResult:
    """
    Полный pipeline:
      peaks -> frames -> curves -> note candidates

    curve_depth:
        глубина, на которой считать близость/непрерывность
        между соседними координатами.
        Если None, используется максимальная глубина координат.
    """
    cfg = config.normalized() if config is not None else BinMappingConfig().normalized()

    frames = peaks_to_frames(peaks, config=cfg)
    curves = chain_points_into_curves(
        frames,
        affinity_threshold=affinity_threshold,
        depth=curve_depth,
    )
    note_candidates = infer_note_candidates_from_curves(curves, top_k=top_k)

    return SpiralAnalysisResult(
        frames=frames,
        curves=curves,
        note_candidates=note_candidates,
    )


# ------------------------------------------------------------
# Matrix -> peaks
# ------------------------------------------------------------

def response_matrix_to_peaks(
    matrix: np.ndarray,
    times: Optional[np.ndarray] = None,
    energy_threshold: float = 0.0,
    top_n_per_frame: Optional[int] = None,
) -> list[ObservedPeak]:
    """
    Превращает матрицу откликов [bins, frames] в список ObservedPeak.

    Это уже не обязательно CQT-матрица.
    Это может быть любая матрица откликов front-end слоя,
    в том числе нашего resonance-probe front-end.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be numpy.ndarray")

    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix [bins, frames], got shape={matrix.shape}")

    n_bins, n_frames = matrix.shape

    if times is not None:
        if not isinstance(times, np.ndarray):
            raise TypeError("times must be numpy.ndarray or None")
        if len(times) != n_frames:
            raise ValueError(
                f"times length mismatch: len(times)={len(times)} vs n_frames={n_frames}"
            )

    peaks: list[ObservedPeak] = []

    for frame_idx in range(n_frames):
        column = matrix[:, frame_idx]

        candidate_bins = np.where(column > energy_threshold)[0]

        if top_n_per_frame is not None and len(candidate_bins) > top_n_per_frame:
            sorted_bins = sorted(
                candidate_bins.tolist(),
                key=lambda b: float(column[b]),
                reverse=True,
            )
            candidate_bins = np.array(sorted_bins[:top_n_per_frame], dtype=int)

        t = float(times[frame_idx]) if times is not None else float(frame_idx)

        for bin_idx in candidate_bins.tolist():
            energy = float(column[bin_idx])
            peaks.append(
                ObservedPeak(
                    time_index=frame_idx,
                    time_seconds=t,
                    bin_index=int(bin_idx),
                    energy=energy,
                    confidence=energy,
                )
            )

    return peaks


def analyze_response_matrix(
    matrix: np.ndarray,
    times: Optional[np.ndarray] = None,
    *,
    detail_depth: int = 1,
    base_octave: int = 0,
    degree_offset: int = 0,
    subdivision_offsets: tuple[int, ...] = (),
    energy_threshold: float = 0.0,
    top_n_per_frame: Optional[int] = 8,
    affinity_threshold: float = 0.45,
    top_k_notes: int = 8,
    curve_depth: Optional[int] = None,
) -> SpiralAnalysisResult:
    """
    High-level вход:
      response matrix -> peaks -> frames -> curves -> note candidates

    detail_depth задаёт, как интерпретировать bins внутри октавы:
      0 -> 12
      1 -> 144
      2 -> 1728
      ...
    """
    peaks = response_matrix_to_peaks(
        matrix=matrix,
        times=times,
        energy_threshold=energy_threshold,
        top_n_per_frame=top_n_per_frame,
    )

    config = BinMappingConfig(
        detail_depth=detail_depth,
        base_octave=base_octave,
        degree_offset=degree_offset,
        subdivision_offsets=subdivision_offsets,
    ).normalized()

    return analyze_peaks(
        peaks=peaks,
        config=config,
        affinity_threshold=affinity_threshold,
        top_k=top_k_notes,
        curve_depth=curve_depth,
    )


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

def summarize_analysis(result: SpiralAnalysisResult) -> dict:
    top = result.top_candidate()

    return {
        "frames": len(result.frames),
        "curves": len(result.curves),
        "note_candidates": len(result.note_candidates),
        "top_candidate": None
        if top is None
        else {
            "label": top.label,
            "source_score": top.source_score,
            "medium_score": top.medium_score,
            "total_score": top.total_score,
            "anchor": {
                "octave": top.anchor.octave,
                "degree12": top.anchor.degree12,
                "subdivisions": list(top.anchor.subdivisions),
                "depth": top.anchor.depth(),
            },
        },
    }