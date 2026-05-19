from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence


# ============================================================
# DUODECIMAL SPIRAL CORE
# ------------------------------------------------------------
# Общий спиральный слой проекта.
#
# ВАЖНО:
#   Здесь НЕТ жёсткого потолка 144 / 1728.
#   Есть только рекурсивная 12-ричная детализация.
#
# depth = 0  -> 12
# depth = 1  -> 144
# depth = 2  -> 1728
# depth = 3  -> 20736
# ...
#
# ДОПОЛНЕНИЕ:
#   Помимо дискретной адресации координаты, здесь теперь есть
#   явный геометрический слой:
#   - phase_turn / phase_angle_deg / phase_angle_rad
#   - radial_level / geometric_radius
#   - coord_to_xy
#   - frequency_to_spiral_position
#
# Это не ломает старую логику, а добавляет наружу геометрию
# уже существующей 12-ричной спиральной координаты.
# ============================================================

BASE12 = 12

ANGLE_FULL_TURN_DEG = 360.0
ANGLE_PER_STEP_DEG = ANGLE_FULL_TURN_DEG / BASE12

# Логарифмический шаг 12-ричной музыкальной спирали
OMEGA_12 = 2.0 ** (1.0 / 12.0)

# Формальный якорь проекта:
# note 9.A = 440 Hz
DEFAULT_ANCHOR_FREQUENCY_HZ = 440.0
DEFAULT_ANCHOR_OCTAVE = 9
DEFAULT_ANCHOR_DEGREE12 = 9


def cells_per_octave(depth: int) -> int:
    """
    depth=0 -> 12
    depth=1 -> 144
    depth=2 -> 1728
    """
    if depth < 0:
        raise ValueError("depth must be >= 0")
    return BASE12 ** (depth + 1)


@dataclass(frozen=True)
class SpiralCoordinate:
    """
    Внутренняя координата в рекурсивной 12-ричной системе.

    octave:
        уровень/виток

    degree12:
        основная ступень 0..11

    subdivisions:
        дополнительные слои 12-ричной детализации.
        Примеры:
          ()         -> 12
          (5,)       -> 144
          (5, 2)     -> 1728
          (5, 2, 9)  -> 20736
    """
    octave: int
    degree12: int
    subdivisions: tuple[int, ...] = ()
    energy: float = 0.0
    confidence: float = 0.0

    def depth(self) -> int:
        return len(self.subdivisions)

    def normalized(self) -> "SpiralCoordinate":
        """
        Нормализация всех слоёв с переносами по основанию 12.
        """
        octave = int(self.octave)
        degree12 = int(self.degree12)
        subs = list(int(x) for x in self.subdivisions)

        # Нормализуем subdivisions с конца к началу
        for i in range(len(subs) - 1, -1, -1):
            value = subs[i]
            carry, rem = divmod(value, BASE12)
            subs[i] = rem
            if i == 0:
                degree12 += carry
            else:
                subs[i - 1] += carry

        degree_carry, degree_rem = divmod(degree12, BASE12)
        degree12 = degree_rem
        octave += degree_carry

        return SpiralCoordinate(
            octave=octave,
            degree12=degree12,
            subdivisions=tuple(subs),
            energy=self.energy,
            confidence=self.confidence,
        )

    def with_depth(self, target_depth: int) -> "SpiralCoordinate":
        """
        Приводит координату к нужной глубине:
        - если глубина меньше -> дополняем центром (=6)
        - если больше -> обрезаем хвост

        ВАЖНО:
        Пустая subdivision-координата означает "чистый центр",
        а не нулевую ячейку. Поэтому дополняем не нулями, а 6.
        """
        if target_depth < 0:
            raise ValueError("target_depth must be >= 0")

        c = self.normalized()
        subs = list(c.subdivisions)
        center = BASE12 // 2  # 6

        if len(subs) < target_depth:
            subs.extend([center] * (target_depth - len(subs)))
        elif len(subs) > target_depth:
            subs = subs[:target_depth]

        return SpiralCoordinate(
            octave=c.octave,
            degree12=c.degree12,
            subdivisions=tuple(subs),
            energy=c.energy,
            confidence=c.confidence,
        )

    def linear_step(self, depth: Optional[int] = None) -> int:
        """
        Линейный индекс в выбранной глубине.

        depth=None -> использовать собственную глубину координаты
        """
        c = self.normalized()
        if depth is None:
            depth = c.depth()

        c = c.with_depth(depth)

        value = c.octave * cells_per_octave(depth)
        value += c.degree12 * (BASE12 ** depth)

        for i, sub in enumerate(c.subdivisions):
            power = depth - 1 - i
            value += sub * (BASE12 ** power)

        return value


# ============================================================
# GEOMETRIC LAYER
# ------------------------------------------------------------
# Явная геометрия спирали поверх уже существующей дискретной
# координаты.
# ============================================================

def _normalized_fractional_step(coord: SpiralCoordinate, depth: int | None = None) -> float:
    """
    Дробная позиция внутри октавы:

        degree12 + sub1/12 + sub2/12^2 + ...

    Примеры:
      degree12=5, subdivisions=()       -> 5
      degree12=5, subdivisions=(2,)     -> 5 + 2/12
      degree12=5, subdivisions=(2, 9)   -> 5 + 2/12 + 9/144
    """
    c = coord.normalized()

    if depth is None:
        depth = c.depth()

    c = c.with_depth(depth)

    value = float(c.degree12)
    for i, sub in enumerate(c.subdivisions, start=1):
        value += float(sub) / (BASE12 ** i)

    return value


def phase_turn(coord: SpiralCoordinate, depth: int | None = None) -> float:
    """
    Фаза как доля полного оборота [0, 1).
    """
    frac = _normalized_fractional_step(coord, depth=depth)
    return frac / BASE12


def phase_angle_deg(coord: SpiralCoordinate, depth: int | None = None) -> float:
    """
    Фаза в градусах [0, 360).
    """
    return phase_turn(coord, depth=depth) * ANGLE_FULL_TURN_DEG


def phase_angle_rad(coord: SpiralCoordinate, depth: int | None = None) -> float:
    """
    Фаза в радианах [0, 2π).
    """
    return phase_turn(coord, depth=depth) * (2.0 * math.pi)


def radial_level(coord: SpiralCoordinate, depth: int | None = None) -> float:
    """
    Радиальный уровень как виток + дробная внутривитковая поправка.

    Это внутренняя спиральная координата радиуса,
    а не обязательно физический радиус в плоскости.
    """
    c = coord.normalized()
    frac = _normalized_fractional_step(c, depth=depth) / BASE12
    return float(c.octave) + frac


def geometric_radius(
    coord: SpiralCoordinate,
    radius_base: float = 1.0,
    radius_growth_per_octave: float = OMEGA_12 ** BASE12,
    depth: int | None = None,
) -> float:
    """
    Геометрический радиус спирали.

    По умолчанию:
        radius_growth_per_octave = OMEGA_12 ** 12 = 2.0

    То есть каждый полный виток (октава) удваивает масштаб.
    """
    return radius_base * (radius_growth_per_octave ** radial_level(coord, depth=depth))


def coord_to_xy(
    coord: SpiralCoordinate,
    radius_base: float = 1.0,
    radius_growth_per_octave: float = OMEGA_12 ** BASE12,
    depth: int | None = None,
) -> tuple[float, float]:
    """
    Перевод спиральной координаты в 2D-плоскость.
    """
    theta = phase_angle_rad(coord, depth=depth)
    r = geometric_radius(
        coord,
        radius_base=radius_base,
        radius_growth_per_octave=radius_growth_per_octave,
        depth=depth,
    )
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return x, y


def phase_distance_steps(
    a: SpiralCoordinate,
    b: SpiralCoordinate,
    depth: int | None = None,
) -> float:
    """
    Минимальное расстояние по фазе внутри одного оборота
    в единицах шагов выбранной глубины.

    Например:
      depth=0 -> 12 шагов на оборот
      depth=1 -> 144 шагов на оборот
    """
    if depth is None:
        depth = max(a.depth(), b.depth())

    a_turn = phase_turn(a, depth=depth)
    b_turn = phase_turn(b, depth=depth)

    diff_turn = abs(a_turn - b_turn)
    diff_turn = min(diff_turn, 1.0 - diff_turn)

    return diff_turn * cells_per_octave(depth)


def phase_distance_deg(
    a: SpiralCoordinate,
    b: SpiralCoordinate,
    depth: int | None = None,
) -> float:
    """
    Минимальное угловое расстояние в градусах.
    """
    if depth is None:
        depth = max(a.depth(), b.depth())

    a_deg = phase_angle_deg(a, depth=depth)
    b_deg = phase_angle_deg(b, depth=depth)

    diff = abs(a_deg - b_deg) % ANGLE_FULL_TURN_DEG
    return min(diff, ANGLE_FULL_TURN_DEG - diff)


def radial_distance(
    a: SpiralCoordinate,
    b: SpiralCoordinate,
    depth: int | None = None,
) -> float:
    """
    Расстояние по радиальному уровню.
    """
    return abs(radial_level(a, depth=depth) - radial_level(b, depth=depth))


def spiral_geometric_distance(
    a: SpiralCoordinate,
    b: SpiralCoordinate,
    radius_base: float = 1.0,
    radius_growth_per_octave: float = OMEGA_12 ** BASE12,
    depth: int | None = None,
) -> float:
    """
    Евклидово расстояние между двумя точками
    в геометрическом 2D-представлении спирали.
    """
    ax, ay = coord_to_xy(
        a,
        radius_base=radius_base,
        radius_growth_per_octave=radius_growth_per_octave,
        depth=depth,
    )
    bx, by = coord_to_xy(
        b,
        radius_base=radius_base,
        radius_growth_per_octave=radius_growth_per_octave,
        depth=depth,
    )
    return math.hypot(ax - bx, ay - by)


def frequency_to_spiral_position(
    freq_hz: float,
    anchor_hz: float = DEFAULT_ANCHOR_FREQUENCY_HZ,
) -> tuple[float, float]:
    """
    Частота -> (radial_level, phase_turn) в непрерывной спиральной геометрии.

    Возвращает:
      radial_level : непрерывный радиальный уровень
      phase_turn   : доля полного оборота [0, 1)
    """
    if freq_hz <= 0.0:
        raise ValueError("freq_hz must be > 0")
    if anchor_hz <= 0.0:
        raise ValueError("anchor_hz must be > 0")

    semitone_offset = 12.0 * math.log2(freq_hz / anchor_hz)

    absolute_step = (
        float(DEFAULT_ANCHOR_OCTAVE) * BASE12
        + float(DEFAULT_ANCHOR_DEGREE12)
        + semitone_offset
    )

    radial = absolute_step / BASE12
    phase = (absolute_step % BASE12) / BASE12
    return radial, phase


def frequency_to_phase_turn(
    freq_hz: float,
    anchor_hz: float = DEFAULT_ANCHOR_FREQUENCY_HZ,
) -> float:
    """
    Частота -> фаза [0, 1)
    """
    _, phase = frequency_to_spiral_position(freq_hz=freq_hz, anchor_hz=anchor_hz)
    return phase


def frequency_to_phase_angle_deg(
    freq_hz: float,
    anchor_hz: float = DEFAULT_ANCHOR_FREQUENCY_HZ,
) -> float:
    """
    Частота -> угол фазы в градусах.
    """
    return frequency_to_phase_turn(freq_hz=freq_hz, anchor_hz=anchor_hz) * ANGLE_FULL_TURN_DEG


# ============================================================
# DISCRETE / LEGACY SPIRAL LOGIC
# ------------------------------------------------------------
# Старый служебный слой сохраняется без ломки:
# - линейные шаги
# - continuity
# - chaining
# - note candidates
# ============================================================

@dataclass(frozen=True)
class SpiralPoint:
    time_index: int
    time_seconds: float
    coord: SpiralCoordinate
    amplitude: float = 0.0
    resonance_score: float = 0.0


@dataclass
class ResonanceCurve:
    points: list[SpiralPoint] = field(default_factory=list)
    source_hint: Optional[str] = None
    medium_hint: Optional[str] = None
    stability_score: float = 0.0
    continuity_score: float = 0.0

    def add(self, point: SpiralPoint) -> None:
        self.points.append(point)

    def is_empty(self) -> bool:
        return len(self.points) == 0

    def first_point(self) -> Optional[SpiralPoint]:
        return self.points[0] if self.points else None

    def last_point(self) -> Optional[SpiralPoint]:
        return self.points[-1] if self.points else None

    def mean_amplitude(self) -> float:
        if not self.points:
            return 0.0
        return sum(p.amplitude for p in self.points) / len(self.points)

    def mean_resonance_score(self) -> float:
        if not self.points:
            return 0.0
        return sum(p.resonance_score for p in self.points) / len(self.points)

    def time_span(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return self.points[-1].time_seconds - self.points[0].time_seconds


@dataclass(frozen=True)
class NoteCandidate:
    label: str
    anchor: SpiralCoordinate
    source_score: float
    medium_score: float
    total_score: float


@dataclass
class SpiralFrame:
    time_index: int
    time_seconds: float
    points: list[SpiralPoint] = field(default_factory=list)

    def add_point(self, point: SpiralPoint) -> None:
        self.points.append(point)

    def strongest_point(self) -> Optional[SpiralPoint]:
        if not self.points:
            return None
        return max(self.points, key=lambda p: (p.resonance_score, p.amplitude))


def spiral_distance(a: SpiralCoordinate, b: SpiralCoordinate, depth: Optional[int] = None) -> int:
    if depth is None:
        depth = max(a.depth(), b.depth())
    return abs(a.linear_step(depth=depth) - b.linear_step(depth=depth))


def local_continuity_score(
    prev_coord: SpiralCoordinate,
    next_coord: SpiralCoordinate,
    depth: Optional[int] = None,
) -> float:
    d = spiral_distance(prev_coord, next_coord, depth=depth)
    return 1.0 / (1.0 + float(d))


def local_resonance_affinity(
    a: SpiralPoint,
    b: SpiralPoint,
    depth: Optional[int] = None,
) -> float:
    continuity = local_continuity_score(a.coord, b.coord, depth=depth)

    amp_max = max(a.amplitude, b.amplitude)
    amp_balance = min(a.amplitude, b.amplitude) / amp_max if amp_max > 0 else 0.0

    res_max = max(a.resonance_score, b.resonance_score)
    res_balance = min(a.resonance_score, b.resonance_score) / res_max if res_max > 0 else 0.0

    return (continuity + amp_balance + res_balance) / 3.0


def chain_points_into_curves(
    frames: Iterable[SpiralFrame],
    affinity_threshold: float = 0.45,
    depth: Optional[int] = None,
) -> list[ResonanceCurve]:
    curves: list[ResonanceCurve] = []
    active_curves: list[ResonanceCurve] = []

    for frame in frames:
        new_active: list[ResonanceCurve] = []

        for point in frame.points:
            best_curve = None
            best_score = -1.0

            for curve in active_curves:
                last_point = curve.last_point()
                if last_point is None:
                    continue
                score = local_resonance_affinity(last_point, point, depth=depth)
                if score > best_score:
                    best_score = score
                    best_curve = curve

            if best_curve is not None and best_score >= affinity_threshold:
                best_curve.add(point)
                new_active.append(best_curve)
            else:
                curve = ResonanceCurve(points=[point])
                curves.append(curve)
                new_active.append(curve)

        active_curves = new_active

    for curve in curves:
        curve.continuity_score = _estimate_curve_continuity(curve, depth=depth)
        curve.stability_score = _estimate_curve_stability(curve)

    return curves


def _estimate_curve_continuity(curve: ResonanceCurve, depth: Optional[int] = None) -> float:
    if len(curve.points) < 2:
        return 0.0

    scores = []
    for i in range(1, len(curve.points)):
        scores.append(
            local_continuity_score(
                curve.points[i - 1].coord,
                curve.points[i].coord,
                depth=depth,
            )
        )
    return sum(scores) / len(scores)


def _estimate_curve_stability(curve: ResonanceCurve) -> float:
    if not curve.points:
        return 0.0

    mean_res = curve.mean_resonance_score()
    mean_amp = curve.mean_amplitude()
    span = curve.time_span()
    return (mean_res * 0.5) + (mean_amp * 0.3) + (span * 0.2)


def curve_source_score(curve: ResonanceCurve) -> float:
    if curve.is_empty():
        return 0.0
    return (
        curve.stability_score * 0.4
        + curve.continuity_score * 0.4
        + curve.mean_amplitude() * 0.2
    )


def curve_medium_score(curve: ResonanceCurve) -> float:
    if curve.is_empty():
        return 0.0
    return max(0.0, 1.0 - ((curve.stability_score + curve.continuity_score) / 2.0))


def infer_note_candidates_from_curves(
    curves: Iterable[ResonanceCurve],
    top_k: int = 8,
) -> list[NoteCandidate]:
    result: list[NoteCandidate] = []

    for curve in curves:
        if curve.is_empty():
            continue

        anchor_point = curve.first_point()
        if anchor_point is None:
            continue

        source_score = curve_source_score(curve)
        medium_score = curve_medium_score(curve)
        total_score = source_score - 0.35 * medium_score

        label = technical_label_from_coordinate(anchor_point.coord)

        result.append(
            NoteCandidate(
                label=label,
                anchor=anchor_point.coord.normalized(),
                source_score=source_score,
                medium_score=medium_score,
                total_score=total_score,
            )
        )

    result.sort(key=lambda x: x.total_score, reverse=True)
    return result[:top_k]


def technical_label_from_coordinate(coord: SpiralCoordinate) -> str:
    c = coord.normalized()
    if not c.subdivisions:
        return f"{c.octave}.{c.degree12}"
    sub = ".".join(str(x) for x in c.subdivisions)
    return f"{c.octave}.{c.degree12}<{sub}>"


def try_notation12_label(coord: SpiralCoordinate) -> str:
    return technical_label_from_coordinate(coord)