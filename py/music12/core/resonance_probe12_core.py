from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import math
import wave

import numpy as np

from music12.core.spiral12_core import BASE12, SpiralCoordinate, cells_per_octave


# ============================================================
# RESONANCE PROBE 12 CORE
# ------------------------------------------------------------
# Собственный front-end проекта.
#
# Идея:
#   WAV
#   -> analytical time grid
#   -> bank of internal resonance probes
#   -> response matrix [probe x time]
#
# ВАЖНО:
# 1. Это НЕ CQT.
# 2. Это НЕ FFT-биновая онтология.
# 3. detail_depth задаёт только глубину детализации:
#       0 -> 12
#       1 -> 144
#       2 -> 1728
#       3 -> 20736
#       ...
# 4. Алгоритм не ограничен сверху 1728.
# 5. Формальный якорь системы:
#       9.A'- = 440 Hz
#    Это чистая опорная точка камертона для эталонной сетки.
#    Реальный якорь WAV-файла может отличаться и вычисляется отдельным этапом.
# ============================================================


# ------------------------------------------------------------
# WAV I/O
# ------------------------------------------------------------

@dataclass(frozen=True)
class WavData:
    sample_rate: int
    samples: np.ndarray  # mono float32 in [-1, 1]


def load_wav_mono(wav_path: str | Path) -> WavData:
    """
    Загружает PCM WAV в mono float32 [-1, 1].

    Поддержка:
    - mono / stereo
    - 16-bit PCM
    - 24-bit PCM
    - 32-bit PCM (integer)
    """
    wav_path = Path(wav_path)

    with wave.open(str(wav_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sample_width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    elif sample_width == 3:
        a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        signed = (
            a[:, 0].astype(np.int32)
            | (a[:, 1].astype(np.int32) << 8)
            | (a[:, 2].astype(np.int32) << 16)
        )
        sign_mask = 1 << 23
        signed = (signed ^ sign_mask) - sign_mask
        data = signed.astype(np.float32) / float(1 << 23)

    elif sample_width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / float(1 << 31)

    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)

    return WavData(
        sample_rate=int(sample_rate),
        samples=data.astype(np.float32),
    )


# ------------------------------------------------------------
# Analytical time grid
# ------------------------------------------------------------

@dataclass(frozen=True)
class TimeGridConfig:
    """
    Аналитическая сетка времени проекта.
    """
    step_seconds: float = 1.0 / 60.0
    window_seconds: float = 1.0 / 20.0
    center_frames: bool = True


@dataclass(frozen=True)
class TimeFrame:
    frame_index: int
    center_time_seconds: float
    start_sample: int
    end_sample: int


def build_time_grid(
    n_samples: int,
    sample_rate: int,
    config: TimeGridConfig | None = None,
) -> list[TimeFrame]:
    config = config or TimeGridConfig()

    if config.step_seconds <= 0:
        raise ValueError("step_seconds must be > 0")
    if config.window_seconds <= 0:
        raise ValueError("window_seconds must be > 0")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    if n_samples < 0:
        raise ValueError("n_samples must be >= 0")

    step_samples = max(1, int(round(config.step_seconds * sample_rate)))
    window_samples = max(8, int(round(config.window_seconds * sample_rate)))
    half_window = window_samples // 2

    total_frames = int(math.ceil(n_samples / step_samples)) if n_samples > 0 else 0
    frames: list[TimeFrame] = []

    for frame_index in range(total_frames):
        center_sample = frame_index * step_samples

        if config.center_frames:
            start = center_sample - half_window
            end = center_sample + half_window
        else:
            start = center_sample
            end = center_sample + window_samples

        start = max(0, start)
        end = min(n_samples, end)

        frames.append(
            TimeFrame(
                frame_index=frame_index,
                center_time_seconds=center_sample / float(sample_rate),
                start_sample=int(start),
                end_sample=int(end),
            )
        )

    return frames


# ------------------------------------------------------------
# Coordinate <-> linear index helpers
# ------------------------------------------------------------

def _digits_base12(value: int, ndigits: int) -> tuple[int, ...]:
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


def local_index_to_subdivisions(local_index: int, detail_depth: int) -> tuple[int, tuple[int, ...]]:
    """
    Перевод из локального индекса внутри октавы в:

      degree12 + ABSOLUTE subdivisions
    """
    if detail_depth < 0:
        raise ValueError("detail_depth must be >= 0")

    digits = _digits_base12(local_index, detail_depth + 1)

    degree12 = digits[0]
    raw_subs = tuple(int(d) for d in digits[1:])

    center = BASE12 // 2  # 6

    if not raw_subs or all(v == center for v in raw_subs):
        return degree12, ()

    return degree12, raw_subs


def coordinate_from_global_probe_index(
    global_index: int,
    *,
    detail_depth: int,
    octave_min: int,
) -> SpiralCoordinate:
    if global_index < 0:
        raise ValueError("global_index must be >= 0")

    cpo = cells_per_octave(detail_depth)
    octave_shift, local_index = divmod(global_index, cpo)
    degree12, subdivisions = local_index_to_subdivisions(local_index, detail_depth)

    return SpiralCoordinate(
        octave=octave_min + octave_shift,
        degree12=degree12,
        subdivisions=tuple(subdivisions),
    ).normalized()


def local_octave_cell_index(coord: SpiralCoordinate, detail_depth: int) -> int:
    """
    Индекс координаты внутри одной октавы для заданной detail_depth.
    """
    if detail_depth < 0:
        raise ValueError("detail_depth must be >= 0")

    c = coord.with_depth(detail_depth).normalized()
    center = BASE12 // 2  # 6

    value = c.degree12 * (BASE12 ** detail_depth)

    abs_subs: list[int] = [center] * detail_depth

    for i, sub in enumerate(c.subdivisions[:detail_depth]):
        sv = int(sub)

        # Только старые отрицательные delta-значения переводим в absolute.
        if sv < 0:
            candidate = center + sv
            if 0 <= candidate < BASE12:
                sv = candidate

        if not 0 <= sv < BASE12:
            raise ValueError(
                f"Subdivision value out of allowed absolute range 0..11 after normalization: {sub!r}"
            )

        abs_subs[i] = sv

    for i, sub in enumerate(abs_subs):
        power = detail_depth - 1 - i
        value += sub * (BASE12 ** power)

    return int(value)


# ------------------------------------------------------------
# Frequency projection
# ------------------------------------------------------------

@dataclass(frozen=True)
class ProbeFrequencyConfig:
    """
    Внутренний канонический мост:
      формальный якорь системы = 9.A'- = 440 Hz
    """
    detail_depth_for_projection: int = 1


def _canonical_anchor_coord() -> SpiralCoordinate:
    """
    Формальный anchor системы:
      9.A'- = 440 Hz
    """
    return SpiralCoordinate(
        octave=9,
        degree12=9,
        subdivisions=(),
    ).normalized()


def _canonical_anchor_frequency_hz() -> float:
    return 440.0


def project_coord_to_frequency_hz(
    coord: SpiralCoordinate,
    freq_cfg: ProbeFrequencyConfig,
) -> float:
    """
    Универсальная внешняя проекция через внутренний формальный якорь:
      9.A'- = 440 Hz
    """
    depth = int(freq_cfg.detail_depth_for_projection)
    if depth < 0:
        raise ValueError("detail_depth_for_projection must be >= 0")

    anchor = _canonical_anchor_coord().with_depth(depth).normalized()
    target = coord.with_depth(depth).normalized()

    cpo = cells_per_octave(depth)

    anchor_step = anchor.octave * cpo + local_octave_cell_index(anchor, depth)
    target_step = target.octave * cpo + local_octave_cell_index(target, depth)

    delta_steps = target_step - anchor_step
    ratio = 2.0 ** (delta_steps / float(cpo))

    return float(_canonical_anchor_frequency_hz() * ratio)


# ------------------------------------------------------------
# Probe shape
# ------------------------------------------------------------

@dataclass(frozen=True)
class ProbeShapeConfig:
    """
    Форма одной резонансной пробы.

    window_type:
      - "hamming"      -> окно Хэмминга
      - "attack_decay" -> старая sin² attack/sustain/decay огибающая
    """
    attack_portion: float = 0.15
    decay_portion: float = 0.20
    harmonic_weights: tuple[float, ...] = (1.0, 0.45, 0.22, 0.10)
    normalize_input_segment: bool = True
    window_type: str = "hamming"


def build_probe_envelope(
    n: int,
    attack_portion: float,
    decay_portion: float,
    window_type: str = "hamming",
) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.float32)

    wt = str(window_type).strip().lower()

    if wt == "hamming":
        return np.hamming(n).astype(np.float32)

    if wt == "attack_decay":
        attack_n = int(round(n * attack_portion))
        decay_n = int(round(n * decay_portion))

        attack_n = max(0, min(attack_n, n))
        decay_n = max(0, min(decay_n, n - attack_n))
        sustain_n = max(0, n - attack_n - decay_n)

        parts = []

        if attack_n > 0:
            x = np.linspace(0.0, 1.0, attack_n, endpoint=False, dtype=np.float32)
            parts.append(np.sin(0.5 * np.pi * x) ** 2)

        if sustain_n > 0:
            parts.append(np.ones(sustain_n, dtype=np.float32))

        if decay_n > 0:
            x = np.linspace(1.0, 0.0, decay_n, endpoint=True, dtype=np.float32)
            parts.append(np.sin(0.5 * np.pi * x) ** 2)

        env = np.concatenate(parts) if parts else np.ones(n, dtype=np.float32)

        if len(env) < n:
            env = np.pad(env, (0, n - len(env)), mode="edge")
        elif len(env) > n:
            env = env[:n]

        return env.astype(np.float32)

    raise ValueError(f"Unsupported window_type: {window_type!r}")


def build_phase_invariant_probe_pair(
    frequency_hz: float,
    sample_rate: int,
    n_samples: int,
    shape_cfg: ProbeShapeConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Строит sin/cos пару для фазово-инвариантного отклика.
    """
    shape_cfg = shape_cfg or ProbeShapeConfig()

    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be > 0")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    t = np.arange(n_samples, dtype=np.float32) / float(sample_rate)
    env = build_probe_envelope(
        n=n_samples,
        attack_portion=shape_cfg.attack_portion,
        decay_portion=shape_cfg.decay_portion,
        window_type=shape_cfg.window_type,
    )

    probe_sin = np.zeros(n_samples, dtype=np.float32)
    probe_cos = np.zeros(n_samples, dtype=np.float32)

    for harmonic_index, weight in enumerate(shape_cfg.harmonic_weights, start=1):
        if weight == 0:
            continue
        f = frequency_hz * harmonic_index
        omega_t = 2.0 * np.pi * f * t
        probe_sin += float(weight) * np.sin(omega_t)
        probe_cos += float(weight) * np.cos(omega_t)

    probe_sin *= env
    probe_cos *= env

    sin_norm = float(np.linalg.norm(probe_sin))
    cos_norm = float(np.linalg.norm(probe_cos))

    if sin_norm > 0:
        probe_sin /= sin_norm
    if cos_norm > 0:
        probe_cos /= cos_norm

    return probe_sin, probe_cos


# ------------------------------------------------------------
# Probe
# ------------------------------------------------------------

@dataclass(frozen=True)
class ResonanceProbe:
    coord: SpiralCoordinate
    frequency_hz: float
    global_index: int

    def response(
        self,
        signal_segment: np.ndarray,
        sample_rate: int,
        shape_cfg: ProbeShapeConfig | None = None,
    ) -> float:
        shape_cfg = shape_cfg or ProbeShapeConfig()

        n = int(len(signal_segment))
        if n <= 0:
            return 0.0

        x = signal_segment.astype(np.float32, copy=False)

        if shape_cfg.normalize_input_segment:
            x_norm = float(np.linalg.norm(x))
            if x_norm > 0:
                x = x / x_norm

        probe_sin, probe_cos = build_phase_invariant_probe_pair(
            frequency_hz=self.frequency_hz,
            sample_rate=sample_rate,
            n_samples=n,
            shape_cfg=shape_cfg,
        )

        a = float(np.dot(x, probe_sin))
        b = float(np.dot(x, probe_cos))

        return math.sqrt(a * a + b * b)


# ------------------------------------------------------------
# Probe bank
# ------------------------------------------------------------

@dataclass(frozen=True)
class ProbeBankConfig:
    octave_min: int
    octave_max: int
    detail_depth: int = 1

    def validate(self) -> None:
        if self.octave_max < self.octave_min:
            raise ValueError("octave_max must be >= octave_min")
        if self.detail_depth < 0:
            raise ValueError("detail_depth must be >= 0")


def build_probe_bank(
    bank_cfg: ProbeBankConfig,
    freq_cfg: ProbeFrequencyConfig,
) -> list[ResonanceProbe]:
    bank_cfg.validate()

    probes: list[ResonanceProbe] = []
    cpo = cells_per_octave(bank_cfg.detail_depth)

    global_index = 0

    for octave in range(bank_cfg.octave_min, bank_cfg.octave_max + 1):
        for local_index in range(cpo):
            degree12, subdivisions = local_index_to_subdivisions(
                local_index=local_index,
                detail_depth=bank_cfg.detail_depth,
            )

            coord = SpiralCoordinate(
                octave=octave,
                degree12=degree12,
                subdivisions=tuple(subdivisions),
            ).normalized()

            f_hz = project_coord_to_frequency_hz(coord, freq_cfg=freq_cfg)

            probes.append(
                ResonanceProbe(
                    coord=coord,
                    frequency_hz=float(f_hz),
                    global_index=global_index,
                )
            )
            global_index += 1

    return probes


# ------------------------------------------------------------
# Response matrix
# ------------------------------------------------------------

@dataclass(frozen=True)
class ProbeResponseMatrix:
    matrix: np.ndarray
    frame_times: np.ndarray
    coords: list[SpiralCoordinate]
    frequencies_hz: np.ndarray
    global_indices: np.ndarray
    detail_depth: int


def compute_probe_response_matrix(
    wav: WavData,
    probes: Iterable[ResonanceProbe],
    time_cfg: TimeGridConfig | None = None,
    shape_cfg: ProbeShapeConfig | None = None,
    *,
    detail_depth: int,
) -> ProbeResponseMatrix:
    probes = list(probes)
    time_cfg = time_cfg or TimeGridConfig()
    shape_cfg = shape_cfg or ProbeShapeConfig()

    frames = build_time_grid(
        n_samples=len(wav.samples),
        sample_rate=wav.sample_rate,
        config=time_cfg,
    )

    n_probes = len(probes)
    n_frames = len(frames)

    matrix = np.zeros((n_probes, n_frames), dtype=np.float32)
    frame_times = np.zeros(n_frames, dtype=np.float32)

    for frame_idx, frame in enumerate(frames):
        frame_times[frame_idx] = frame.center_time_seconds

        segment = wav.samples[frame.start_sample:frame.end_sample]
        if len(segment) == 0:
            continue

        for probe_idx, probe in enumerate(probes):
            matrix[probe_idx, frame_idx] = probe.response(
                signal_segment=segment,
                sample_rate=wav.sample_rate,
                shape_cfg=shape_cfg,
            )

    coords = [p.coord.with_depth(detail_depth).normalized() for p in probes]
    freqs = np.array([p.frequency_hz for p in probes], dtype=np.float32)
    indices = np.array([p.global_index for p in probes], dtype=np.int32)

    return ProbeResponseMatrix(
        matrix=matrix,
        frame_times=frame_times,
        coords=coords,
        frequencies_hz=freqs,
        global_indices=indices,
        detail_depth=int(detail_depth),
    )


# ------------------------------------------------------------
# High-level scan config
# ------------------------------------------------------------

@dataclass(frozen=True)
class ResonanceScanConfig:
    probe_bank: ProbeBankConfig
    probe_frequency: ProbeFrequencyConfig
    time_grid: TimeGridConfig
    probe_shape: ProbeShapeConfig


def scan_wav_with_resonance_probes(
    wav_path: str | Path,
    config: ResonanceScanConfig,
) -> ProbeResponseMatrix:
    wav = load_wav_mono(wav_path)

    probes = build_probe_bank(
        bank_cfg=config.probe_bank,
        freq_cfg=config.probe_frequency,
    )

    return compute_probe_response_matrix(
        wav=wav,
        probes=probes,
        time_cfg=config.time_grid,
        shape_cfg=config.probe_shape,
        detail_depth=config.probe_bank.detail_depth,
    )