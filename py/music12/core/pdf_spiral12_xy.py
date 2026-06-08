from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any


LOG2 = math.log(2.0)
OMEGA12 = 2.0 ** (1.0 / 12.0)
STEPS_PER_OCTAVE = 12.0

DIGITS12 = "123456789ABC"
STEPS = list(DIGITS12)
STEP_INDEX = {s: i for i, s in enumerate(STEPS)}
_VAL12 = {ch: i + 1 for i, ch in enumerate(DIGITS12)}
_CH12 = {i + 1: ch for i, ch in enumerate(DIGITS12)}


@dataclass(frozen=True)
class PdfSpiralXYPosition:
    note_token: str
    semitone_offset: float
    abs_step_float: float
    octave_float: float
    degree12_float: float
    phase12_deg: float
    phase12_rad: float
    radial_level: float
    x12: float
    y12: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_token_text(s: str) -> str:
    return (
        (s or "")
        .replace("Рђ", "A")
        .replace("Р’", "B")
        .replace("РЎ", "C")
        .replace("Р С’", "A")
        .replace("Р вЂ™", "B")
        .replace("Р РЋ", "C")
        .upper()
        .strip()
    )


def bij12_to_int(s: str) -> int:
    s = normalize_token_text(s)
    n = 0
    for ch in s:
        if ch not in _VAL12:
            raise ValueError(f"Bad 12 digit: {ch!r}")
        n = n * 12 + _VAL12[ch]
    return n


def int_to_bij12(n: int) -> str:
    n = int(n)
    if n <= 0:
        return ""
    out = []
    while n > 0:
        n, r = divmod(n - 1, 12)
        out.append(_CH12[r + 1])
    return "".join(reversed(out))


def int_to_base12_digit(i0: int) -> str:
    return _CH12[i0 + 1]


def parse_token(token: str) -> tuple[str, str]:
    tok = normalize_token_text(token).replace("'", "").rstrip("-")
    if "." not in tok:
        raise ValueError(f"Bad note token: {tok!r}")
    oct_s, step_s = tok.split(".", 1)
    return oct_s, step_s[:1]


def token_to_abs_step(token: str) -> int:
    oct_s, step = parse_token(token)
    octave0 = bij12_to_int(oct_s) - 1
    degree0 = _VAL12[step] - 1
    return octave0 * 12 + degree0


def abs_step_to_token(abs_step: int, micro: str = "-") -> str:
    octave0, degree0 = divmod(int(abs_step), 12)
    base = f"{int_to_bij12(octave0 + 1)}.{int_to_base12_digit(degree0)}"
    return f"{base}'{micro}" if micro else base


def hz_to_token_with_micro(
    freq_hz: float,
    *,
    anchor_token: str,
    anchor_hz: float,
    micro_steps_per_semitone: int = 12,
) -> str:
    if freq_hz <= 0:
        return ""

    anchor_abs = token_to_abs_step(anchor_token)
    semitone_offset = 12.0 * math.log2(freq_hz / anchor_hz)

    nearest = int(round(semitone_offset))
    residual = semitone_offset - nearest
    abs_step = anchor_abs + nearest

    micro = int(round(residual * micro_steps_per_semitone))
    if micro == 0:
        return abs_step_to_token(abs_step, "-")

    sign = "i" if micro > 0 else "a"
    mag = abs(micro)

    while mag >= micro_steps_per_semitone:
        abs_step += 1 if sign == "i" else -1
        mag -= micro_steps_per_semitone

    if mag == 0:
        return abs_step_to_token(abs_step, "-")

    return abs_step_to_token(abs_step, f"{sign}{int_to_base12_digit(mag)}")


@lru_cache(maxsize=1)
def phi_sequence_degrees() -> tuple[float, ...]:
    return tuple(360.0 / (OMEGA12**n) for n in range(13))


@lru_cache(maxsize=1)
def pdf_sector_angles_deg() -> tuple[float, ...]:
    phi = phi_sequence_degrees()
    return tuple(2.0 * (phi[n] - phi[n + 1]) for n in range(12))


@lru_cache(maxsize=1)
def pdf_cumulative_angles_deg() -> tuple[float, ...]:
    acc = 0.0
    out = [0.0]
    for a in pdf_sector_angles_deg():
        acc += a
        out.append(acc)
    return tuple(out)


@lru_cache(maxsize=1)
def pdf_a_offset_deg() -> float:
    return pdf_cumulative_angles_deg()[STEP_INDEX["A"]]


def angle_for_step(step: str) -> float:
    return (pdf_cumulative_angles_deg()[STEP_INDEX[step]] - pdf_a_offset_deg()) % 360.0


def continuous_angle_from_relative_step(relative_step_float: float) -> float:
    """
    Exact continuous angle from Joe's corrected spiral.

    relative_step_float:
        semitone distance from anchor A-position.
    """
    base = STEP_INDEX["A"] + relative_step_float
    turns = math.floor(base / STEPS_PER_OCTAVE)
    local = base % STEPS_PER_OCTAVE

    i = int(math.floor(local))
    frac = local - i

    if i >= 12:
        i = 11
        frac = 1.0

    cum = pdf_cumulative_angles_deg()
    a0 = cum[i]
    a1 = cum[i + 1]
    return (a0 + (a1 - a0) * frac - pdf_a_offset_deg()) + 360.0 * turns


def pdf_radius_from_semitone_offset(semitone_offset: float) -> float:
    return OMEGA12 ** semitone_offset


def _empty_position() -> PdfSpiralXYPosition:
    return PdfSpiralXYPosition(
        note_token="",
        semitone_offset=0.0,
        abs_step_float=0.0,
        octave_float=0.0,
        degree12_float=0.0,
        phase12_deg=0.0,
        phase12_rad=0.0,
        radial_level=0.0,
        x12=0.0,
        y12=0.0,
    )


def pdf_spiral_xy_from_frequency(
    freq_hz: float,
    *,
    anchor_token: str,
    anchor_hz: float,
) -> PdfSpiralXYPosition:
    if freq_hz <= 0:
        return _empty_position()

    anchor_abs = float(token_to_abs_step(anchor_token))
    semitone_offset = 12.0 * math.log(freq_hz / anchor_hz) / LOG2
    abs_step_float = anchor_abs + semitone_offset

    octave_float = math.floor(abs_step_float / STEPS_PER_OCTAVE)
    degree12_float = semitone_offset % STEPS_PER_OCTAVE

    phase12_deg = continuous_angle_from_relative_step(semitone_offset)
    phase12_rad = math.radians(phase12_deg)
    radial_level = pdf_radius_from_semitone_offset(semitone_offset)

    return PdfSpiralXYPosition(
        note_token=hz_to_token_with_micro(freq_hz, anchor_token=anchor_token, anchor_hz=anchor_hz),
        semitone_offset=semitone_offset,
        abs_step_float=abs_step_float,
        octave_float=octave_float,
        degree12_float=degree12_float,
        phase12_deg=phase12_deg,
        phase12_rad=phase12_rad,
        radial_level=radial_level,
        x12=radial_level * math.cos(phase12_rad),
        y12=radial_level * math.sin(phase12_rad),
    )


def pdf_spiral_xy_from_token(
    token: str,
    *,
    anchor_token: str,
) -> PdfSpiralXYPosition | None:
    if not token:
        return None

    abs_step_float = float(token_to_abs_step(token))
    anchor_abs = float(token_to_abs_step(anchor_token))
    semitone_offset = abs_step_float - anchor_abs

    octave_float = math.floor(abs_step_float / STEPS_PER_OCTAVE)
    degree12_float = semitone_offset % STEPS_PER_OCTAVE
    phase12_deg = continuous_angle_from_relative_step(semitone_offset)
    phase12_rad = math.radians(phase12_deg)
    radial_level = pdf_radius_from_semitone_offset(semitone_offset)

    return PdfSpiralXYPosition(
        note_token=token,
        semitone_offset=semitone_offset,
        abs_step_float=abs_step_float,
        octave_float=octave_float,
        degree12_float=degree12_float,
        phase12_deg=phase12_deg,
        phase12_rad=phase12_rad,
        radial_level=radial_level,
        x12=radial_level * math.cos(phase12_rad),
        y12=radial_level * math.sin(phase12_rad),
    )
