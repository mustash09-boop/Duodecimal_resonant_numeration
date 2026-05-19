# -*- coding: utf-8 -*-
"""
music12.core.spiral12_geometry

TRUE duodecimal spiral geometry.

No phase/radial hacks.
No linear approximations.
No "degree * 30".

Everything is expressed as motion along a logarithmic spiral.

Core idea:
    position = arc on spiral
    interval = delta along spiral
    harmony = consistency of deltas

This module must become the ONLY geometry used in Block002+.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


# ============================================================
# CONSTANTS
# ============================================================

LOG2 = math.log(2.0)

# one octave = 12 steps in your system
STEPS_PER_OCTAVE = 12.0

# tolerance for floating comparisons
EPS = 1e-9


# ============================================================
# DATA STRUCTURE
# ============================================================

@dataclass(frozen=True)
class SpiralPosition:
    """
    Continuous position on duodecimal spiral.

    absolute_arc:
        continuous coordinate along spiral (in 12-based units)

    turn_index:
        integer spiral turn (octave index)

    local_step:
        position inside the turn (0..12)

    NOTE:
        There is NO "phase" or "radius" here.
        Those are derived, not primary.
    """

    absolute_arc: float
    turn_index: int
    local_step: float

    def distance_to(self, other: "SpiralPosition") -> float:
        """Distance along spiral (signed)."""
        return other.absolute_arc - self.absolute_arc

    def abs_distance(self, other: "SpiralPosition") -> float:
        return abs(self.distance_to(other))


# ============================================================
# TOKEN → SPIRAL
# ============================================================

def duodecimal_digit_to_int(ch: str) -> int:
    ch = ch.strip().upper()
    mapping = {
        "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
        "7": 7, "8": 8, "9": 9, "A": 10, "B": 11, "C": 12,
    }
    if ch not in mapping:
        raise ValueError(f"Invalid duodecimal digit: {ch}")
    return mapping[ch]


def duodecimal_str_to_int(s: str) -> int:
    s = s.strip().upper()
    if not s:
        raise ValueError("Empty duodecimal string")

    value = 0
    for ch in s:
        value = value * 12 + duodecimal_digit_to_int(ch)
    return value


def parse_token_to_spiral(token: str) -> Optional[SpiralPosition]:
    """
    Convert token like:
        6.4
        6.4-
        A.B
        11.1
    into SpiralPosition.

    This is NOT geometry approximation.
    This is mapping into continuous arc space.
    """
    if not token:
        return None

    token = str(token).strip().replace(" ", "")

    import re
    m = re.match(r"^([1-9ABC]+)\.([1-9ABC]+)", token, flags=re.IGNORECASE)
    if not m:
        return None

    octave = duodecimal_str_to_int(m.group(1))
    degree = duodecimal_str_to_int(m.group(2))

    # continuous coordinate:
    # each degree is 1 step
    absolute_arc = octave * STEPS_PER_OCTAVE + degree

    turn_index = octave
    local_step = float(degree)

    return SpiralPosition(
        absolute_arc=absolute_arc,
        turn_index=turn_index,
        local_step=local_step,
    )


# ============================================================
# FREQUENCY → SPIRAL
# ============================================================

def frequency_to_spiral(frequency_hz: float, a4_hz: float = 440.0) -> SpiralPosition:
    """
    Map real frequency to spiral position.

    Uses logarithmic relation:
        log2(f / A4) * 12

    This is the TRUE physical mapping.
    """
    if frequency_hz <= 0:
        raise ValueError("frequency must be positive")

    # distance in octaves
    octaves = math.log(frequency_hz / a4_hz) / LOG2

    # convert to duodecimal steps
    absolute_arc = octaves * STEPS_PER_OCTAVE

    turn_index = int(math.floor(absolute_arc / STEPS_PER_OCTAVE))
    local_step = absolute_arc - (turn_index * STEPS_PER_OCTAVE)

    return SpiralPosition(
        absolute_arc=absolute_arc,
        turn_index=turn_index,
        local_step=local_step,
    )


# ============================================================
# RELATIONS
# ============================================================

def is_same_position(a: SpiralPosition, b: SpiralPosition, tol: float = 1e-6) -> bool:
    return abs(a.absolute_arc - b.absolute_arc) <= tol


def is_octave_relation(a: SpiralPosition, b: SpiralPosition, tol: float = 1e-6) -> bool:
    delta = abs(a.absolute_arc - b.absolute_arc)
    return abs(delta % STEPS_PER_OCTAVE) <= tol


def is_harmonic_relation(a: SpiralPosition, b: SpiralPosition, tol: float = 0.1) -> bool:
    """
    Check if two positions are harmonically related
    based on stable arc differences.

    This replaces naive harmonic detection.
    """
    delta = abs(a.absolute_arc - b.absolute_arc)

    # basic harmonic set (can be expanded)
    harmonic_steps = [0, 12, 19, 24, 28, 31, 36]

    return any(abs(delta - h) <= tol for h in harmonic_steps)


# ============================================================
# CHAIN CONSISTENCY
# ============================================================

def chain_consistency_score(positions: list[SpiralPosition]) -> float:
    """
    Evaluate if positions form a consistent chain.

    Instead of counting matches, we evaluate:
        stability of arc differences
    """
    if len(positions) < 2:
        return 0.0

    deltas = []
    for i in range(len(positions) - 1):
        d = positions[i].distance_to(positions[i + 1])
        deltas.append(d)

    if not deltas:
        return 0.0

    mean = sum(deltas) / len(deltas)
    variance = sum((x - mean) ** 2 for x in deltas) / len(deltas)

    # lower variance → stronger chain
    return 1.0 / (1.0 + variance)