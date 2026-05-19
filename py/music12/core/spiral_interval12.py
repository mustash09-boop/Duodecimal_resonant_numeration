from __future__ import annotations

from dataclasses import dataclass

from music12.core.interval12_algebra import interval_between


STEP_DEG = 360.0 / 12.0


@dataclass(frozen=True)
class SpiralInterval:
    angle_deg: float
    radial: float


def spiral_interval(a: str, b: str) -> SpiralInterval:
    iv = interval_between(a, b)
    semis = iv.total_micro / 144.0

    return SpiralInterval(
        angle_deg=semis * STEP_DEG,
        radial=semis / 12.0,
    )