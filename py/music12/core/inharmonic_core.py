"""
music12.core.inharmonic_core

Inharmonicity model for string instruments (esp. piano).
Used as a shared DSP primitive across blocks:
  - Block004 (instrument calibration / templates)
  - Block002 (multi-pitch / harmonic coherence)
  - Block003 (verification / scoring)

Model (piano string approximation):
  f_k ≈ k * f0 * sqrt(1 + B * k^2)

Where:
  f0 - fundamental frequency (Hz)
  k  - partial index (1..)
  B  - inharmonicity coefficient (dimensionless), typically small (0..~0.01)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import math
import numpy as np


def cents_diff(a_hz: float, b_hz: float) -> float:
    """Return cents difference: 1200*log2(a/b)."""
    return 1200.0 * math.log2(max(a_hz / max(b_hz, 1e-12), 1e-12))


@dataclass(frozen=True)
class InharmonicModel:
    """Inharmonic partial frequency model."""
    B: float = 0.0

    def partial_hz(self, f0: float, k: int) -> float:
        if f0 <= 0.0 or k <= 0:
            return 0.0
        kk = float(k)
        return float(kk * f0 * math.sqrt(1.0 + self.B * (kk * kk)))

    def partials_hz(self, f0: float, ks: Iterable[int]) -> np.ndarray:
        ks_arr = np.array(list(ks), dtype=np.int32)
        if f0 <= 0.0:
            return np.zeros_like(ks_arr, dtype=np.float64)
        kf = ks_arr.astype(np.float64)
        return kf * float(f0) * np.sqrt(1.0 + float(self.B) * (kf ** 2))


def fit_inharmonic_B_grid(
    f_obs: Sequence[float],
    ks: Sequence[int],
    f0: float,
    *,
    B_min: float = 0.0,
    B_max: float = 0.01,
    n_grid: int = 400,
    robust: bool = True,
) -> float:
    """
    Fit B by brute-force grid search (robust + dependency-free).
    Minimizes error in log-frequency domain:

      pred = log(k*f0*sqrt(1 + B*k^2))
      err  = log(f_obs) - pred

    If robust=True: minimize median squared error (more robust to outliers).
    Otherwise: minimize mean squared error.
    """
    f_obs = np.array(f_obs, dtype=np.float64)
    ks = np.array(ks, dtype=np.int32)

    if f0 <= 0.0:
        return 0.0

    mask = (f_obs > 0.0) & (ks > 0)
    f_obs = f_obs[mask]
    ks = ks[mask]
    if len(f_obs) < 2:
        return 0.0

    log_obs = np.log(np.maximum(f_obs, 1e-12))
    base = np.log(np.maximum(ks.astype(np.float64) * float(f0), 1e-12))
    k2 = (ks.astype(np.float64) ** 2)

    Bs = np.linspace(B_min, B_max, int(n_grid), dtype=np.float64)

    best_B = 0.0
    best_err = 1e99
    for B in Bs:
        pred = base + 0.5 * np.log(1.0 + B * k2)
        e2 = (log_obs - pred) ** 2
        err = float(np.median(e2) if robust else np.mean(e2))
        if err < best_err:
            best_err = err
            best_B = float(B)

    return best_B


def infer_B_from_peaks_near_partials(
    peak_freqs_hz: Sequence[float],
    peak_mags: Sequence[float],
    *,
    f0: float,
    ks: Sequence[int],
    tol_cents: float = 40.0,
    B_min: float = 0.0,
    B_max: float = 0.01,
    n_grid: int = 400,
) -> float:
    """
    Convenience helper:
    Given peak lists (freqs + mags) and a tentative f0,
    choose the nearest peak for each partial k (using *harmonic* targets),
    then fit B on those observed partial freqs.

    NOTE: This is a bootstrap helper; for best results, call on k=2..8 (or similar).
    """
    pf = np.array(peak_freqs_hz, dtype=np.float64)
    pm = np.array(peak_mags, dtype=np.float64)
    if len(pf) == 0 or f0 <= 0.0:
        return 0.0

    obs = []
    obs_ks = []

    for k in ks:
        target = float(k) * float(f0)
        j = int(np.argmin(np.abs(pf - target)))
        r = pf[j] / max(target, 1e-12)
        cents = abs(1200.0 * np.log2(max(r, 1e-12)))
        if cents <= float(tol_cents):
            obs.append(float(pf[j]))
            obs_ks.append(int(k))

    if len(obs) < 2:
        return 0.0

    return fit_inharmonic_B_grid(
        obs,
        obs_ks,
        f0,
        B_min=B_min,
        B_max=B_max,
        n_grid=n_grid,
        robust=True,
    )
