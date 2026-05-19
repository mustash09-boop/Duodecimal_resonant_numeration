# py/music12/core/time60_core.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Iterable, Tuple, Dict
import math

import pandas as pd


@dataclass(frozen=True)
class Time60Params:
    # base-60 resolution levels
    base1: int = 60       # ticks per second
    base2: int = 3600     # ticks per second (60*60)

    # rounding mode for ticks
    # - "round": nearest tick (symmetric)
    # - "floor": always earlier
    # - "ceil" : always later
    rounding: str = "round"


def _ticks(t_sec: float, base: int, rounding: str) -> int:
    if t_sec is None or (isinstance(t_sec, float) and math.isnan(t_sec)):
        return 0
    x = float(t_sec) * float(base)
    if rounding == "floor":
        return int(math.floor(x))
    if rounding == "ceil":
        return int(math.ceil(x))
    # default round (bankers-safe-ish): Python round -> bankers, we want classical
    # Use floor(x+0.5) for positive. Our times are non-negative.
    return int(math.floor(x + 0.5))


def _resid(t_sec: float, tick: int, base: int) -> float:
    if t_sec is None or (isinstance(t_sec, float) and math.isnan(t_sec)):
        return float("nan")
    return float(t_sec) - (float(tick) / float(base))


def add_sample_indices(
    df: pd.DataFrame,
    *,
    sr: int,
    t0_col: str = "t_start",
    t1_col: str = "t_end",
    out_s0: str = "sample_start",
    out_s1: str = "sample_end",
    rounding: str = "round",
) -> pd.DataFrame:
    """
    Adds sample_start/sample_end derived from time columns and sample rate (sr).
    Does NOT alter audio/DSP; it is purely a coordinate transform.

    Rules:
      - sample = round(t * sr) by default
      - if time missing -> NaN remains NaN (you can fill later with flags)
    """
    if sr <= 0:
        raise ValueError("sr must be positive")

    if t0_col not in df.columns or t1_col not in df.columns:
        return df

    d = df.copy()

    def to_sample(x: float) -> float:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return float("nan")
        v = float(x) * float(sr)
        if rounding == "floor":
            return float(math.floor(v))
        if rounding == "ceil":
            return float(math.ceil(v))
        return float(math.floor(v + 0.5))

    if out_s0 not in d.columns:
        d[out_s0] = d[t0_col].apply(to_sample)
    if out_s1 not in d.columns:
        d[out_s1] = d[t1_col].apply(to_sample)

    # keep as int where possible, but preserve NaN:
    for c in (out_s0, out_s1):
        if c in d.columns:
            # pandas nullable integer
            d[c] = pd.to_numeric(d[c], errors="coerce").round(0).astype("Int64")

    return d


def add_time60(
    df: pd.DataFrame,
    *,
    t0_col: str = "t_start",
    t1_col: str = "t_end",
    params: Optional[Time60Params] = None,
    prefix: str = "",
) -> pd.DataFrame:
    """
    Adds base-60 and base-3600 tick coordinates + residuals, without losing precision:
      t_start_60, t_end_60, dur_60, t_start_s_resid_60, t_end_s_resid_60
      t_start_3600, ... , resid_3600

    Residual keeps exact remainder after tick quantization.
    """
    if params is None:
        params = Time60Params()

    if t0_col not in df.columns or t1_col not in df.columns:
        return df

    d = df.copy()

    b1 = int(params.base1)
    b2 = int(params.base2)
    rmode = str(params.rounding or "round").strip().lower()

    # Helpers
    def mk_cols(base: int) -> Tuple[str, str, str, str, str]:
        return (
            f"{prefix}{t0_col}_{base}",
            f"{prefix}{t1_col}_{base}",
            f"{prefix}dur_{base}",
            f"{prefix}{t0_col}_s_resid_{base}",
            f"{prefix}{t1_col}_s_resid_{base}",
        )

    for base in (b1, b2):
        c0, c1, cdur, cr0, cr1 = mk_cols(base)

        if c0 not in d.columns:
            d[c0] = d[t0_col].apply(lambda x: _ticks(x, base, rmode))
        if c1 not in d.columns:
            d[c1] = d[t1_col].apply(lambda x: _ticks(x, base, rmode))

        # duration in ticks (prefer t1-t0 if available)
        if cdur not in d.columns:
            d[cdur] = (d[c1].astype("Int64") - d[c0].astype("Int64")).astype("Int64")

        if cr0 not in d.columns:
            d[cr0] = [
                _resid(ts, int(tk) if tk is not None and tk == tk else 0, base)
                for ts, tk in zip(d[t0_col].tolist(), d[c0].tolist())
            ]
        if cr1 not in d.columns:
            d[cr1] = [
                _resid(ts, int(tk) if tk is not None and tk == tk else 0, base)
                for ts, tk in zip(d[t1_col].tolist(), d[c1].tolist())
            ]

        # ticks as nullable int
        d[c0] = pd.to_numeric(d[c0], errors="coerce").astype("Int64")
        d[c1] = pd.to_numeric(d[c1], errors="coerce").astype("Int64")

    return d