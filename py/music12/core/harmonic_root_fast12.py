from __future__ import annotations

"""
music12.core.harmonic_root_fast12
---------------------------------

Fast root-step detector for the 12-radix music system.

Idea:
- compress observed token12 events into a 12-step vector
- build one base harmonic template
- detect best root as best cyclic shift of the template

This module detects ROOT STEP first (mod 12).
Octave resolution should be done later as a separate stage.

Depends on canonical project notation from notation12.py
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import math

import pandas as pd

from music12.core.notation12 import STEP_ORDER, normalize_token

# ------------------------------------------------------------
# Basic constants
# ------------------------------------------------------------

N_STEPS = 12
STEP_TO_INDEX = {s: i for i, s in enumerate(STEP_ORDER)}
INDEX_TO_STEP = {i: s for i, s in enumerate(STEP_ORDER)}


# ------------------------------------------------------------
# Data classes
# ------------------------------------------------------------

@dataclass
class RootFastParams:
    max_harmonic: int = 12
    harmonic_decay: float = 0.92

    early_bonus_h1: float = 1.80
    early_bonus_h2: float = 1.00
    early_bonus_h3: float = 1.45

    odd_harmonic_bonus: float = 1.35
    even_harmonic_penalty: float = 0.90

    use_energy: bool = True
    use_count: bool = True
    missing_penalty: float = 0.15

    lowest_root_bonus: float = 0.20

    # --- НОВОЕ ---
    f0_penalty: float = 0.6           # штраф за физически маловероятный корень
    f0_harmonic_limit: int = 10       # если для объяснения самой низкой частоты нужен h > limit → штраф


@dataclass
class RootScore:
    root_step: str
    shift: int
    score: float
    matched_steps: List[str]
    template_weights: List[float]
    observed_vector: List[float]


# ------------------------------------------------------------
# Token helpers
# ------------------------------------------------------------

def split_token12(token: str) -> Tuple[str, str]:
    """
    Canonical token: 9.A, A.3, 5.C'-, 5.C'i3 etc.
    We only need octave.step base here, so normalize first.
    """
    tok = normalize_token(str(token).strip())
    if "." not in tok:
        raise ValueError(f"Bad token12: {token}")
    octv, step_and_micro = tok.split(".", 1)

    # base step = first symbol after dot
    step = step_and_micro[0]
    if step not in STEP_TO_INDEX:
        raise ValueError(f"Bad step in token12: {token}")
    return octv, step


def token_to_step(token: str) -> str:
    return split_token12(token)[1]


# ------------------------------------------------------------
# Harmonic template
# ------------------------------------------------------------

def harmonic_step_offset(h: int) -> int:
    """
    Temporary harmonic placement by physical approximation:
        round(12 * log2(h)) mod 12

    IMPORTANT:
    This is only the first fast layer.
    Later this can be replaced by your true 12-radix harmonic alphabet law.
    """
    if h < 1:
        raise ValueError("h must be >= 1")
    return int(round(12.0 * math.log2(h))) % 12


def build_base_harmonic_template(
    params: Optional[RootFastParams] = None,
) -> List[float]:
    """
    Build one base template for root step = STEP_ORDER[0]
    as a 12-length cyclic vector.

    Odd harmonics are emphasized because they are better octave anchors.
    """
    params = params or RootFastParams()
    tpl = [0.0] * N_STEPS

    for h in range(1, params.max_harmonic + 1):
        idx = harmonic_step_offset(h)

        w = params.harmonic_decay ** (h - 1)

        # early harmonics
        if h == 1:
            w *= params.early_bonus_h1
        elif h == 2:
            w *= params.early_bonus_h2
        elif h == 3:
            w *= params.early_bonus_h3

        # odd/even logic
        if h % 2 == 1:
            w *= params.odd_harmonic_bonus
        else:
            w *= params.even_harmonic_penalty

        tpl[idx] += w

    return tpl


def rotate_vector(v: Sequence[float], shift: int) -> List[float]:
    """
    Cyclic rotation to the right by shift.
    """
    n = len(v)
    s = shift % n
    if s == 0:
        return list(v)
    return list(v[-s:] + v[:-s])


# ------------------------------------------------------------
# Observed vector
# ------------------------------------------------------------

def build_observed_step_vector(
    df: pd.DataFrame,
    token_col: str = "token12",
    energy_col: Optional[str] = "energy_db",
    count_weight: float = 1.0,
    energy_weight: float = 0.15,
    duration_col: Optional[str] = None,
    duration_weight: float = 0.0,
) -> List[float]:
    """
    Compress observed token events into 12-step mod12 vector.
    Octave is intentionally ignored here.
    """
    vec = [0.0] * N_STEPS

    if token_col not in df.columns:
        return vec

    for _, row in df.iterrows():
        tok = row.get(token_col)
        if pd.isna(tok):
            continue

        try:
            step = token_to_step(str(tok))
        except Exception:
            continue

        idx = STEP_TO_INDEX[step]

        w = 0.0

        # count contribution
        w += count_weight

        # energy contribution
        if energy_col and energy_col in df.columns:
            try:
                ene = float(row.get(energy_col, 0.0))
                if math.isfinite(ene):
                    w += energy_weight * ene
            except Exception:
                pass

        # duration contribution
        if duration_col and duration_col in df.columns:
            try:
                dur = float(row.get(duration_col, 0.0))
                if math.isfinite(dur):
                    w += duration_weight * dur
            except Exception:
                pass

        vec[idx] += w

    return vec


# ------------------------------------------------------------
# Root scoring
# ------------------------------------------------------------

def score_shift(
    observed: Sequence[float],
    template: Sequence[float],
    shift: int,
    params: Optional[RootFastParams] = None,
    lowest_freq: Optional[float] = None,
    a4_hz: float = 440.0,
) -> RootScore:
    params = params or RootFastParams()

    tpl = rotate_vector(template, shift)

    score = 0.0
    matched_steps: List[str] = []

    for i in range(N_STEPS):
        contrib = observed[i] * tpl[i]
        score += contrib
        if observed[i] > 0 and tpl[i] > 0:
            matched_steps.append(INDEX_TO_STEP[i])

    # penalty for expected structure absent in observation
    for i in range(N_STEPS):
        if tpl[i] > 0 and observed[i] <= 0:
            score -= params.missing_penalty * tpl[i]

    # prefer lower root classes slightly
    # useful when octave ambiguity creates competing explanations
    score -= params.lowest_root_bonus * shift

    # -------------------------------------------------
    # F0 constraint: penalize roots that require
    # unrealistically high harmonic number to explain
    # the lowest observed frequency
    # -------------------------------------------------
    if lowest_freq and lowest_freq > 0:

        # convert root step → approximate frequency
        # (temporary bridge via token)
        try:
            from music12.core.pitch12 import token_to_hz
            root_token = f"5.{INDEX_TO_STEP[shift]}"
            root_freq = token_to_hz(root_token, a4_hz=a4_hz)
        except Exception:
            root_freq = None

        if root_freq and root_freq > 0:
            harmonic_no = lowest_freq / root_freq

            if harmonic_no > params.f0_harmonic_limit:
                score -= params.f0_penalty * harmonic_no

    return RootScore(
        root_step=INDEX_TO_STEP[shift],
        shift=shift,
        score=float(score),
        matched_steps=matched_steps,
        template_weights=list(tpl),
        observed_vector=list(observed),
    )


def pick_best_root_step(
    df: pd.DataFrame,
    token_col: str = "token12",
    energy_col: Optional[str] = "energy_db",
    duration_col: Optional[str] = None,
    params: Optional[RootFastParams] = None,
) -> Tuple[RootScore, List[RootScore], List[float]]:

    params = params or RootFastParams()

    observed = build_observed_step_vector(
        df=df,
        token_col=token_col,
        energy_col=energy_col,
        duration_col=duration_col,
    )

    # строим базовый гармонический шаблон
    base_tpl = build_base_harmonic_template(params)

    # ищем минимальную наблюдаемую частоту
    lowest_freq = None
    if "f0_hz_med" in df.columns:
        try:
            lowest_freq = float(df["f0_hz_med"].min())
        except Exception:
            lowest_freq = None

    scores = [
        score_shift(
            observed,
            base_tpl,
            shift=i,
            params=params,
            lowest_freq=lowest_freq,
        )
        for i in range(N_STEPS)
    ]

    scores.sort(key=lambda x: x.score, reverse=True)

    if not scores:
        raise ValueError("No root scores computed")

    return scores[0], scores, observed


# ------------------------------------------------------------
# Reporting helpers
# ------------------------------------------------------------

def scores_to_dataframe(scores: Sequence[RootScore]) -> pd.DataFrame:
    rows = []
    for s in scores:
        rows.append({
            "root_step": s.root_step,
            "shift": s.shift,
            "score": s.score,
            "matched_steps": " | ".join(s.matched_steps),
            "template_weights": " | ".join(f"{x:.4f}" for x in s.template_weights),
            "observed_vector": " | ".join(f"{x:.4f}" for x in s.observed_vector),
        })
    return pd.DataFrame(rows)


def best_root_summary(best: RootScore) -> Dict[str, object]:
    return {
        "best_root_step": best.root_step,
        "best_shift": best.shift,
        "best_score": best.score,
        "matched_steps": " | ".join(best.matched_steps),
    }