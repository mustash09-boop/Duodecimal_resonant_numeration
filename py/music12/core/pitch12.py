# -*- coding: utf-8 -*-
"""music12.core.pitch12 (DEPRECATED)

This module used to implement the canonical 12‑radix pitch container.
The SSOT has been consolidated into: music12.core.notation12

Keep this file as a thin compatibility layer so existing imports keep working.

IMPORTANT:
- Block002 tokenizers historically import: from music12.core.pitch12 import hz_to_pitch12
- Different generations passed different keyword arguments (a4 vs a4_hz, anchor_token, micro_depth, ...)

This file provides a *stable* bridge that accepts both old and new call styles,
and delegates to notation12.hz_to_token whenever available.
"""

from __future__ import annotations

from music12.core.notation12 import *  # noqa: F401,F403

import math
from typing import Any, Optional
from music12.core import notation12 as _n12


def _anchor_abs_index(anchor_token: str) -> int:
    """Convert anchor token (e.g. '9.A') to absolute semitone index without depending on optional APIs."""
    anchor_token = str(anchor_token or "").strip()
    if not anchor_token:
        raise ValueError("anchor_token is empty")

    fn = getattr(_n12, "token_to_abs_semitone_index", None)
    if callable(fn):
        return int(fn(anchor_token))

    # Fallback: parse token + compute absolute index (micro ignored; legacy_alt folded)
    t = _n12.parse_token(anchor_token)
    abs_int = _n12.oct_index0(t.oct) * 12 + _n12.step_index0(t.step) + _n12._alt_shift(getattr(t, "legacy_alt", "") or "")
    return int(abs_int)


def hz_to_pitch12(
    freq_hz: float,
    a4: Optional[float] = None,
    *,
    a4_hz: Optional[float] = None,
    anchor_token: Optional[str] = None,
    micro_depth: int = 2,
    force_micro_dash_when_exact: bool = True,
    **_kwargs: Any,
) -> str:
    """Compatibility bridge for Block002 tokenization.

    Accepts both old and new call styles:
      - hz_to_pitch12(freq_hz, a4_hz=440.0)
      - hz_to_pitch12(freq_hz, a4=440.0, anchor_token='9.A', micro_depth=2)

    Returns canonical 12-radix token string (e.g. "9.A'-", "7.9'i3A", ...).
    """
    try:
        f = float(freq_hz)
    except Exception:
        return ""
    if not (f > 0):
        return ""

    # Resolve A4 reference keyword variants
    ref = a4_hz if a4_hz is not None else (a4 if a4 is not None else 440.0)

    # Resolve anchor token
    if anchor_token is None:
        anchor_token = getattr(_n12, "DEFAULT_ANCHOR_TOKEN", "9.A")

    # Preferred: delegate to SSOT
    hz_to_token = getattr(_n12, "hz_to_token", None)
    if callable(hz_to_token):
        try:
            return str(
                hz_to_token(
                    f,
                    a4_hz=float(ref),
                    anchor_token=str(anchor_token),
                    micro_depth=int(micro_depth),
                    force_micro_dash_when_exact=bool(force_micro_dash_when_exact),
                )
            )
        except TypeError:
            # Some transitional variants may not accept all kwargs; fall through.
            pass

    # Fallback (approx, semitone rounding + coarse micro):
    abs_anchor = _anchor_abs_index(str(anchor_token))
    semitone_offset = 12.0 * math.log2(f / float(ref))
    abs_index = int(round(semitone_offset)) + abs_anchor

    oct_s, step = _n12.abs_semitone_to_oct_step(abs_index)

    # residual in cents (after semitone rounding)
    cents = (semitone_offset - round(semitone_offset)) * 100.0
    abs_c = abs(cents)

    if force_micro_dash_when_exact and abs_c < 1e-6:
        micro = "'-"
    else:
        if abs_c < 25:
            micro = "'-"
        else:
            micro = "'a" if cents > 0 else "'i"

    token = f"{oct_s}.{step}{micro}"
    return _n12.normalize_token(token)
