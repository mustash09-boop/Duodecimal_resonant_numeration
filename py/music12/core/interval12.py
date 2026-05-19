# -*- coding: utf-8 -*-
"""
music12.core.interval12

Structured interval arithmetic for directed duodecimal tokens.

Principle:
- We do NOT add "tokens to tokens" as plain numbers.
- We compute movement by layers:
    1) octave level
    2) step inside octave
    3) fractional octave digits after apostrophe
    4) tail direction (i/a) of the last fractional digit

Main operations:
- compare_directed_tokens(a, b)
- subtract_directed_tokens(a, b) -> DirectedInterval12
- apply_directed_interval(base, interval) -> DirectedToken12

This is a safe universal layer for experiments and future projects.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import zip_longest
from typing import Optional, Tuple

from music12.core.notation12 import (
    DIGITS12,
    DirectedToken12,
    bij12_to_int,
    int_to_bij12,
    parse_directed_token12,
    format_directed_token12,
    parse_token,
)


# ------------------------------------------------------------
# Digit helpers
# ------------------------------------------------------------

_DIGIT_TO_INT = {ch: i + 1 for i, ch in enumerate(DIGITS12)}   # 1..12
_INT_TO_DIGIT = {i + 1: ch for i, ch in enumerate(DIGITS12)}   # 1..12 -> digit


def digit12_to_int(ch: str) -> int:
    ch = str(ch).strip().upper()
    if ch not in _DIGIT_TO_INT:
        raise ValueError(f"Bad bij12 digit: {ch!r}")
    return _DIGIT_TO_INT[ch]


def int_to_digit12(n: int) -> str:
    n = int(n)
    if n not in _INT_TO_DIGIT:
        raise ValueError(f"int_to_digit12 expects 1..12, got {n}")
    return _INT_TO_DIGIT[n]


# ------------------------------------------------------------
# Direction helpers
# ------------------------------------------------------------

def direction_to_sign(direction: Optional[str]) -> int:
    if direction is None:
        return 0
    if direction == "i":
        return 1
    if direction == "a":
        return -1
    raise ValueError(f"Bad direction: {direction!r}")


def sign_to_direction(sign: int) -> Optional[str]:
    if sign > 0:
        return "i"
    if sign < 0:
        return "a"
    return None


# ------------------------------------------------------------
# Structured interval
# ------------------------------------------------------------

@dataclass(frozen=True)
class DirectedInterval12:
    """
    Interval between two directed tokens.

    octave_delta:
        difference between octave indices (bij12 integer domain)

    step_delta:
        difference between base steps inside octave, using 1..12 digit scale

    fraction_deltas:
        positional deltas of fractional octave digits after apostrophe.
        Stored left-to-right, same logical depth as written.

    tail_direction_delta:
        difference of direction tendency on the last fractional digit:
            i -> +1
            a -> -1
            None -> 0
    """
    octave_delta: int
    step_delta: int
    fraction_deltas: Tuple[int, ...] = ()
    tail_direction_delta: int = 0

    @property
    def has_fraction(self) -> bool:
        return len(self.fraction_deltas) > 0

    @property
    def is_zero(self) -> bool:
        return (
            self.octave_delta == 0
            and self.step_delta == 0
            and all(v == 0 for v in self.fraction_deltas)
            and self.tail_direction_delta == 0
        )


# ------------------------------------------------------------
# Internal extraction / normalization
# ------------------------------------------------------------

def _token_base_parts(base_token: str) -> tuple[str, str]:
    t = parse_token(base_token)
    return t.oct, t.step


def _fraction_digits_to_ints(frac: str) -> Tuple[int, ...]:
    if not frac:
        return ()
    return tuple(digit12_to_int(ch) for ch in frac)


def _ints_to_fraction_digits(values: Tuple[int, ...]) -> str:
    if not values:
        return ""
    return "".join(int_to_digit12(v) for v in values)


def _pad_fraction_vectors(a: Tuple[int, ...], b: Tuple[int, ...]) -> tuple[Tuple[int, ...], Tuple[int, ...]]:
    n = max(len(a), len(b))
    aa = a + (0,) * (n - len(a))
    bb = b + (0,) * (n - len(b))
    return aa, bb


def _trim_right_zeros(vals: Tuple[int, ...]) -> Tuple[int, ...]:
    lst = list(vals)
    while lst and lst[-1] == 0:
        lst.pop()
    return tuple(lst)


def _normalize_fraction_digits_signed(vals: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Normalize signed positional digit deltas by carrying in base-12.
    This is interval normalization, not token normalization.

    Keeps each cell roughly in range [-11..11], carrying overflow to the left.
    """
    if not vals:
        return ()

    arr = list(vals)

    # work right-to-left
    for i in range(len(arr) - 1, 0, -1):
        v = arr[i]
        while v > 11:
            arr[i] -= 12
            arr[i - 1] += 1
            v = arr[i]
        while v < -11:
            arr[i] += 12
            arr[i - 1] -= 1
            v = arr[i]

    return _trim_right_zeros(tuple(arr))


def _token_scalar_key(dt: DirectedToken12) -> tuple:
    """
    Lexicographic comparison key:
      octave, step, fractional digits, tail direction sign

    This is enough for ordering and interval extraction,
    without pretending tokens are plain arithmetic numbers.
    """
    oct_s, step = _token_base_parts(dt.base_token)
    oct_idx = bij12_to_int(oct_s)
    step_idx = digit12_to_int(step)
    frac = _fraction_digits_to_ints(dt.fraction_digits)
    tail = direction_to_sign(dt.tail_direction)
    return (oct_idx, step_idx, frac, tail)


# ------------------------------------------------------------
# Public operations
# ------------------------------------------------------------

def compare_directed_tokens(a: str | DirectedToken12, b: str | DirectedToken12) -> int:
    """
    Compare two directed tokens.

    Returns:
      -1 if a < b
       0 if a == b
      +1 if a > b
    """
    aa = parse_directed_token12(a) if isinstance(a, str) else a
    bb = parse_directed_token12(b) if isinstance(b, str) else b

    ka = _token_scalar_key(aa)
    kb = _token_scalar_key(bb)

    if ka < kb:
        return -1
    if ka > kb:
        return 1
    return 0


def subtract_directed_tokens(a: str | DirectedToken12, b: str | DirectedToken12) -> DirectedInterval12:
    """
    Compute structured interval: a - b
    """
    aa = parse_directed_token12(a) if isinstance(a, str) else a
    bb = parse_directed_token12(b) if isinstance(b, str) else b

    aoct, astep = _token_base_parts(aa.base_token)
    boct, bstep = _token_base_parts(bb.base_token)

    octave_delta = bij12_to_int(aoct) - bij12_to_int(boct)
    step_delta = digit12_to_int(astep) - digit12_to_int(bstep)

    af = _fraction_digits_to_ints(aa.fraction_digits)
    bf = _fraction_digits_to_ints(bb.fraction_digits)
    afp, bfp = _pad_fraction_vectors(af, bf)

    fraction_deltas = tuple(x - y for x, y in zip(afp, bfp))
    fraction_deltas = _normalize_fraction_digits_signed(fraction_deltas)

    tail_direction_delta = direction_to_sign(aa.tail_direction) - direction_to_sign(bb.tail_direction)

    return DirectedInterval12(
        octave_delta=octave_delta,
        step_delta=step_delta,
        fraction_deltas=fraction_deltas,
        tail_direction_delta=tail_direction_delta,
    )


def apply_directed_interval(
    base: str | DirectedToken12,
    interval: DirectedInterval12,
) -> DirectedToken12:
    """
    Apply a structured interval to a base directed token.

    This is a conservative positional application:
    - octave and step are shifted first
    - fraction digits are shifted positionally
    - tail direction is updated by signed delta and clipped to {-1,0,+1}

    NOTE:
    this is intended for controlled experiments and interval transport,
    not yet as the final universal "physical arithmetic of resonance".
    """
    bb = parse_directed_token12(base) if isinstance(base, str) else base

    boct, bstep = _token_base_parts(bb.base_token)

    new_oct = bij12_to_int(boct) + interval.octave_delta
    if new_oct < 1:
        raise ValueError("apply_directed_interval produced octave index < 1")

    new_step = digit12_to_int(bstep) + interval.step_delta

    # normalize step with octave carry
    while new_step > 12:
        new_step -= 12
        new_oct += 1
    while new_step < 1:
        new_step += 12
        new_oct -= 1
        if new_oct < 1:
            raise ValueError("apply_directed_interval produced octave index < 1 after step borrow")

    base_frac = _fraction_digits_to_ints(bb.fraction_digits)
    frac_deltas = interval.fraction_deltas
    base_pad, delta_pad = _pad_fraction_vectors(base_frac, frac_deltas)

    out_frac = []
    carry = 0

    # right-to-left positional carry
    for i in range(len(base_pad) - 1, -1, -1):
        v = base_pad[i] + delta_pad[i] + carry
        carry = 0

        while v > 12:
            v -= 12
            carry += 1
        while v < 1:
            v += 12
            carry -= 1

        out_frac.append(v)

    out_frac = list(reversed(out_frac))

    if carry != 0:
        # carry from fractional domain goes into step
        new_step += carry
        while new_step > 12:
            new_step -= 12
            new_oct += 1
        while new_step < 1:
            new_step += 12
            new_oct -= 1
            if new_oct < 1:
                raise ValueError("apply_directed_interval produced octave index < 1 after fraction carry")

    frac_digits = _ints_to_fraction_digits(tuple(out_frac)).rstrip()
    frac_vals = _trim_right_zeros(tuple(out_frac))
    frac_digits = _ints_to_fraction_digits(frac_vals)

    tail_sign = direction_to_sign(bb.tail_direction) + interval.tail_direction_delta
    if tail_sign > 0:
        tail_sign = 1
    elif tail_sign < 0:
        tail_sign = -1
    tail_direction = sign_to_direction(tail_sign)

    base_token = f"{int_to_bij12(new_oct)}.{int_to_digit12(new_step)}"

    # if there is no fractional part, direction cannot survive
    if frac_digits == "":
        tail_direction = None

    return DirectedToken12(
        base_token=base_token,
        fraction_digits=frac_digits,
        tail_direction=tail_direction,
    )


def format_directed_interval(iv: DirectedInterval12) -> str:
    """
    Human-readable compact interval representation.
    Example:
      Δ(oct=+1, step=-2, frac=(0,3,-1), dir=+1)
    """
    frac_text = ",".join(str(x) for x in iv.fraction_deltas) if iv.fraction_deltas else ""
    dir_text = f"{iv.tail_direction_delta:+d}"
    return f"Δ(oct={iv.octave_delta:+d}, step={iv.step_delta:+d}, frac=({frac_text}), dir={dir_text})"