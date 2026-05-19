# -*- coding: utf-8 -*-
"""
music12.core.notation12

Single Source of Truth (SSOT) for our 12-radix musical token language.

Alphabet (bijective base-12, NO ZERO):
  DIGITS12 = "123456789ABC"

Pitch token (canonical):
  OCT.STEP['MICRO]

Where:
  OCT        : bijective base-12 string (multi-digit allowed)
  STEP       : one digit from DIGITS12
  'MICRO     :
               - "'" + "-"          => exact (no micro shift)
               - "'" + "i" + FRAC   => upward refinement
               - "'" + "a" + FRAC   => downward refinement

Rules:
- Zero is forbidden in token digits
- After apostrophe direction is mandatory for fractional refinement
- Forbidden:
    "'FRAC"
    "58iA"
    "5a8"
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

DIGITS12 = "123456789ABC"
_VAL12: Dict[str, int] = {ch: i + 1 for i, ch in enumerate(DIGITS12)}
_CH12: Dict[int, str] = {v: ch for ch, v in _VAL12.items()}

STEP_ORDER = list(DIGITS12)


def normalize_letters(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("А", "A").replace("В", "B").replace("С", "C")
    s = s.replace("а", "A").replace("в", "B").replace("с", "C")
    return s


def bij12_to_int(s: str) -> int:
    s = normalize_letters(s).upper()
    if not s or any(ch not in _VAL12 for ch in s):
        raise ValueError(f"Bad bij12 number: {s!r}")
    n = 0
    for ch in s:
        n = n * 12 + _VAL12[ch]
    return n


def int_to_base12_digit(i: int) -> str:
    if not 0 <= int(i) < 12:
        raise ValueError("int_to_base12_digit expects 0..11")
    return _CH12[int(i) + 1]


def int_to_bij12(n: int) -> str:
    if n <= 0:
        raise ValueError("int_to_bij12 expects n>=1")
    out: List[str] = []
    while n > 0:
        n, r = divmod(n - 1, 12)
        out.append(_CH12[r + 1])
    return "".join(reversed(out))


def oct_index0(oct_s: str) -> int:
    return bij12_to_int(oct_s) - 1


def step_index0(step: str) -> int:
    step = normalize_letters(step).upper()
    if step not in _VAL12:
        raise ValueError(f"Bad step digit: {step!r}")
    return _VAL12[step] - 1


# 🔴 ОБНОВЛЁННЫЙ REGEX (строгий)
_TOKEN_RX = re.compile(
    r"^(?P<oct>[123456789ABC]+)\.(?P<step>[123456789ABC])"
    r"(?:'(?P<micro>(?:-|(?:(?P<mdir>[ia])(?P<frac>[123456789ABC]+)))))?$",
    flags=re.IGNORECASE,
)

TOKEN_RE = _TOKEN_RX


@dataclass(frozen=True)
class Token:
    oct: str
    step: str
    legacy_alt: str = ""
    micro_dir: Optional[str] = None
    micro_frac_raw: str = ""
    k144: str = ""
    k1728: str = ""

    @property
    def frac(self) -> str:
        if self.micro_dir == "-":
            return "-"
        if self.micro_dir in ("i", "a"):
            return self.micro_frac_raw
        return ""


def parse_token(tok: str) -> Token:
    tok = normalize_letters(tok)
    m = _TOKEN_RX.match(tok)
    if not m:
        raise ValueError(f"Bad token: {tok!r}")

    oct_s = m.group("oct").upper()
    step = m.group("step").upper()

    micro = m.group("micro")
    if micro is None:
        return Token(oct_s, step)

    if micro == "-":
        return Token(oct_s, step, micro_dir="-")

    mdir = m.group("mdir")
    frac = (m.group("frac") or "").upper()

    if mdir not in ("i", "a"):
        raise ValueError(f"Invalid micro direction in {tok}")

    if not frac:
        raise ValueError(f"Fraction required after i/a in {tok}")

    k144 = frac[:1]
    k1728 = frac[1:] if len(frac) > 1 else ""

    return Token(oct_s, step, "", mdir, frac, k144, k1728)


def format_token(t: Token) -> str:
    s = f"{t.oct}.{t.step}"

    if t.micro_dir is None:
        return s
    if t.micro_dir == "-":
        return s + "'-"

    return s + "'" + t.micro_dir + t.micro_frac_raw


def normalize_token(tok: str) -> str:
    return format_token(parse_token(tok))


# -----------------------------
# DirectedToken12 (упрощён)
# -----------------------------

@dataclass(frozen=True)
class DirectedToken12:
    base_token: str
    fraction_digits: str = ""
    tail_direction: Optional[str] = None


def parse_directed_token12(token: str) -> DirectedToken12:
    if "'" not in token:
        raise ValueError("Apostrophe required")

    left, right = token.split("'", 1)

    if right == "" or right == "-":
        return DirectedToken12(left)

    if right[0] not in ("i", "a"):
        raise ValueError("Only i/a allowed after apostrophe")

    return DirectedToken12(
        base_token=left,
        fraction_digits=right[1:].upper(),
        tail_direction=right[0],
    )


def format_directed_token12(dt: DirectedToken12) -> str:
    base = dt.base_token

    if dt.tail_direction is None:
        return base + "'"

    if not dt.fraction_digits:
        return base + "'-"

    return base + "'" + dt.tail_direction + dt.fraction_digits


def is_directed_token12(token: str) -> bool:
    try:
        parse_directed_token12(token)
        return True
    except Exception:
        return False