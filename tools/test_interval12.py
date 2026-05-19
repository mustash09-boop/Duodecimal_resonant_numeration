# -*- coding: utf-8 -*-

from music12.core.interval12 import (
    subtract_directed_tokens,
    apply_directed_interval,
    format_directed_interval,
)
from music12.core.notation12 import format_directed_token12


pairs = [
    ("1.1'58", "1.1'5a8"),
    ("1.1'58iA", "1.1'58A"),
    ("11.A'9", "C.3'5iB"),
]

for a, b in pairs:
    print("=" * 70)
    print("A =", a)
    print("B =", b)

    iv = subtract_directed_tokens(a, b)
    print("A - B =", format_directed_interval(iv))

    back = apply_directed_interval(b, iv)
    print("B + (A - B) =", format_directed_token12(back))