from __future__ import annotations

from dataclasses import dataclass
from math import pow


BIJ12_DIGITS = "123456789ABC"
DIGIT_TO_INT = {ch: i + 1 for i, ch in enumerate(BIJ12_DIGITS)}
INT_TO_DIGIT = {i + 1: ch for i, ch in enumerate(BIJ12_DIGITS)}

STEPS_PER_OCTAVE = 12


def bij12_to_int(text: str) -> int:
    s = str(text).strip().upper()
    value = 0
    for ch in s:
        value = value * 12 + DIGIT_TO_INT[ch]
    return value


def int_to_bij12(value: int) -> str:
    out = []
    n = value
    while n > 0:
        n -= 1
        n, rem = divmod(n, 12)
        out.append(INT_TO_DIGIT[rem + 1])
    return "".join(reversed(out))


def split_note_token(token: str):
    s = token.strip().upper()
    octave, rest = s.split(".", 1)

    step = ""
    suffix = ""

    for i, ch in enumerate(rest):
        if ch in BIJ12_DIGITS:
            step += ch
        else:
            suffix = rest[i:]
            break

    return octave, step, suffix


def parse_suffix(suffix: str) -> int:
    if not suffix or suffix == "'-":
        return 0

    s = suffix[1:]
    sign = 1
    count = 0
    i = 0

    while i < len(s) and s[i] in ("I", "A"):
        if s[i] == "I":
            sign = 1
        else:
            sign = -1
        count += 1
        i += 1

    fine = 0
    if i < len(s):
        fine = bij12_to_int(s[i:])

    return sign * (count * 12 + fine)


def micro_to_suffix(value: int) -> str:
    if value == 0:
        return "'-"

    sign = "I" if value > 0 else "A"
    n = abs(value)
    coarse, fine = divmod(n, 12)

    out = sign * coarse
    if fine:
        out += int_to_bij12(fine)

    return "'" + out


def note_to_micro(token: str) -> int:
    octave, step, suffix = split_note_token(token)

    o = bij12_to_int(octave)
    s = bij12_to_int(step) - 1

    base = (o - 1) * 12 + s
    micro = parse_suffix(suffix)

    return base * 144 + micro


def micro_to_note(value: int) -> str:
    semis, micro = divmod(value, 144)
    o, s = divmod(semis, 12)

    octave = int_to_bij12(o + 1)
    step = int_to_bij12(s + 1)
    suffix = micro_to_suffix(micro)

    return f"{octave}.{step}{suffix}"


@dataclass(frozen=True)
class Interval12:
    total_micro: int

    def __add__(self, other: "Interval12"):
        return Interval12(self.total_micro + other.total_micro)

    def __sub__(self, other: "Interval12"):
        return Interval12(self.total_micro - other.total_micro)

    def __neg__(self):
        return Interval12(-self.total_micro)

    def apply(self, anchor: str) -> str:
        return micro_to_note(note_to_micro(anchor) + self.total_micro)

    def text(self):
        sign = "+" if self.total_micro >= 0 else "-"
        val = abs(self.total_micro)
        semis, micro = divmod(val, 144)

        if micro == 0:
            return f"d{{{sign}{int_to_bij12(semis)}}}"

        return f"d{{{sign}{int_to_bij12(semis)}.{micro_to_suffix(micro)[1:]}}}"

    def ratio(self):
        return pow(2.0, self.total_micro / (144 * 12))


def interval_between(a: str, b: str) -> Interval12:
    return Interval12(note_to_micro(b) - note_to_micro(a))