from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from music12.core.notation12 import STEP_ORDER, normalize_token
from music12.core.harmonic_alphabet12 import harmonic_token_from_root


@dataclass
class HarmonicAnchor:
    root_token: str
    h_left: int
    h_right: int
    token_left: str
    token_right: str
    step_left: str
    step_right: str
    pair_signature: str


def token_step(token: str) -> str:
    tok = normalize_token(token)
    if "." not in tok:
        raise ValueError(f"Bad token: {token}")
    return tok.split(".", 1)[1][0]


def build_anchor_for_root(root_token: str, h_left: int, h_right: int) -> HarmonicAnchor:
    left = harmonic_token_from_root(root_token, h_left)
    right = harmonic_token_from_root(root_token, h_right)

    step_left = token_step(left)
    step_right = token_step(right)

    return HarmonicAnchor(
        root_token=normalize_token(root_token),
        h_left=h_left,
        h_right=h_right,
        token_left=left,
        token_right=right,
        step_left=step_left,
        step_right=step_right,
        pair_signature=f"{step_left}->{step_right}",
    )


def build_anchor_table(
    root_octave: str = "5",
    anchor_pairs: Sequence[Tuple[int, int]] = ((1, 3), (3, 5), (5, 7)),
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    for step in STEP_ORDER:
        root = normalize_token(f"{root_octave}.{step}")
        row: Dict[str, str] = {
            "root_token": root,
            "root_step": step,
        }

        for hl, hr in anchor_pairs:
            a = build_anchor_for_root(root, hl, hr)
            key = f"h{hl}_h{hr}"
            row[f"{key}_left"] = a.token_left
            row[f"{key}_right"] = a.token_right
            row[f"{key}_sig"] = a.pair_signature

        rows.append(row)

    return rows