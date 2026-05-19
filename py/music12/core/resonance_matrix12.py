from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import csv
import math
from pathlib import Path

from music12.core.notation12 import STEP_ORDER, normalize_token
from music12.core.harmonic_alphabet12 import harmonic_token_from_root


N = 12
STEP_TO_INDEX = {s: i for i, s in enumerate(STEP_ORDER)}
INDEX_TO_STEP = {i: s for i, s in enumerate(STEP_ORDER)}


@dataclass
class ResonanceMatrixRow:
    root_step: str
    harmonic_steps: List[str]
    harmonic_tokens: List[str]
    delta_steps: List[int]
    harmonic_weights: List[float]


def token_step(token: str) -> str:
    tok = normalize_token(token)
    if "." not in tok:
        raise ValueError(f"Bad token: {token}")
    return tok.split(".", 1)[1][0]


def step_distance_mod12(a: str, b: str) -> int:
    ia = STEP_TO_INDEX[a]
    ib = STEP_TO_INDEX[b]
    return (ib - ia) % 12


def harmonic_weight(
    harmonic_no: int,
    decay: float = 0.92,
    odd_bonus: float = 1.35,
    even_penalty: float = 0.90,
    h1_bonus: float = 1.80,
    h2_bonus: float = 1.00,
    h3_bonus: float = 1.45,
) -> float:
    w = decay ** (harmonic_no - 1)

    if harmonic_no == 1:
        w *= h1_bonus
    elif harmonic_no == 2:
        w *= h2_bonus
    elif harmonic_no == 3:
        w *= h3_bonus

    if harmonic_no % 2 == 1:
        w *= odd_bonus
    else:
        w *= even_penalty

    return w


def build_resonance_row(
    root_step: str,
    root_octave: str = "5",
    max_harmonic: int = 12,
) -> ResonanceMatrixRow:
    root_token = normalize_token(f"{root_octave}.{root_step}")

    harmonic_tokens: List[str] = []
    harmonic_steps: List[str] = []
    harmonic_weights: List[float] = []

    for h in range(1, max_harmonic + 1):
        tok = harmonic_token_from_root(root_token, h)
        step = token_step(tok)

        harmonic_tokens.append(tok)
        harmonic_steps.append(step)
        harmonic_weights.append(harmonic_weight(h))

    delta_steps: List[int] = []
    for i in range(len(harmonic_steps) - 1):
        delta_steps.append(step_distance_mod12(harmonic_steps[i], harmonic_steps[i + 1]))

    return ResonanceMatrixRow(
        root_step=root_step,
        harmonic_steps=harmonic_steps,
        harmonic_tokens=harmonic_tokens,
        delta_steps=delta_steps,
        harmonic_weights=harmonic_weights,
    )


def build_resonance_matrix(
    root_octave: str = "5",
    max_harmonic: int = 12,
) -> List[ResonanceMatrixRow]:
    rows: List[ResonanceMatrixRow] = []
    for step in STEP_ORDER:
        rows.append(
            build_resonance_row(
                root_step=step,
                root_octave=root_octave,
                max_harmonic=max_harmonic,
            )
        )
    return rows


def export_resonance_matrix_csv(
    out_csv: str | Path,
    root_octave: str = "5",
    max_harmonic: int = 12,
) -> None:
    rows = build_resonance_matrix(
        root_octave=root_octave,
        max_harmonic=max_harmonic,
    )

    flat_rows: List[Dict[str, object]] = []
    for row in rows:
        d: Dict[str, object] = {
            "root_step": row.root_step,
            "harmonic_steps": " | ".join(row.harmonic_steps),
            "harmonic_tokens": " | ".join(row.harmonic_tokens),
            "delta_steps": " | ".join(str(x) for x in row.delta_steps),
            "harmonic_weights": " | ".join(f"{x:.6f}" for x in row.harmonic_weights),
        }
        for i, s in enumerate(row.harmonic_steps, start=1):
            d[f"h{i}_step"] = s
        for i, t in enumerate(row.harmonic_tokens, start=1):
            d[f"h{i}_token"] = t
        for i, x in enumerate(row.delta_steps, start=1):
            d[f"d{i}_{i+1}"] = x
        for i, w in enumerate(row.harmonic_weights, start=1):
            d[f"w{i}"] = w

        flat_rows.append(d)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)


def _demo() -> None:
    export_resonance_matrix_csv(
        out_csv="tools/reports/resonance_matrix12.csv",
        root_octave="5",
        max_harmonic=12,
    )
    print("[OK] saved tools/reports/resonance_matrix12.csv")


if __name__ == "__main__":
    _demo()