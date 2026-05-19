from __future__ import annotations

"""
music12.core.harmonic_alphabet12
--------------------------------

Core module for building a 12-radix harmonic alphabet on top of the
canonical notation defined in music12.core.notation12.

Design goals:
1. Use project SSOT from notation12.py
2. No zero in musical step alphabet
3. Keep canonical token formatting compatible with notation12
4. Provide a temporary physical harmonic mapping layer
5. Allow later replacement by an exact project-specific harmonic law

This module is CORE, not Block002-specific.
"""

from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Sequence
import csv
import math
from pathlib import Path

from music12.core.notation12 import (
    STEP_ORDER,
    normalize_token,
    token_to_abs_semitone_index,
    abs_semitone_to_token,
)

# ------------------------------------------------------------
# Types
# ------------------------------------------------------------

Token12 = str


@dataclass
class HarmonicEntry:
    harmonic_no: int
    root_token: Token12
    harmonic_token: Token12
    approx_semitone_offset: int
    exact_ratio: float
    exact_cents: float
    present_in_chain: bool = True


@dataclass
class HarmonicAlphabetRow:
    root_token: Token12
    root_step: str
    chain: List[Token12]
    entries: List[HarmonicEntry]


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def ratio_to_cents(r: float) -> float:
    return 1200.0 * math.log2(r)


def harmonic_semitone_offset(harmonic_no: int) -> int:
    """
    Temporary physical approximation:
    semitone_offset ~= round(12 * log2(h))

    IMPORTANT:
    This is only the first approximation layer.
    Later it should be replaced by your exact 12-radix harmonic law.
    """
    if harmonic_no < 1:
        raise ValueError("harmonic_no must be >= 1")
    return int(round(12.0 * math.log2(harmonic_no)))


def token_step(token: Token12) -> str:
    tok = normalize_token(token)
    if "." not in tok:
        raise ValueError(f"Bad token12: {token}")
    return tok.split(".", 1)[1]


def shift_token_by_semitones(token: Token12, semitone_offset: int) -> Token12:
    """
    Shift canonical token by N semitone-like 12-step units.
    """
    root_idx = token_to_abs_semitone_index(normalize_token(token))
    out_idx = root_idx + int(semitone_offset)
    return normalize_token(abs_semitone_to_token(out_idx))


def harmonic_token_from_root(root_token: Token12, harmonic_no: int) -> Token12:
    """
    Root token -> expected harmonic token by temporary physical law.
    """
    semitone_offset = harmonic_semitone_offset(harmonic_no)
    return shift_token_by_semitones(root_token, semitone_offset)


# ------------------------------------------------------------
# Core harmonic alphabet builders
# ------------------------------------------------------------

def build_harmonic_entries_for_root(
    root_token: Token12,
    max_harmonic: int = 16,
) -> List[HarmonicEntry]:
    """
    Build harmonic entries for a single root token.
    """
    root_token = normalize_token(root_token)

    out: List[HarmonicEntry] = []
    for h in range(1, max_harmonic + 1):
        tok = harmonic_token_from_root(root_token, h)
        entry = HarmonicEntry(
            harmonic_no=h,
            root_token=root_token,
            harmonic_token=tok,
            approx_semitone_offset=harmonic_semitone_offset(h),
            exact_ratio=float(h),
            exact_cents=ratio_to_cents(float(h)),
            present_in_chain=True,
        )
        out.append(entry)
    return out


def build_harmonic_alphabet_for_octave(
    root_octave: str = "5",
    max_harmonic: int = 16,
) -> List[HarmonicAlphabetRow]:
    """
    Build harmonic alphabet for all 12 project steps in one octave.

    Example roots:
        5.1, 5.2, ... 5.C
    """
    rows: List[HarmonicAlphabetRow] = []

    for step in STEP_ORDER:
        root_token = normalize_token(f"{root_octave}.{step}")
        entries = build_harmonic_entries_for_root(
            root_token=root_token,
            max_harmonic=max_harmonic,
        )
        chain = [e.harmonic_token for e in entries]
        rows.append(
            HarmonicAlphabetRow(
                root_token=root_token,
                root_step=step,
                chain=chain,
                entries=entries,
            )
        )

    return rows


# ------------------------------------------------------------
# Template-based alphabet support
# ------------------------------------------------------------

def rotate_step(step: str, semitone_shift: int) -> str:
    i = STEP_ORDER.index(step)
    return STEP_ORDER[(i + semitone_shift) % 12]


def transpose_template_chain(
    base_root_step: str,
    target_root_step: str,
    template_chain_steps: Sequence[str],
    base_root_octave: str = "5",
) -> List[Token12]:
    """
    Build a token chain from a pure step-template by transposition.

    This is the hook for the REAL 12-radix alphabet you want to use later.

    Example:
        base_root_step = "1"
        target_root_step = "5"
        template_chain_steps = ["1","1","8","1","5","8","B",...]

    Then the whole chain is transposed so that root becomes target_root_step.
    Octave handling remains simple for now; exact octave placement should be
    upgraded later with your true harmonic law.
    """
    if base_root_step not in STEP_ORDER:
        raise ValueError(f"Unknown base_root_step: {base_root_step}")
    if target_root_step not in STEP_ORDER:
        raise ValueError(f"Unknown target_root_step: {target_root_step}")

    shift = STEP_ORDER.index(target_root_step) - STEP_ORDER.index(base_root_step)

    # Temporary octave logic:
    # build within one abstract layer first, then normalize by semitone transfer
    root_token = normalize_token(f"{base_root_octave}.{base_root_step}")
    target_root_token = normalize_token(f"{base_root_octave}.{target_root_step}")

    root_abs = token_to_abs_semitone_index(root_token)
    target_abs = token_to_abs_semitone_index(target_root_token)
    delta_abs = target_abs - root_abs

    out: List[Token12] = []
    for step in template_chain_steps:
        # build naive token in base octave
        tok = normalize_token(f"{base_root_octave}.{step}")
        shifted = shift_token_by_semitones(tok, delta_abs)
        out.append(shifted)

    return out


# ------------------------------------------------------------
# CSV / flat exports
# ------------------------------------------------------------

def flatten_alphabet_rows(rows: Sequence[HarmonicAlphabetRow]) -> List[Dict[str, object]]:
    """
    Flat row per harmonic per root, good for CSV export.
    """
    flat: List[Dict[str, object]] = []
    for row in rows:
        for e in row.entries:
            flat.append(
                {
                    "root_token": row.root_token,
                    "root_step": row.root_step,
                    "harmonic_no": e.harmonic_no,
                    "harmonic_token": e.harmonic_token,
                    "approx_semitone_offset": e.approx_semitone_offset,
                    "exact_ratio": e.exact_ratio,
                    "exact_cents": e.exact_cents,
                    "present_in_chain": e.present_in_chain,
                }
            )
    return flat


def compact_alphabet_table(rows: Sequence[HarmonicAlphabetRow]) -> List[Dict[str, object]]:
    """
    One row per root, with h1..hN columns.
    """
    out: List[Dict[str, object]] = []
    for row in rows:
        data: Dict[str, object] = {
            "root_token": row.root_token,
            "root_step": row.root_step,
            "chain": " | ".join(row.chain),
        }
        for e in row.entries:
            data[f"h{e.harmonic_no}"] = e.harmonic_token
        out.append(data)
    return out


def save_csv(rows: Sequence[Dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to save")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------
# Project-friendly public builders
# ------------------------------------------------------------

def build_default_harmonic_alphabet(
    root_octave: str = "5",
    max_harmonic: int = 16,
) -> List[HarmonicAlphabetRow]:
    """
    Default project builder:
    uses notation12 canonical tokens + temporary physical harmonic offsets.
    """
    return build_harmonic_alphabet_for_octave(
        root_octave=root_octave,
        max_harmonic=max_harmonic,
    )


def export_default_harmonic_alphabet(
    out_csv_flat: str | Path,
    out_csv_compact: Optional[str | Path] = None,
    root_octave: str = "5",
    max_harmonic: int = 16,
) -> None:
    rows = build_default_harmonic_alphabet(
        root_octave=root_octave,
        max_harmonic=max_harmonic,
    )

    out_csv_flat = Path(out_csv_flat)
    save_csv(flatten_alphabet_rows(rows), out_csv_flat)

    if out_csv_compact is not None:
        out_csv_compact = Path(out_csv_compact)
        save_csv(compact_alphabet_table(rows), out_csv_compact)


# ------------------------------------------------------------
# CLI-like local test
# ------------------------------------------------------------

def _demo() -> None:
    flat = Path("tools/reports/harmonic_alphabet12_flat.csv")
    compact = Path("tools/reports/harmonic_alphabet12_compact.csv")
    export_default_harmonic_alphabet(
        out_csv_flat=flat,
        out_csv_compact=compact,
        root_octave="5",
        max_harmonic=16,
    )
    print(f"[OK] saved {flat}")
    print(f"[OK] saved {compact}")


if __name__ == "__main__":
    _demo()