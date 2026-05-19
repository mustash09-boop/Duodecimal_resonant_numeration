from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from music12.core.notation12 import (
    hz_to_token,
    token_to_abs_semitone_index,
)

# ============================================================
# CONSTANTS
# ============================================================

ANCHOR_TOKEN = "9.A"
ANCHOR_HZ = 440.0

DEFAULT_HARMONICS = [1, 2, 3, 4, 5, 6, 7, 8]
DEFAULT_RANGE_MIN_TOKEN = "5.A"
DEFAULT_RANGE_MAX_TOKEN = "11.1"


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass(frozen=True)
class HarmonicChainPoint:
    harmonic_index: int
    expected_hz: float
    expected_token: str

    @property
    def label(self) -> str:
        return f"h{self.harmonic_index}:{self.expected_token}"


@dataclass(frozen=True)
class TheoreticalRCChain:
    root_token: str
    root_hz: float
    root_abs_index: int
    harmonics: tuple[HarmonicChainPoint, ...]

    @property
    def chain_tokens(self) -> tuple[str, ...]:
        return tuple(h.expected_token for h in self.harmonics)

    @property
    def chain_labels(self) -> tuple[str, ...]:
        return tuple(h.label for h in self.harmonics)

    @property
    def chain_string(self) -> str:
        return " -> ".join(self.chain_tokens)

    @property
    def chain_label_string(self) -> str:
        return " | ".join(self.chain_labels)

    def get_harmonic_point(self, harmonic_index: int) -> Optional[HarmonicChainPoint]:
        for hp in self.harmonics:
            if hp.harmonic_index == harmonic_index:
                return hp
        return None


@dataclass(frozen=True)
class ChainMatchResult:
    root_token: str
    observed_tokens: tuple[str, ...]
    matched_harmonics: tuple[int, ...]
    missing_harmonics: tuple[int, ...]
    extra_tokens: tuple[str, ...]
    score: float
    chain_string: str
    chain_label_string: str


# ============================================================
# BASIC CONVERSION
# ============================================================

def token_to_hz(
    token: str,
    *,
    anchor_token: str = ANCHOR_TOKEN,
    anchor_hz: float = ANCHOR_HZ,
) -> float:
    idx = token_to_abs_semitone_index(token)
    anchor_idx = token_to_abs_semitone_index(anchor_token)
    semitone_delta = idx - anchor_idx
    return anchor_hz * (2.0 ** (semitone_delta / 12.0))


def harmonic_hz(root_hz: float, harmonic_index: int) -> float:
    return root_hz * float(harmonic_index)


# ============================================================
# RANGE BUILDING
# ============================================================

def build_token_range(
    token_min: str = DEFAULT_RANGE_MIN_TOKEN,
    token_max: str = DEFAULT_RANGE_MAX_TOKEN,
    *,
    anchor_token: str = ANCHOR_TOKEN,
    anchor_hz: float = ANCHOR_HZ,
) -> List[str]:
    idx_min = token_to_abs_semitone_index(token_min)
    idx_max = token_to_abs_semitone_index(token_max)

    if idx_max < idx_min:
        raise ValueError(f"token_max {token_max} is below token_min {token_min}")

    anchor_idx = token_to_abs_semitone_index(anchor_token)

    out: List[str] = []
    for abs_idx in range(idx_min, idx_max + 1):
        hz = anchor_hz * (2.0 ** ((abs_idx - anchor_idx) / 12.0))
        token = hz_to_token(
            hz,
            a4_hz=anchor_hz,
            anchor_token=anchor_token,
            micro_depth=2,
            force_micro_dash_when_exact=True,
        )
        out.append(token)

    return out


# ============================================================
# CHAIN BUILDING
# ============================================================

def build_theoretical_chain(
    root_token: str,
    *,
    harmonics: Sequence[int] = DEFAULT_HARMONICS,
    anchor_token: str = ANCHOR_TOKEN,
    anchor_hz: float = ANCHOR_HZ,
) -> TheoreticalRCChain:
    root_hz = token_to_hz(root_token, anchor_token=anchor_token, anchor_hz=anchor_hz)
    root_abs_index = token_to_abs_semitone_index(root_token)

    harmonic_points: List[HarmonicChainPoint] = []

    for h in harmonics:
        expected_hz = harmonic_hz(root_hz, h)
        expected_token = hz_to_token(
            expected_hz,
            a4_hz=anchor_hz,
            anchor_token=anchor_token,
            micro_depth=2,
            force_micro_dash_when_exact=True,
        )
        harmonic_points.append(
            HarmonicChainPoint(
                harmonic_index=int(h),
                expected_hz=expected_hz,
                expected_token=expected_token,
            )
        )

    return TheoreticalRCChain(
        root_token=root_token,
        root_hz=root_hz,
        root_abs_index=root_abs_index,
        harmonics=tuple(harmonic_points),
    )


def build_full_theoretical_chain_table(
    token_min: str = DEFAULT_RANGE_MIN_TOKEN,
    token_max: str = DEFAULT_RANGE_MAX_TOKEN,
    *,
    harmonics: Sequence[int] = DEFAULT_HARMONICS,
    anchor_token: str = ANCHOR_TOKEN,
    anchor_hz: float = ANCHOR_HZ,
) -> Dict[str, TheoreticalRCChain]:
    tokens = build_token_range(
        token_min=token_min,
        token_max=token_max,
        anchor_token=anchor_token,
        anchor_hz=anchor_hz,
    )

    table: Dict[str, TheoreticalRCChain] = {}
    for token in tokens:
        table[token] = build_theoretical_chain(
            token,
            harmonics=harmonics,
            anchor_token=anchor_token,
            anchor_hz=anchor_hz,
        )
    return table


# ============================================================
# LOOKUP HELPERS
# ============================================================

def get_chain(
    root_token: str,
    *,
    table: Optional[Dict[str, TheoreticalRCChain]] = None,
    token_min: str = DEFAULT_RANGE_MIN_TOKEN,
    token_max: str = DEFAULT_RANGE_MAX_TOKEN,
    harmonics: Sequence[int] = DEFAULT_HARMONICS,
) -> TheoreticalRCChain:
    if table is None:
        table = build_full_theoretical_chain_table(
            token_min=token_min,
            token_max=token_max,
            harmonics=harmonics,
        )

    if root_token not in table:
        raise KeyError(f"Root token not found in theoretical chain table: {root_token}")

    return table[root_token]


def find_matching_roots_for_observed_token(
    observed_token: str,
    *,
    table: Optional[Dict[str, TheoreticalRCChain]] = None,
    token_min: str = DEFAULT_RANGE_MIN_TOKEN,
    token_max: str = DEFAULT_RANGE_MAX_TOKEN,
    harmonics: Sequence[int] = DEFAULT_HARMONICS,
) -> List[dict]:
    if table is None:
        table = build_full_theoretical_chain_table(
            token_min=token_min,
            token_max=token_max,
            harmonics=harmonics,
        )

    matches: List[dict] = []

    for root_token, chain in table.items():
        for hp in chain.harmonics:
            if hp.expected_token == observed_token:
                matches.append(
                    {
                        "root_token": root_token,
                        "root_hz": chain.root_hz,
                        "harmonic_index": hp.harmonic_index,
                        "expected_hz": hp.expected_hz,
                        "expected_token": hp.expected_token,
                        "chain_string": chain.chain_string,
                    }
                )

    matches.sort(key=lambda x: (x["harmonic_index"], x["root_hz"]))
    return matches


# ============================================================
# CHAIN SCORING
# ============================================================

def normalize_observed_tokens(tokens: Sequence[str]) -> List[str]:
    out: List[str] = []
    for t in tokens:
        tt = str(t).strip()
        if tt:
            out.append(tt)
    return out


def score_observed_tokens_against_chain(
    observed_tokens: Sequence[str],
    root_token: str,
    *,
    table: Optional[Dict[str, TheoreticalRCChain]] = None,
    token_min: str = DEFAULT_RANGE_MIN_TOKEN,
    token_max: str = DEFAULT_RANGE_MAX_TOKEN,
    harmonics: Sequence[int] = DEFAULT_HARMONICS,
    hit_weight: float = 2.0,
    root_bonus: float = 1.5,
    missing_penalty: float = 0.5,
    extra_penalty: float = 0.25,
) -> ChainMatchResult:
    chain = get_chain(
        root_token,
        table=table,
        token_min=token_min,
        token_max=token_max,
        harmonics=harmonics,
    )

    observed = normalize_observed_tokens(observed_tokens)
    observed_set = set(observed)

    matched_harmonics: List[int] = []
    missing_harmonics: List[int] = []

    for hp in chain.harmonics:
        if hp.expected_token in observed_set:
            matched_harmonics.append(hp.harmonic_index)
        else:
            missing_harmonics.append(hp.harmonic_index)

    chain_token_set = set(chain.chain_tokens)
    extras = sorted(tok for tok in observed_set if tok not in chain_token_set)

    score = (
        len(matched_harmonics) * hit_weight
        + (root_bonus if 1 in matched_harmonics else 0.0)
        - len(missing_harmonics) * missing_penalty
        - len(extras) * extra_penalty
    )

    return ChainMatchResult(
        root_token=root_token,
        observed_tokens=tuple(observed),
        matched_harmonics=tuple(matched_harmonics),
        missing_harmonics=tuple(missing_harmonics),
        extra_tokens=tuple(extras),
        score=score,
        chain_string=chain.chain_string,
        chain_label_string=chain.chain_label_string,
    )


def score_observed_supports_against_theoretical_chain(
    *,
    chosen_rc_note: str,
    strongest_peak_note: str = "",
    support_tokens: Sequence[str] = (),
    root_token: Optional[str] = None,
    table: Optional[Dict[str, TheoreticalRCChain]] = None,
    token_min: str = DEFAULT_RANGE_MIN_TOKEN,
    token_max: str = DEFAULT_RANGE_MAX_TOKEN,
    harmonics: Sequence[int] = DEFAULT_HARMONICS,
) -> ChainMatchResult:
    """
    Main helper for Block002:
    compares observed local evidence against one theoretical root chain.

    Observed evidence is assembled from:
    - chosen_rc_note
    - strongest_peak_note
    - support_h2..support_h8 tokens
    """
    observed_tokens: List[str] = []

    if chosen_rc_note:
        observed_tokens.append(chosen_rc_note)
    if strongest_peak_note:
        observed_tokens.append(strongest_peak_note)

    for tok in support_tokens:
        tok = str(tok).strip()
        if tok:
            observed_tokens.append(tok)

    if not root_token:
        if not chosen_rc_note:
            raise ValueError("root_token is required if chosen_rc_note is empty")
        root_token = chosen_rc_note

    return score_observed_tokens_against_chain(
        observed_tokens,
        root_token=root_token,
        table=table,
        token_min=token_min,
        token_max=token_max,
        harmonics=harmonics,
    )


def rank_possible_roots_for_observed_supports(
    *,
    chosen_rc_note: str,
    strongest_peak_note: str = "",
    support_tokens: Sequence[str] = (),
    table: Optional[Dict[str, TheoreticalRCChain]] = None,
    token_min: str = DEFAULT_RANGE_MIN_TOKEN,
    token_max: str = DEFAULT_RANGE_MAX_TOKEN,
    harmonics: Sequence[int] = DEFAULT_HARMONICS,
    top_k: int = 10,
) -> List[ChainMatchResult]:
    """
    Useful for low notes:
    rank all possible theoretical roots by how well they explain observed evidence.
    """
    if table is None:
        table = build_full_theoretical_chain_table(
            token_min=token_min,
            token_max=token_max,
            harmonics=harmonics,
        )

    results: List[ChainMatchResult] = []
    for root_token in table.keys():
        res = score_observed_supports_against_theoretical_chain(
            chosen_rc_note=chosen_rc_note,
            strongest_peak_note=strongest_peak_note,
            support_tokens=support_tokens,
            root_token=root_token,
            table=table,
            token_min=token_min,
            token_max=token_max,
            harmonics=harmonics,
        )
        results.append(res)

    results.sort(
        key=lambda r: (
            r.score,
            len(r.matched_harmonics),
            -len(r.missing_harmonics),
            -len(r.extra_tokens),
        ),
        reverse=True,
    )
    return results[:top_k]


# ============================================================
# EXPORT / VISUAL INSPECTION
# ============================================================

def export_theoretical_chain_table_csv(
    out_csv: str | Path,
    *,
    table: Optional[Dict[str, TheoreticalRCChain]] = None,
    token_min: str = DEFAULT_RANGE_MIN_TOKEN,
    token_max: str = DEFAULT_RANGE_MAX_TOKEN,
    harmonics: Sequence[int] = DEFAULT_HARMONICS,
) -> Path:
    if table is None:
        table = build_full_theoretical_chain_table(
            token_min=token_min,
            token_max=token_max,
            harmonics=harmonics,
        )

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "root_token",
        "root_hz",
        "chain_string",
        "chain_label_string",
    ]
    for h in harmonics:
        fieldnames.extend([f"h{h}_token", f"h{h}_hz"])

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for root_token, chain in table.items():
            row = {
                "root_token": root_token,
                "root_hz": round(chain.root_hz, 6),
                "chain_string": chain.chain_string,
                "chain_label_string": chain.chain_label_string,
            }
            for hp in chain.harmonics:
                row[f"h{hp.harmonic_index}_token"] = hp.expected_token
                row[f"h{hp.harmonic_index}_hz"] = round(hp.expected_hz, 6)

            writer.writerow(row)

    return out_csv


def export_theoretical_chain_table_txt(
    out_txt: str | Path,
    *,
    table: Optional[Dict[str, TheoreticalRCChain]] = None,
    token_min: str = DEFAULT_RANGE_MIN_TOKEN,
    token_max: str = DEFAULT_RANGE_MAX_TOKEN,
    harmonics: Sequence[int] = DEFAULT_HARMONICS,
) -> Path:
    if table is None:
        table = build_full_theoretical_chain_table(
            token_min=token_min,
            token_max=token_max,
            harmonics=harmonics,
        )

    out_txt = Path(out_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    with out_txt.open("w", encoding="utf-8") as f:
        f.write("THEORETICAL RC CHAINS\n")
        f.write("=" * 100 + "\n")
        f.write(f"range: {token_min} .. {token_max}\n")
        f.write(f"harmonics: {list(harmonics)}\n\n")

        for root_token, chain in table.items():
            f.write(f"{root_token}  ({chain.root_hz:.6f} Hz)\n")
            f.write(f"  chain: {chain.chain_string}\n")
            f.write(f"  labels: {chain.chain_label_string}\n")
            f.write("\n")

    return out_txt


# ============================================================
# PREBUILT DEFAULT TABLE
# ============================================================

THEORETICAL_RC_CHAIN_TABLE_5A_TO_11_1 = build_full_theoretical_chain_table(
    token_min="5.A",
    token_max="11.1",
    harmonics=DEFAULT_HARMONICS,
)