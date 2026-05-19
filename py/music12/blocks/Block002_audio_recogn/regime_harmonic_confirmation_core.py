from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from music12.core.notation12 import normalize_token, token_to_abs_semitone_index


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class HarmonicRelationCandidate:
    """
    Candidate note inside local chain context.

    Spiral/chain version:
    no phase/radial geometry is used here.
    """
    note_token: str

    spiral_coherence_score: float
    adaptive_score: float

    arc_stability_score: float
    arc_dispersion: float

    time_span_start_60: int
    time_span_end_60: int

    regime_id: int
    regime_name: str
    regime_confidence: float

    target_zone_ratio: float
    core_ratio: float
    mean_target_convergence_score: float

    chain_support_ratio: float
    support_hits: int
    window_chain_match_score: float

    competing_root_ratio: float

    rank: int = 0
    source_cluster_id: int = 0


@dataclass(frozen=True)
class HarmonicPairEvidence:
    left_note_token: str
    right_note_token: str

    interval_semitones: int
    arc_distance: float
    time_overlap_ratio: float

    pair_type: str
    pair_score: float


@dataclass(frozen=True)
class RegimeChainConfirmation:
    root_note_token: str
    regime_id: int
    regime_name: str

    confirmation_mode: str

    root_self_score: float
    pair_support_score: float
    structural_support_score: float
    confirmation_score: float

    supporting_pairs: tuple[HarmonicPairEvidence, ...]
    explanation: str


# ============================================================
# BASIC HELPERS
# ============================================================

def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def time_overlap_ratio(
    a0: int,
    a1: int,
    b0: int,
    b1: int,
) -> float:
    left = max(min(a0, a1), min(b0, b1))
    right = min(max(a0, a1), max(b0, b1))
    overlap = max(0, right - left)

    la = max(1, abs(a1 - a0))
    lb = max(1, abs(b1 - b0))
    denom = max(la, lb)
    return clamp01(overlap / denom)


def semitone_distance(a_token: str, b_token: str) -> int:
    a = token_to_abs_semitone_index(normalize_token(a_token))
    b = token_to_abs_semitone_index(normalize_token(b_token))
    return abs(b - a)


def arc_distance(a: HarmonicRelationCandidate, b: HarmonicRelationCandidate) -> float:
    """
    Since higher layers already transformed geometry into spiral metrics,
    we use dispersion/stability proxy instead of old radial distance.
    """
    return abs(float(a.arc_dispersion) - float(b.arc_dispersion))


# ============================================================
# INTERVAL / PAIR LOGIC
# ============================================================

def classify_interval_pair(interval_semitones: int) -> str:
    """
    Lightweight structural interval classification.

    This is not tonal functional analysis.
    This is structural support classification for chain confirmation.
    """
    good = {0, 3, 4, 5, 7, 8, 9, 12}
    medium = {1, 2, 6, 10, 11}

    if interval_semitones in good:
        return "STRUCTURAL"
    if interval_semitones in medium:
        return "WEAK_STRUCTURAL"
    return "OUTSIDE"


def build_pair_score(
    *,
    interval_semitones: int,
    overlap: float,
    left: HarmonicRelationCandidate,
    right: HarmonicRelationCandidate,
) -> float:
    pair_type = classify_interval_pair(interval_semitones)

    base = {
        "STRUCTURAL": 0.80,
        "WEAK_STRUCTURAL": 0.45,
        "OUTSIDE": 0.10,
    }[pair_type]

    score = (
        0.35 * base
        + 0.20 * overlap
        + 0.15 * left.spiral_coherence_score
        + 0.15 * right.spiral_coherence_score
        + 0.10 * left.chain_support_ratio
        + 0.05 * right.chain_support_ratio
    )
    return clamp01(score)


def build_pair_evidence(
    left: HarmonicRelationCandidate,
    right: HarmonicRelationCandidate,
) -> HarmonicPairEvidence:
    interval = semitone_distance(left.note_token, right.note_token)
    overlap = time_overlap_ratio(
        left.time_span_start_60,
        left.time_span_end_60,
        right.time_span_start_60,
        right.time_span_end_60,
    )
    pair_type = classify_interval_pair(interval)
    score = build_pair_score(
        interval_semitones=interval,
        overlap=overlap,
        left=left,
        right=right,
    )

    return HarmonicPairEvidence(
        left_note_token=left.note_token,
        right_note_token=right.note_token,
        interval_semitones=interval,
        arc_distance=arc_distance(left, right),
        time_overlap_ratio=overlap,
        pair_type=pair_type,
        pair_score=score,
    )


# ============================================================
# ROOT CONFIRMATION
# ============================================================

def build_root_self_score(c: HarmonicRelationCandidate) -> float:
    return clamp01(
        0.22 * c.adaptive_score +
        0.18 * c.spiral_coherence_score +
        0.16 * c.arc_stability_score +
        0.12 * (1.0 - c.arc_dispersion) +
        0.12 * c.chain_support_ratio +
        0.10 * c.window_chain_match_score +
        0.10 * c.regime_confidence
    )


def build_structural_support_score(
    root: HarmonicRelationCandidate,
    pairs: list[HarmonicPairEvidence],
) -> float:
    if not pairs:
        return clamp01(
            0.45 * root.chain_support_ratio +
            0.35 * root.window_chain_match_score +
            0.20 * root.target_zone_ratio
        )

    pair_mean = sum(p.pair_score for p in pairs) / len(pairs)

    return clamp01(
        0.35 * pair_mean +
        0.25 * root.chain_support_ratio +
        0.20 * root.window_chain_match_score +
        0.10 * root.core_ratio +
        0.10 * root.target_zone_ratio
    )


def build_confirmation_mode(
    root: HarmonicRelationCandidate,
    pair_support_score: float,
    structural_support_score: float,
) -> str:
    if root.adaptive_score >= 0.75 and pair_support_score >= 0.60 and structural_support_score >= 0.60:
        return "CHAIN_REGIME_CONFIRMED"

    if root.adaptive_score >= 0.55 and structural_support_score >= 0.45:
        return "CHAIN_REGIME_PARTIAL"

    if root.adaptive_score >= 0.35:
        return "CHAIN_REGIME_WEAK"

    return "CHAIN_REGIME_UNCERTAIN"


def confirm_root_against_group(
    root: HarmonicRelationCandidate,
    group: Iterable[HarmonicRelationCandidate],
) -> RegimeChainConfirmation:
    others = [x for x in group if x.note_token != root.note_token]

    pair_evidences: list[HarmonicPairEvidence] = []
    for other in others:
        pair = build_pair_evidence(root, other)
        if pair.pair_type != "OUTSIDE" or pair.pair_score >= 0.35:
            pair_evidences.append(pair)

    root_self_score = build_root_self_score(root)

    if pair_evidences:
        pair_support_score = clamp01(
            sum(p.pair_score for p in pair_evidences) / len(pair_evidences)
        )
    else:
        pair_support_score = 0.0

    structural_support_score = build_structural_support_score(root, pair_evidences)

    confirmation_score = clamp01(
        0.45 * root_self_score +
        0.25 * pair_support_score +
        0.30 * structural_support_score
    )

    mode = build_confirmation_mode(
        root=root,
        pair_support_score=pair_support_score,
        structural_support_score=structural_support_score,
    )

    explanation = (
        f"mode={mode}; "
        f"adaptive={root.adaptive_score:.3f}; "
        f"spiral={root.spiral_coherence_score:.3f}; "
        f"arc_stability={root.arc_stability_score:.3f}; "
        f"chain_support={root.chain_support_ratio:.3f}; "
        f"pair_support={pair_support_score:.3f}; "
        f"structural={structural_support_score:.3f}"
    )

    return RegimeChainConfirmation(
        root_note_token=root.note_token,
        regime_id=root.regime_id,
        regime_name=root.regime_name,
        confirmation_mode=mode,
        root_self_score=root_self_score,
        pair_support_score=pair_support_score,
        structural_support_score=structural_support_score,
        confirmation_score=confirmation_score,
        supporting_pairs=tuple(pair_evidences),
        explanation=explanation,
    )


def rank_regime_confirmations(
    candidates: Iterable[HarmonicRelationCandidate],
    *,
    max_notes: int = 8,
) -> list[RegimeChainConfirmation]:
    group = list(candidates)
    confirmations = [confirm_root_against_group(root, group) for root in group]
    confirmations.sort(key=lambda x: x.confirmation_score, reverse=True)
    return confirmations[:max_notes]