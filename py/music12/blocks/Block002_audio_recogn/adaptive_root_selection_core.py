from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


# ============================================================
# DATA MODEL
# ============================================================

@dataclass(frozen=True)
class AdaptiveRootCandidate:
    """
    Candidate for adaptive root ranking in spiral/chain logic.

    This model is no longer phase/radial based.
    It ranks possible roots from chain evidence and spiral stability.
    """
    note_token: str

    # chain / spiral metrics
    raw_note_confidence: float
    spiral_coherence_score: float
    arc_stability_score: float
    arc_dispersion: float

    chain_support_ratio: float
    support_hits: int
    window_chain_match_score: float

    # comparative / regime metrics
    regime_id: int
    regime_name: str
    regime_confidence: float

    target_zone_ratio: float
    core_ratio: float
    mean_target_convergence_score: float

    competing_root_ratio: float

    # optional context
    template_match_score: float = 0.0
    composition_field_score: float = 0.0

    # metadata
    rank: int = 0
    source_cluster_id: int = 0


@dataclass(frozen=True)
class AdaptiveRootDecision:
    note_token: str
    adaptive_score: float

    regime_id: int
    regime_name: str

    weight_core: float
    weight_convergence: float
    weight_spiral: float
    weight_context: float

    core_term: float
    convergence_term: float
    spiral_term: float
    context_term: float

    raw_note_confidence: float
    spiral_coherence_score: float
    arc_stability_score: float
    arc_dispersion: float
    chain_support_ratio: float
    support_hits: int
    target_zone_ratio: float
    core_ratio: float
    mean_target_convergence_score: float
    window_chain_match_score: float
    competing_root_ratio: float

    rank: int
    source_cluster_id: int


# ============================================================
# HELPERS
# ============================================================

def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def normalize_convergence(x: float, scale: float = 15.0) -> float:
    return clamp01(float(x) / float(scale))


def normalize_raw_note_confidence(x: float, scale: float = 10.0) -> float:
    return clamp01(float(x) / float(scale))


def normalize_support_hits(x: int, scale: float = 8.0) -> float:
    return clamp01(float(x) / float(scale))


def normalize_arc_dispersion(x: float, scale: float = 2.0) -> float:
    """
    Lower dispersion is better.
    """
    return 1.0 - clamp01(float(x) / float(scale))


# ============================================================
# REGIME WEIGHTS
# ============================================================

def regime_weights(regime_id: int) -> tuple[float, float, float, float]:
    """
    Returns:
      weight_core, weight_convergence, weight_spiral, weight_context
    """
    # regime 3 = central stable plateau
    if regime_id == 3:
        return (0.40, 0.28, 0.22, 0.10)

    # regime 2 = rising stable zone
    if regime_id == 2:
        return (0.25, 0.28, 0.32, 0.15)

    # regime 4 = transition reorganization
    if regime_id == 4:
        return (0.18, 0.22, 0.35, 0.25)

    # regime 5 = high decentered zone
    if regime_id == 5:
        return (0.08, 0.16, 0.48, 0.28)

    # regime 1 = low anchor zone (default)
    return (0.18, 0.22, 0.34, 0.26)


# ============================================================
# TERM BUILDERS
# ============================================================

def build_core_term(c: AdaptiveRootCandidate) -> float:
    """
    Core-related support.

    Strong root candidates should have:
    - solid core ratio
    - reasonable target-zone presence
    - support in chain
    """
    return clamp01(
        0.40 * c.core_ratio +
        0.25 * c.target_zone_ratio +
        0.20 * c.chain_support_ratio +
        0.15 * normalize_support_hits(c.support_hits)
    )


def build_convergence_term(c: AdaptiveRootCandidate) -> float:
    """
    Convergence / root-emergence support.
    """
    return clamp01(
        0.45 * normalize_convergence(c.mean_target_convergence_score) +
        0.30 * normalize_raw_note_confidence(c.raw_note_confidence) +
        0.25 * c.window_chain_match_score
    )


def build_spiral_term(c: AdaptiveRootCandidate) -> float:
    """
    Spiral support:
    - high spiral coherence is good
    - high arc stability is good
    - low arc dispersion is good
    - competing root is bad
    """
    return clamp01(
        0.35 * c.spiral_coherence_score +
        0.30 * c.arc_stability_score +
        0.20 * normalize_arc_dispersion(c.arc_dispersion) +
        0.15 * (1.0 - c.competing_root_ratio)
    )


def build_context_term(c: AdaptiveRootCandidate) -> float:
    """
    Context support:
    - regime confidence
    - optional composition field
    - optional template match
    """
    return clamp01(
        0.45 * c.regime_confidence +
        0.35 * c.composition_field_score +
        0.20 * c.template_match_score
    )


# ============================================================
# ADAPTIVE SCORING
# ============================================================

def adaptive_score_candidate(c: AdaptiveRootCandidate) -> AdaptiveRootDecision:
    wc, wv, ws, wx = regime_weights(c.regime_id)

    core_term = build_core_term(c)
    convergence_term = build_convergence_term(c)
    spiral_term = build_spiral_term(c)
    context_term = build_context_term(c)

    adaptive_score = (
        wc * core_term +
        wv * convergence_term +
        ws * spiral_term +
        wx * context_term
    )

    # soft penalty for strong ambiguity
    adaptive_score *= (1.0 - 0.35 * clamp01(c.competing_root_ratio))

    return AdaptiveRootDecision(
        note_token=c.note_token,
        adaptive_score=adaptive_score,

        regime_id=c.regime_id,
        regime_name=c.regime_name,

        weight_core=wc,
        weight_convergence=wv,
        weight_spiral=ws,
        weight_context=wx,

        core_term=core_term,
        convergence_term=convergence_term,
        spiral_term=spiral_term,
        context_term=context_term,

        raw_note_confidence=c.raw_note_confidence,
        spiral_coherence_score=c.spiral_coherence_score,
        arc_stability_score=c.arc_stability_score,
        arc_dispersion=c.arc_dispersion,
        chain_support_ratio=c.chain_support_ratio,
        support_hits=c.support_hits,
        target_zone_ratio=c.target_zone_ratio,
        core_ratio=c.core_ratio,
        mean_target_convergence_score=c.mean_target_convergence_score,
        window_chain_match_score=c.window_chain_match_score,
        competing_root_ratio=c.competing_root_ratio,

        rank=c.rank,
        source_cluster_id=c.source_cluster_id,
    )


def rank_adaptive_root_candidates(
    candidates: Iterable[AdaptiveRootCandidate],
    *,
    max_notes: int = 8,
) -> list[AdaptiveRootDecision]:
    decisions = [adaptive_score_candidate(c) for c in candidates]
    decisions.sort(key=lambda x: x.adaptive_score, reverse=True)
    return decisions[:max_notes]