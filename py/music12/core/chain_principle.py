# -*- coding: utf-8 -*-
"""
music12.core.chain_principle

Formal principle of harmonic chain in the music12 system.

This module contains no executable analytical logic.
Its purpose is to fix the missing bridge law of the project:

    observation -> curve -> chain -> note

Without this bridge, the system easily falls back into naive DSP logic,
where a peak, strongest bin, or provisional f0 guess is treated as if it
were already a note.

That is forbidden by the project ontology.

----------------------------------------------------------------------
CORE LAW
----------------------------------------------------------------------

In music12, a NOTE must not be recognized directly from:
    - a single peak
    - a strongest bin
    - a raw frequency estimate
    - a token string by itself
    - an isolated trajectory without harmonic confirmation

A NOTE is recognized only through a sufficiently confirmed harmonic chain.

Therefore:

    observation  -> raw evidence
    curve        -> temporal resonant continuity
    chain        -> causal harmonic organization
    note         -> inferred excitation source

This order is mandatory.

----------------------------------------------------------------------
WHY CHAIN IS NECESSARY
----------------------------------------------------------------------

A curve alone is not yet a note.

A curve may represent:
    - one partial of a note
    - a transient trace
    - a medium-dependent resonance
    - a noisy or accidental structure
    - a sideband-like or inharmonic fragment

Only a chain can connect local observations into a causal explanation.

The chain answers the question:

    which excitation source best explains the observed family
    of resonant components?

Therefore the chain is not an optional convenience.
It is the required inferential bridge between observed resonance and note identity.

----------------------------------------------------------------------
WHAT A CHAIN IS
----------------------------------------------------------------------

A harmonic chain in music12 is a causally organized family of resonant
support centered around a hypothesized source/root.

It is evaluated not as a random list of visible components, but as an
internally meaningful structure with at least the following dimensions:

    1. interval relation
    2. temporal continuity
    3. support sufficiency
    4. explanatory power

In simplified project language:

    chain = harmonically organized causal support of a note hypothesis

----------------------------------------------------------------------
THREE LEVELS OF CHAIN
----------------------------------------------------------------------

The system should distinguish three levels:

1. THEORETICAL CHAIN
   Ideal structural expectation for a given root.
   This is the project-side model of what harmonic support should look like.

2. OBSERVED CHAIN
   Real support extracted from signal observations, curves, and local evidence.

3. CONFIRMED CHAIN
   Observed chain that is sufficiently stable and explanatory to justify
   recognition of a note.

Only a confirmed chain is sufficient for note recognition.

----------------------------------------------------------------------
RELATION TO CURVES
----------------------------------------------------------------------

A curve is a temporal trajectory in resonant space.

A chain is a higher-order interpretive structure that may include:
    - one curve
    - several related curves
    - local harmonic supports around one root hypothesis

Thus:

    curve != chain
    chain may be built from curve evidence
    note may be recognized only after chain confirmation

So the project must not collapse:
    curve -> note

The required bridge is:
    curve -> chain -> note

----------------------------------------------------------------------
ROOT INFERENCE
----------------------------------------------------------------------

The root of a note must be inferred from the chain.

It must not be fixed prematurely from:
    - lowest visible component
    - strongest peak
    - naive root guess
    - convenience heuristic alone

A provisional root hypothesis is allowed for exploration,
but final note recognition requires chain confirmation.

Therefore:

    early_root_guess is allowed as a hypothesis
    final_root_assignment without chain is forbidden

----------------------------------------------------------------------
RELATION TO TOKEN
----------------------------------------------------------------------

A token is not the note itself.

A token is the canonical coordinate-language representation of a recognized
or hypothesized note in the project notation system.

Therefore:

    token = representation
    note  = inferred excitation source

A token may be attached:
    - to a hypothesis
    - to a chain root candidate
    - to a confirmed note

But the presence of a token string does not itself prove that a note exists.

----------------------------------------------------------------------
POLYPHONY
----------------------------------------------------------------------

In polyphony, the system must not think in terms of:
    "many peaks = many notes"

Instead it must think in terms of:
    "coexisting chains = coexisting note causes"

Thus, multiple chains may coexist in the same time region.

Polyphony is the coexistence of several distinguishable causal harmonic chains,
not merely the coexistence of several local maxima.

----------------------------------------------------------------------
CONSEQUENCE FOR BLOCK002
----------------------------------------------------------------------

Block002 must follow this order:

    observed signal
    -> resonance observations
    -> curve construction
    -> chain construction
    -> note recognition

The block must not skip the chain stage when assigning final notes.

If a chain is absent or insufficient, the correct result is not a forced note,
but an unresolved hypothesis.

----------------------------------------------------------------------
PROJECT LAW
----------------------------------------------------------------------

In music12:

    peak is not note
    curve is not note
    chain is required for note
    root must be justified by chain
    token is representation, not ontology

The project recognizes a NOTE only as the root of a sufficiently confirmed
harmonic chain in the internal coordinate system.
"""

# ----------------------------------------------------------------------
# Project law flags
# ----------------------------------------------------------------------

CHAIN_REQUIRED_FOR_NOTE = True
PEAK_IS_NOT_NOTE = True
CURVE_IS_NOT_NOTE = True
TOKEN_IS_NOT_NOTE = True

ROOT_MUST_BE_INFERRED_FROM_CHAIN = True
EARLY_ROOT_GUESS_ALLOWED = True
FINAL_ROOT_WITHOUT_CHAIN_FORBIDDEN = True

THEORETICAL_CHAIN_EXISTS = True
OBSERVED_CHAIN_EXISTS = True
CONFIRMED_CHAIN_REQUIRED_FOR_NOTE = True

MULTIPLE_CHAINS_ALLOWED = True
POLYPHONY_IS_CHAIN_COEXISTENCE = True

MANDATORY_INFERENCE_ORDER = (
    "observation",
    "curve",
    "chain",
    "note",
)


def chain_required_for_note() -> bool:
    return CHAIN_REQUIRED_FOR_NOTE


def final_root_without_chain_forbidden() -> bool:
    return FINAL_ROOT_WITHOUT_CHAIN_FORBIDDEN


def mandatory_inference_order() -> tuple[str, ...]:
    return MANDATORY_INFERENCE_ORDER


def chain_principle_flags() -> dict[str, object]:
    return {
        "CHAIN_REQUIRED_FOR_NOTE": CHAIN_REQUIRED_FOR_NOTE,
        "PEAK_IS_NOT_NOTE": PEAK_IS_NOT_NOTE,
        "CURVE_IS_NOT_NOTE": CURVE_IS_NOT_NOTE,
        "TOKEN_IS_NOT_NOTE": TOKEN_IS_NOT_NOTE,
        "ROOT_MUST_BE_INFERRED_FROM_CHAIN": ROOT_MUST_BE_INFERRED_FROM_CHAIN,
        "EARLY_ROOT_GUESS_ALLOWED": EARLY_ROOT_GUESS_ALLOWED,
        "FINAL_ROOT_WITHOUT_CHAIN_FORBIDDEN": FINAL_ROOT_WITHOUT_CHAIN_FORBIDDEN,
        "THEORETICAL_CHAIN_EXISTS": THEORETICAL_CHAIN_EXISTS,
        "OBSERVED_CHAIN_EXISTS": OBSERVED_CHAIN_EXISTS,
        "CONFIRMED_CHAIN_REQUIRED_FOR_NOTE": CONFIRMED_CHAIN_REQUIRED_FOR_NOTE,
        "MULTIPLE_CHAINS_ALLOWED": MULTIPLE_CHAINS_ALLOWED,
        "POLYPHONY_IS_CHAIN_COEXISTENCE": POLYPHONY_IS_CHAIN_COEXISTENCE,
        "MANDATORY_INFERENCE_ORDER": MANDATORY_INFERENCE_ORDER,
    }