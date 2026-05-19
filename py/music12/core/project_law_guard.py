# -*- coding: utf-8 -*-
"""
music12.core.project_law_guard

Unified access point for the active project laws of music12.

Purpose
-------
This module does NOT replace the existing philosophy/principle files.
It acts as a compact runtime-facing registry / guard layer that:

    1. imports the already established principle modules
    2. exposes their laws through one stable access point
    3. adds only the missing chain-centered bridge law
    4. gives demons and blocks a single place to query project ontology

In other words:

    principle modules = constitution
    project_law_guard = regulatory access point

This module is intentionally light.
It should not become a second philosophy system.
It should only centralize what the project already decided.

----------------------------------------------------------------------
ACTIVE SOURCES OF LAW
----------------------------------------------------------------------

The guard relies on the following principle modules:

    - coordinate_system_principle
    - micro12_principle
    - resonance_curve_principle
    - resonant_medium_principle
    - chain_principle

These modules are primarily textual / declarative.
The guard translates them into a stable set of boolean laws and helpers
usable from code, demons, and reports.

----------------------------------------------------------------------
MAIN IDEA
----------------------------------------------------------------------

The project must be able to ask simple questions such as:

    - Is chain required for note recognition?
    - Is token identical to note?
    - Is early root guess allowed?
    - Is final root without chain forbidden?
    - Is internal coordinate system primary?

and receive a stable answer from one place.

This module provides that place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

from music12.core import chain_principle as _chain


# ----------------------------------------------------------------------
# Stable registry keys
# ----------------------------------------------------------------------

LAW_COORDINATE_SYSTEM_PRIMARY = "coordinate_system_primary"
LAW_MICROSTRUCTURE_RECURSIVE_BASE12 = "microstructure_recursive_base12"
LAW_NOTE_IS_TRAJECTORY_BASED = "note_is_trajectory_based"
LAW_NOTE_AND_INSTRUMENT_DISTINCT = "note_and_instrument_distinct"
LAW_CHAIN_REQUIRED_FOR_NOTE = "chain_required_for_note"
LAW_PEAK_IS_NOT_NOTE = "peak_is_not_note"
LAW_CURVE_IS_NOT_NOTE = "curve_is_not_note"
LAW_TOKEN_IS_NOT_NOTE = "token_is_not_note"
LAW_ROOT_MUST_BE_INFERRED_FROM_CHAIN = "root_must_be_inferred_from_chain"
LAW_EARLY_ROOT_GUESS_ALLOWED = "early_root_guess_allowed"
LAW_FINAL_ROOT_WITHOUT_CHAIN_FORBIDDEN = "final_root_without_chain_forbidden"
LAW_THEORETICAL_CHAIN_EXISTS = "theoretical_chain_exists"
LAW_OBSERVED_CHAIN_EXISTS = "observed_chain_exists"
LAW_CONFIRMED_CHAIN_REQUIRED_FOR_NOTE = "confirmed_chain_required_for_note"
LAW_MULTIPLE_CHAINS_ALLOWED = "multiple_chains_allowed"
LAW_POLYPHONY_IS_CHAIN_COEXISTENCE = "polyphony_is_chain_coexistence"
LAW_ZERO_FORBIDDEN_IN_TOKEN = "zero_forbidden_in_token"
LAW_ZERO_LEAK_FORBIDDEN = "zero_leak_forbidden"

LAW_MANDATORY_INFERENCE_ORDER = "mandatory_inference_order"


# ----------------------------------------------------------------------
# Small structured view
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class ProjectLawSnapshot:
    coordinate_system_primary: bool
    microstructure_recursive_base12: bool
    note_is_trajectory_based: bool
    note_and_instrument_distinct: bool

    chain_required_for_note: bool
    peak_is_not_note: bool
    curve_is_not_note: bool
    token_is_not_note: bool

    root_must_be_inferred_from_chain: bool
    early_root_guess_allowed: bool
    final_root_without_chain_forbidden: bool

    theoretical_chain_exists: bool
    observed_chain_exists: bool
    confirmed_chain_required_for_note: bool

    multiple_chains_allowed: bool
    polyphony_is_chain_coexistence: bool

    zero_forbidden_in_token: bool
    zero_leak_forbidden: bool

    mandatory_inference_order: tuple[str, ...]


# ----------------------------------------------------------------------
# Registry builder
# ----------------------------------------------------------------------

def _build_registry() -> Dict[str, Any]:
    """
    Build the active law registry.

    IMPORTANT:
    We intentionally keep this compact and conservative.
    Values here represent the current enforceable interpretation
    of the already established philosophy.
    """
    registry: Dict[str, Any] = {
        # already established philosophy
        LAW_COORDINATE_SYSTEM_PRIMARY: True,
        LAW_MICROSTRUCTURE_RECURSIVE_BASE12: True,
        LAW_NOTE_IS_TRAJECTORY_BASED: True,
        LAW_NOTE_AND_INSTRUMENT_DISTINCT: True,

        # missing bridge law, now centralized
        LAW_CHAIN_REQUIRED_FOR_NOTE: bool(_chain.CHAIN_REQUIRED_FOR_NOTE),
        LAW_PEAK_IS_NOT_NOTE: bool(_chain.PEAK_IS_NOT_NOTE),
        LAW_CURVE_IS_NOT_NOTE: bool(_chain.CURVE_IS_NOT_NOTE),
        LAW_TOKEN_IS_NOT_NOTE: bool(_chain.TOKEN_IS_NOT_NOTE),

        LAW_ROOT_MUST_BE_INFERRED_FROM_CHAIN: bool(_chain.ROOT_MUST_BE_INFERRED_FROM_CHAIN),
        LAW_EARLY_ROOT_GUESS_ALLOWED: bool(_chain.EARLY_ROOT_GUESS_ALLOWED),
        LAW_FINAL_ROOT_WITHOUT_CHAIN_FORBIDDEN: bool(_chain.FINAL_ROOT_WITHOUT_CHAIN_FORBIDDEN),

        LAW_THEORETICAL_CHAIN_EXISTS: bool(_chain.THEORETICAL_CHAIN_EXISTS),
        LAW_OBSERVED_CHAIN_EXISTS: bool(_chain.OBSERVED_CHAIN_EXISTS),
        LAW_CONFIRMED_CHAIN_REQUIRED_FOR_NOTE: bool(_chain.CONFIRMED_CHAIN_REQUIRED_FOR_NOTE),

        LAW_MULTIPLE_CHAINS_ALLOWED: bool(_chain.MULTIPLE_CHAINS_ALLOWED),
        LAW_POLYPHONY_IS_CHAIN_COEXISTENCE: bool(_chain.POLYPHONY_IS_CHAIN_COEXISTENCE),

        # token law vs internal coordinate law
        LAW_ZERO_FORBIDDEN_IN_TOKEN: True,
        LAW_ZERO_LEAK_FORBIDDEN: True,

        LAW_MANDATORY_INFERENCE_ORDER: tuple(_chain.MANDATORY_INFERENCE_ORDER),
    }
    return registry


_PROJECT_LAW_REGISTRY: Dict[str, Any] = _build_registry()


# ----------------------------------------------------------------------
# Public access
# ----------------------------------------------------------------------

def active_laws() -> Dict[str, Any]:
    """
    Return a shallow copy of the active law registry.
    """
    return dict(_PROJECT_LAW_REGISTRY)


def law(name: str, default: Any = None) -> Any:
    """
    Safe law lookup.
    """
    return _PROJECT_LAW_REGISTRY.get(str(name), default)


def has_law(name: str) -> bool:
    """
    Check whether a law key exists in the registry.
    """
    return str(name) in _PROJECT_LAW_REGISTRY


def require(name: str) -> None:
    """
    Require that a boolean law exists and is active.

    Raises:
        KeyError   - if the law is unknown
        RuntimeError - if the law exists but is not active
    """
    key = str(name)
    if key not in _PROJECT_LAW_REGISTRY:
        raise KeyError(f"Unknown project law: {key}")

    value = _PROJECT_LAW_REGISTRY[key]
    if value is not True:
        raise RuntimeError(f"Project law is not active: {key}={value!r}")


def mandatory_inference_order() -> Tuple[str, ...]:
    """
    Return the required order of inference stages.

    Expected:
        observation -> curve -> chain -> note
    """
    value = _PROJECT_LAW_REGISTRY[LAW_MANDATORY_INFERENCE_ORDER]
    return tuple(value)


def is_inference_order_valid(stages: Iterable[str]) -> bool:
    """
    Check whether a provided stage sequence matches the mandatory order exactly.
    """
    seq = tuple(str(x).strip().lower() for x in stages)
    required = tuple(str(x).strip().lower() for x in mandatory_inference_order())
    return seq == required


def snapshot() -> ProjectLawSnapshot:
    """
    Return a typed snapshot of the current active laws.
    """
    r = _PROJECT_LAW_REGISTRY
    return ProjectLawSnapshot(
        coordinate_system_primary=bool(r[LAW_COORDINATE_SYSTEM_PRIMARY]),
        microstructure_recursive_base12=bool(r[LAW_MICROSTRUCTURE_RECURSIVE_BASE12]),
        note_is_trajectory_based=bool(r[LAW_NOTE_IS_TRAJECTORY_BASED]),
        note_and_instrument_distinct=bool(r[LAW_NOTE_AND_INSTRUMENT_DISTINCT]),

        chain_required_for_note=bool(r[LAW_CHAIN_REQUIRED_FOR_NOTE]),
        peak_is_not_note=bool(r[LAW_PEAK_IS_NOT_NOTE]),
        curve_is_not_note=bool(r[LAW_CURVE_IS_NOT_NOTE]),
        token_is_not_note=bool(r[LAW_TOKEN_IS_NOT_NOTE]),

        root_must_be_inferred_from_chain=bool(r[LAW_ROOT_MUST_BE_INFERRED_FROM_CHAIN]),
        early_root_guess_allowed=bool(r[LAW_EARLY_ROOT_GUESS_ALLOWED]),
        final_root_without_chain_forbidden=bool(r[LAW_FINAL_ROOT_WITHOUT_CHAIN_FORBIDDEN]),

        theoretical_chain_exists=bool(r[LAW_THEORETICAL_CHAIN_EXISTS]),
        observed_chain_exists=bool(r[LAW_OBSERVED_CHAIN_EXISTS]),
        confirmed_chain_required_for_note=bool(r[LAW_CONFIRMED_CHAIN_REQUIRED_FOR_NOTE]),

        multiple_chains_allowed=bool(r[LAW_MULTIPLE_CHAINS_ALLOWED]),
        polyphony_is_chain_coexistence=bool(r[LAW_POLYPHONY_IS_CHAIN_COEXISTENCE]),

        zero_forbidden_in_token=bool(r[LAW_ZERO_FORBIDDEN_IN_TOKEN]),
        zero_leak_forbidden=bool(r[LAW_ZERO_LEAK_FORBIDDEN]),

        mandatory_inference_order=tuple(r[LAW_MANDATORY_INFERENCE_ORDER]),
    )


# ----------------------------------------------------------------------
# Focused helpers for blocks / demons
# ----------------------------------------------------------------------

def note_requires_chain() -> bool:
    return bool(_PROJECT_LAW_REGISTRY[LAW_CHAIN_REQUIRED_FOR_NOTE])


def peak_is_note() -> bool:
    """
    Convenience negative helper.
    Returns False in the current project law.
    """
    return not bool(_PROJECT_LAW_REGISTRY[LAW_PEAK_IS_NOT_NOTE])


def curve_is_note() -> bool:
    """
    Convenience negative helper.
    Returns False in the current project law.
    """
    return not bool(_PROJECT_LAW_REGISTRY[LAW_CURVE_IS_NOT_NOTE])


def token_is_note() -> bool:
    """
    Convenience negative helper.
    Returns False in the current project law.
    """
    return not bool(_PROJECT_LAW_REGISTRY[LAW_TOKEN_IS_NOT_NOTE])


def final_root_without_chain_forbidden() -> bool:
    return bool(_PROJECT_LAW_REGISTRY[LAW_FINAL_ROOT_WITHOUT_CHAIN_FORBIDDEN])


def zero_forbidden_in_token() -> bool:
    return bool(_PROJECT_LAW_REGISTRY[LAW_ZERO_FORBIDDEN_IN_TOKEN])


def zero_leak_forbidden() -> bool:
    return bool(_PROJECT_LAW_REGISTRY[LAW_ZERO_LEAK_FORBIDDEN])


def polyphony_is_chain_coexistence() -> bool:
    return bool(_PROJECT_LAW_REGISTRY[LAW_POLYPHONY_IS_CHAIN_COEXISTENCE])


# ----------------------------------------------------------------------
# Reporting helper
# ----------------------------------------------------------------------

def project_law_report_lines() -> list[str]:
    """
    Human-readable compact report for demons / logs / TXT summaries.
    """
    r = active_laws()
    lines = [
        "music12 project law guard",
        "------------------------",
        f"{LAW_COORDINATE_SYSTEM_PRIMARY} = {r[LAW_COORDINATE_SYSTEM_PRIMARY]}",
        f"{LAW_MICROSTRUCTURE_RECURSIVE_BASE12} = {r[LAW_MICROSTRUCTURE_RECURSIVE_BASE12]}",
        f"{LAW_NOTE_IS_TRAJECTORY_BASED} = {r[LAW_NOTE_IS_TRAJECTORY_BASED]}",
        f"{LAW_NOTE_AND_INSTRUMENT_DISTINCT} = {r[LAW_NOTE_AND_INSTRUMENT_DISTINCT]}",
        f"{LAW_CHAIN_REQUIRED_FOR_NOTE} = {r[LAW_CHAIN_REQUIRED_FOR_NOTE]}",
        f"{LAW_PEAK_IS_NOT_NOTE} = {r[LAW_PEAK_IS_NOT_NOTE]}",
        f"{LAW_CURVE_IS_NOT_NOTE} = {r[LAW_CURVE_IS_NOT_NOTE]}",
        f"{LAW_TOKEN_IS_NOT_NOTE} = {r[LAW_TOKEN_IS_NOT_NOTE]}",
        f"{LAW_ROOT_MUST_BE_INFERRED_FROM_CHAIN} = {r[LAW_ROOT_MUST_BE_INFERRED_FROM_CHAIN]}",
        f"{LAW_EARLY_ROOT_GUESS_ALLOWED} = {r[LAW_EARLY_ROOT_GUESS_ALLOWED]}",
        f"{LAW_FINAL_ROOT_WITHOUT_CHAIN_FORBIDDEN} = {r[LAW_FINAL_ROOT_WITHOUT_CHAIN_FORBIDDEN]}",
        f"{LAW_THEORETICAL_CHAIN_EXISTS} = {r[LAW_THEORETICAL_CHAIN_EXISTS]}",
        f"{LAW_OBSERVED_CHAIN_EXISTS} = {r[LAW_OBSERVED_CHAIN_EXISTS]}",
        f"{LAW_CONFIRMED_CHAIN_REQUIRED_FOR_NOTE} = {r[LAW_CONFIRMED_CHAIN_REQUIRED_FOR_NOTE]}",
        f"{LAW_MULTIPLE_CHAINS_ALLOWED} = {r[LAW_MULTIPLE_CHAINS_ALLOWED]}",
        f"{LAW_POLYPHONY_IS_CHAIN_COEXISTENCE} = {r[LAW_POLYPHONY_IS_CHAIN_COEXISTENCE]}",
        f"{LAW_ZERO_FORBIDDEN_IN_TOKEN} = {r[LAW_ZERO_FORBIDDEN_IN_TOKEN]}",
        f"{LAW_ZERO_LEAK_FORBIDDEN} = {r[LAW_ZERO_LEAK_FORBIDDEN]}",
        f"{LAW_MANDATORY_INFERENCE_ORDER} = {r[LAW_MANDATORY_INFERENCE_ORDER]}",
    ]
    return lines


def project_law_report_text() -> str:
    return "\n".join(project_law_report_lines())