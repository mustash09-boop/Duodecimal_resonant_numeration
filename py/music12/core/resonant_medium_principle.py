"""
Formal principle of excitation source and resonant medium in music12.

This module contains no executable analytical logic.
Its purpose is to fix a core law of the project:

    note detection and instrument detection are related,
    but they are not the same analytical task.

----------------------------------------------------------------------
CORE DISTINCTION
----------------------------------------------------------------------

In music12, a sounding event must be understood through two connected
but non-identical aspects:

    1. excitation source
    2. resonant medium

The excitation source is the element that initiates the sound-producing
process.

The resonant medium is the structure in which the sound develops,
spreads, transforms, and becomes audible as a real spectral object.

Therefore:

    note detection        -> seeks the excitation source
    instrument detection  -> seeks the resonant medium

These two layers are different, but mutually related.

----------------------------------------------------------------------
EXCITATION SOURCE
----------------------------------------------------------------------

The excitation source is the initiating cause of the resonant event.

Examples:
    - hammer strike
    - pluck
    - bow friction
    - air impulse
    - reed activation
    - vocal fold excitation

The purpose of excitation-source analysis is to determine the primary
resonant trigger, that is, the element that defines the note-generating
origin.

In simplified analytical language:

    excitation source = what starts the note

This is the analytical layer closest to note detection.

----------------------------------------------------------------------
RESONANT MEDIUM
----------------------------------------------------------------------

The resonant medium is not reduced to a single string, pipe, or object.

It must be understood as the total resonant environment in which the
excitation develops.

This may include:
    - string or strings
    - body or корпус
    - air cavity
    - soundboard
    - material stiffness
    - coupling between parts
    - distributed modes of vibration

Thus the instrument is not merely a source of harmonics.

It is a resonant medium.

In simplified analytical language:

    resonant medium = where the note becomes physically real

This is the analytical layer closest to instrument detection.

----------------------------------------------------------------------
WHY THE TWO TASKS MUST BE SEPARATED
----------------------------------------------------------------------

A note can often be inferred from the excitation source even when the
resonant medium distorts or colors the resulting sound.

An instrument can often be inferred from the resonant medium even when
the excitation source remains the same note class.

Therefore note and instrument are not identical problems.

The system must not collapse them into one.

It is possible for:

    same excitation source -> different resonant media
    different excitation sources -> similar resonant traces
    one note -> multiple medium-dependent overtone structures

This is why the project must explicitly separate:
    excitation inference
    and
    resonant-medium inference

----------------------------------------------------------------------
DISSONANT OR NON-IDEAL HARMONICS
----------------------------------------------------------------------

Not every resonant medium reproduces the excitation source in a perfectly
harmonic way.

Real resonant media may introduce:
    - inharmonicity
    - stretched overtones
    - missing harmonics
    - additional sidebands
    - beating
    - phase conflict
    - dissonant or weakly aligned components

Therefore, the observed spectral structure is not always a direct mirror
of the excitation source.

Some harmonics belong more to the medium than to the source.

This is not analytical noise.
It is part of the physical identity of the instrument.

----------------------------------------------------------------------
CONSEQUENCE FOR NOTE DETECTION
----------------------------------------------------------------------

When determining the note, the analytical task is to infer the
excitation-generating origin.

This means the system should seek:
    - causal harmonic lineage
    - stable interval structure
    - source-consistent resonant trigger
    - the element that best explains the family of harmonics

The note is therefore not simply "the strongest visible peak".

The note is the most plausible excitation source.

----------------------------------------------------------------------
CONSEQUENCE FOR INSTRUMENT DETECTION
----------------------------------------------------------------------

When determining the instrument, the analytical task is to infer the
character of the resonant medium.

This means the system should seek:
    - overtone distribution
    - resonance density
    - attack/decay behavior
    - sustain behavior
    - phase-coherence behavior
    - agreement or disagreement of harmonics
    - medium-specific deformation of ideal source structure

The instrument is therefore not merely "which harmonics exist",
but rather:

    how the resonant medium transforms the excitation source.

----------------------------------------------------------------------
RELATION BETWEEN THE TWO
----------------------------------------------------------------------

Excitation source and resonant medium are never fully independent.

The source activates the medium.
The medium shapes the source into an audible object.

Therefore the audible sound is a coupled result:

    sound_event = excitation_source × resonant_medium

However, the two factors must still be analytically distinguished.

Otherwise the system cannot properly separate:
    note identity
from
    instrument identity

----------------------------------------------------------------------
PROJECT LAW
----------------------------------------------------------------------

In music12:

    note analysis seeks the excitation source
    instrument analysis seeks the resonant medium

These two processes must be connected,
but never collapsed into one naive task.

The model must preserve the distinction between:
    - what initiates the resonant event
    - what physically shapes the resonant event

----------------------------------------------------------------------
PRACTICAL CONSEQUENCE
----------------------------------------------------------------------

Future modules may evolve toward two connected branches:

    A. source inference branch
        - note
        - root
        - interval lineage
        - causal trigger

    B. medium inference branch
        - instrument
        - overtone field
        - coherence pattern
        - resonance density
        - decay/sustain morphology

The strongest analysis will combine both branches,
but should not confuse them.

----------------------------------------------------------------------
SHORT FORMULA
----------------------------------------------------------------------

The note is not the instrument.
The instrument is not the note.

The note is the excitation source.
The instrument is the resonant medium.

And the audible sound is their coupled manifestation.
"""