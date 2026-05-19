"""
Formal principle of the 12-radix microstructure system.

This module does not implement pitch conversion logic directly.
Its purpose is to fix the conceptual law of the project, so that
all future code remains aligned with the same mathematical meaning.

----------------------------------------------------------------------
CORE PRINCIPLE
----------------------------------------------------------------------

The notation system is not merely a naming scheme for musical pitches.
It is a coordinate system for resonant space.

At the coarse level, the system uses:
    - octave container
    - step container inside octave

At the micro level, the system preserves the same 12-radix structure
inside the semitone itself.

Therefore, the microstructure is not defined by arbitrary decimal
subdivision, nor by direct cent slicing, but by self-similar
12-radix refinement of the semitone interval.

----------------------------------------------------------------------
WHY 144 AND 1728
----------------------------------------------------------------------

The project adopts the scale cascade:

    12 -> 144 -> 1728

where:

    12     = semitone structure inside the octave
    144    = 12 * 12, first micro-layer inside one semitone
    1728   = 12 * 12 * 12, second micro-layer inside one semitone

This choice is not accidental and not merely "for precision".

It preserves self-similarity of the 12-radix system across scales.

Thus:

    octave/step level     -> 12-radix
    first micro level     -> 12-radix refinement
    second micro level    -> 12-radix refinement of refinement

The same structural law governs all levels.

----------------------------------------------------------------------
MEANING OF "INCH"
----------------------------------------------------------------------

In this project, "inch" is NOT:
    - a Hertz value
    - a decimal unit
    - a direct subdivision of cents

Instead, "inch" is the positional micro-coordinate inside the semitone,
defined by repeated multiplication by roots of 2:

    one first-level microstep:
        2 ** (1 / 144)

    one second-level microstep:
        2 ** (1 / 1728)

Therefore, inch notation looks linear in writing
(i1, i2, i3, ... or a1, a2, a3, ...),
but physically it represents logarithmic movement in frequency space.

That is why inch notation is a coordinate language,
not merely an error magnitude.

----------------------------------------------------------------------
DIFFERENCE FROM CENTS
----------------------------------------------------------------------

Cents answer the question:

    "How far apart are two frequencies?"

They are a universal logarithmic metric of interval distance.

Inches answer a different question:

    "Where inside the structured semitone grid is this frequency located?"

Thus:

    cents  -> metric of distance
    inch   -> coordinate inside resonant microstructure

Cents are useful for external musical comparison.

Inches are necessary for internal structural description.

----------------------------------------------------------------------
FORMAL INTERPRETATION
----------------------------------------------------------------------

Let the base pitch token define a coarse resonant location.

Then the full pitch position may be refined by micro-layers:

    octave
    step
    first micro-layer (1 / 144 of semitone in exponent space)
    second micro-layer (1 / 1728 of semitone in exponent space)

This means that a note token in the system is not merely a label
for an equal-tempered pitch.

It is an address of a point in logarithmic resonant space.

----------------------------------------------------------------------
CONSEQUENCE FOR ANALYSIS
----------------------------------------------------------------------

A candidate pitch must be evaluated not only by:
    - harmonic explanation
    - cents error
    - physical frequency error

but also by its coherence inside the project's own microstructure.

That means future analytical modules may use:
    - inch position
    - k144
    - k1728
    - micro-direction (i / a)
as native diagnostic dimensions.

This allows the system to distinguish:
    - simple interval deviation
from
    - structural misplacement in resonant space.

----------------------------------------------------------------------
PROJECT LAW
----------------------------------------------------------------------

The 12-radix notation of the project is self-similar across scales.

The system must never collapse microstructure into cents-only logic.

Any algorithm that works with pitch must preserve the distinction
between:

    1. external metric comparison (cents, Hz)
    2. internal resonant coordinate description (inch, k144, k1728)

Only this preserves the full meaning of the notation.

----------------------------------------------------------------------
SHORT FORMULA
----------------------------------------------------------------------

The system does not merely measure pitch.

It locates resonance.

And therefore:

    note = coordinate in resonant logarithmic space
    not just interval label
"""