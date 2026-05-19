"""
Formal principle of resonance curves in the music12 system.

This module contains no executable analytical logic.
Its purpose is to fix a core law of the project:

    musical pitch movement is not modeled as a straight line,
    but as a curve in the joint space of time and resonance.

----------------------------------------------------------------------
CORE IDEA
----------------------------------------------------------------------

In ordinary signal processing, pitch is often treated as:
    - a peak,
    - a point on a frequency axis,
    - or a line in time-frequency space.

In this project, that is insufficient.

The music12 system treats music as motion in a coordinate space
with at least two principal axes:

    1. time
    2. resonance

Here resonance is not merely raw Hertz value.
It is a structured logarithmic coordinate space, expressed through:

    octave
    step
    inch-scale refinement
    sub-inch refinement

Therefore, a note is not just a frequency value.
A note is a trajectory in resonant space.

----------------------------------------------------------------------
WHY THE TRAJECTORY IS NOT A STRAIGHT LINE
----------------------------------------------------------------------

The internal pitch structure of the project is not linear.

It is generated through multiplicative refinement:

    semitone step:
        2 ** (1 / 12)

    first micro-step:
        2 ** (1 / 144)

    second micro-step:
        2 ** (1 / 1728)

This means that successive pitch refinements are not additive in the
ordinary linear sense.

They are multiplicative in frequency space and logarithmic in structure.

As a result, when pitch develops through time, the corresponding
trajectory is not naturally represented by a straight line.

It is represented by a smooth curve.

----------------------------------------------------------------------
DIFFERENCE FROM ORDINARY TIME-FREQUENCY MODELS
----------------------------------------------------------------------

A standard time-frequency plot often assumes that the relevant object is:

    frequency(t)

and that the meaningful geometry is linear or piecewise linear.

In music12, this is too weak.

The system assumes that pitch lives inside a nested 12-radix resonant grid.

Therefore the meaningful object is not merely:

    frequency over time

but:

    curved motion through nested resonant coordinates

This distinguishes the project from straight-line assumptions common
in ordinary time-frequency or phase-space interpretation.

----------------------------------------------------------------------
WHAT THIS DESCRIBES IN REAL SOUND
----------------------------------------------------------------------

A real musical tone is not a static point.

It contains:
    - attack
    - decay
    - beating
    - vibrato
    - inharmonicity
    - slow drift of resonant emphasis

These are not well described as isolated peaks.

In the music12 view, they are naturally interpreted as motion along
a curved resonant path.

Thus:

    note     -> not a point
    note     -> not a single peak
    note     -> not just a line
    note     -> a resonant curve

----------------------------------------------------------------------
HARMONICS
----------------------------------------------------------------------

Harmonics are also not treated as a random pile of peaks.

They are understood as a family of mutually related resonant curves,
originating from one root source.

Therefore, note detection must not be reduced to:

    strongest peak selection

nor even merely to:

    best straight spectral fit

Instead it should seek:

    the source of a coherent family of resonant curves

This is a stronger and more physical criterion.

----------------------------------------------------------------------
CONSEQUENCE FOR ANALYSIS
----------------------------------------------------------------------

Any serious pitch-analysis module in this project should prefer:

    coherent curved resonant motion

over:

    accidental linear alignment
    isolated local maxima
    nearest-bin approximation

This means the system should evolve toward:
    - curve-aware candidate tracking
    - resonance continuity
    - harmonic family coherence across time
    - rejection of false roots that only explain sparse local peaks

----------------------------------------------------------------------
PROJECT LAW
----------------------------------------------------------------------

In the music12 system:

    time and resonance form a joint coordinate space.

Within that space, notes and harmonics are not fundamentally linear objects.

They are curved trajectories shaped by multiplicative resonance structure.

Therefore, pitch recognition should be formulated not as
"peak picking" or "line fitting",
but as detection of coherent resonant curves.

----------------------------------------------------------------------
SHORT FORMULA
----------------------------------------------------------------------

The system does not search for straight frequency traces.

It searches for curved resonant motion.

And therefore:

    note = curve in time-resonance space
    harmonic family = coherent bundle of such curves
"""