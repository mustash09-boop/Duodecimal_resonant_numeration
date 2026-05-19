"""
Formal principle of the internal coordinate system in music12.

This module contains no executable analytical logic.
Its purpose is to fix one of the main laws of the project:

    the primary language of analysis must be the project's own
    internal coordinate system, not external engineering units.

External units such as:
    - Hertz
    - cents
    - decimal seconds

are useful, but they are projections of the model, not the model itself.

----------------------------------------------------------------------
CORE LAW
----------------------------------------------------------------------

The music12 project must treat sound analysis as motion and structure
inside an internal resonant coordinate system.

This internal system is primary.

External systems of measurement are secondary.

That means:

    internal coordinates  -> primary analytical reality
    external units        -> interface / projection / translation layer

The model must not be built "from Hertz upward".

It must be built from the internal structure of the resonant space,
and only then translated outward when compatibility is needed.

----------------------------------------------------------------------
WHY THIS IS NECESSARY
----------------------------------------------------------------------

Ordinary signal processing often begins with external quantities:

    - frequency in Hertz
    - interval deviation in cents
    - time in decimal seconds

These are convenient engineering descriptions.

But they are not structurally neutral.

They carry assumptions inherited from:
    - decimal notation
    - cents-based interval logic
    - linearized engineering approximation

Such assumptions are useful for instrumentation,
but they are not sufficient as the native language of the project.

The music12 system aims to preserve:
    - 12-radix interval logic
    - self-similar microstructure
    - coherence between resonance and time
    - future phase-aware and curve-aware analysis

Therefore the base analytical layer must not depend
on external decimal conventions as its first ontology.

----------------------------------------------------------------------
INTERNAL COORDINATE LAYERS
----------------------------------------------------------------------

The primary internal description may include, at minimum:

    1. octave container
    2. step container
    3. first micro-layer (inch / k144)
    4. second micro-layer (sub-inch / k1728)
    5. position in time according to project-consistent subdivision
    6. phase or phase-like coordinate
    7. resonance lineage / harmonic family identity
    8. curve-state of motion in time-resonance space

Not all of these layers must always be fully implemented at once.

But they define the proper direction of the system.

----------------------------------------------------------------------
EXTERNAL UNITS AS PROJECTIONS
----------------------------------------------------------------------

The following units remain useful:

    Hertz
    cents
    decimal seconds

However, they must be treated as projections or interface views.

That means:

    Hz       -> physical projection
    cents    -> external musical metric
    seconds  -> external temporal metric

They are not forbidden.

But they must not become the native geometry of the analysis.

The native geometry belongs to the internal coordinate system.

----------------------------------------------------------------------
ANCHOR SHIFT DOES NOT DESTROY THE SYSTEM
----------------------------------------------------------------------

If the project chooses a different anchor convention, the system
does not collapse, provided the internal coordinate relations remain stable.

For example:

    if a chosen internal token is mapped to a physical reference value,
    and that reference is shifted,
    the system remains valid
    as long as the structural laws of transformation remain unchanged.

Thus the essence of the system is not absolute naming,
but invariant resonant relations.

This is why anchor displacement is calibration,
not destruction of the model.

----------------------------------------------------------------------
TIME AND RESONANCE
----------------------------------------------------------------------

The project treats time and resonance as mutually related coordinate axes.

Resonance is not merely raw frequency.
Time is not merely decimal seconds.

The analytical aim is not to accumulate arbitrary measurements,
but to locate sound events inside a coherent multidimensional structure.

This means that:
    - frequency-like quantities
    - time-like quantities
    - phase-like quantities
    - harmonic-lineage quantities

must ultimately be expressible in one internally consistent framework.

----------------------------------------------------------------------
DIFFERENCE BETWEEN MODEL AND REPRESENTATION
----------------------------------------------------------------------

A key distinction must always be preserved:

    model            != representation
    internal law     != exported number
    coordinate       != projection

For example:

    an internal pitch coordinate is not the same thing as its Hertz export
    an internal micro-position is not the same thing as cents
    an internal time subdivision is not the same thing as decimal display

Confusing these levels causes hidden analytical errors.

Therefore all serious modules should keep the distinction explicit.

----------------------------------------------------------------------
PROJECT LAW
----------------------------------------------------------------------

In music12:

    internal resonant coordinates are primary,
    external engineering units are secondary.

Any analytical module should be designed so that:
    1. internal meaning is preserved first
    2. external compatibility is added second

The system must never reduce its own coordinate logic
to external decimal conventions.

----------------------------------------------------------------------
PRACTICAL CONSEQUENCE
----------------------------------------------------------------------

Future modules should aim for the following order:

    internal detection
    -> internal coordinate interpretation
    -> structural validation
    -> external projection (Hz / cents / seconds)

and not the reverse.

That is:

    do not define the sound in external units first
    and only afterwards try to recover structure.

Instead:

    define the structure first,
    then export compatible external measurements.

----------------------------------------------------------------------
SHORT FORMULA
----------------------------------------------------------------------

music12 does not begin from external measurements.

It begins from internal resonant coordinates.

And therefore:

    Hz, cents, seconds = projections
    internal coordinate system = analytical source reality
"""