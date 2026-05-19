# Duodecimal Notation and Time-60 Principle

## Overview

The project uses a custom duodecimal coordinate notation designed for resonance analysis.

Unlike conventional decimal-centered representations, the system is built around:

- 12-radix pitch structure
- hierarchical microtonal subdivision
- resonance-consistent temporal division

The notation is intended not only for musical labeling, but for describing resonance coordinates in a structured harmonic space.

---

# Why Base-12

## Harmonic Structure

Many resonance systems naturally organize around relationships strongly connected to division by 12:

- octave subdivision
- harmonic ratios
- cyclic pitch relationships
- rotational symmetry

The system therefore uses a 12-radix representation rather than decimal indexing.

Digits:

1 2 3 4 5 6 7 8 9 A B C

Zero is intentionally excluded from note-space notation.

# Basic Note Format

Example:

9.A'-

Meaning:

9 → octave region
A → pitch class
' → boundary between the base coordinate and microtonal layer
- → neutral / stable microtonal state

In the current implementation:

9.A'-
≈ A4 ≈ 440 Hz

after anchor calibration.

# Unified Coordinate Structure

General form:

<octave>.<degree>'<microtonal_suffix>

Examples:

5.A'-
7.7'-
8.C'a53
9.A'i6C

The apostrophe is mandatory.

Everything after ' belongs to the microtonal refinement layer.

# Microtonal Extension

The notation supports hierarchical microtonal subdivision.

Examples:

9.A'i6C
8.C'a53

# Meaning of the Apostrophe

The symbol:

'

separates:

the base harmonic coordinate
the microtonal refinement structure

This makes the notation structurally explicit and prevents ambiguity between:

octave / degree identity
microtonal deviation
subdivision hierarchy

#Neutral State

The symbol:

-

means:

neutral microtonal position

Example:

9.A'-

means:

no upward deviation
no downward deviation
stable anchor position

# Microtonal Directions

Two directional symbols are used:

i
i = upward deviation
a
a = downward deviation
# 
Hierarchical Subdivision

The system uses recursive subdivision based on powers of 12.

Examples:

first layer → 12
second layer → 144
third layer → 1728

This allows increasingly fine resonance positioning while preserving harmonic consistency.

# Example Interpretation 

## Example 1

9.A'i6C

can be interpreted as:

base coordinate: 9.A
upward microtonal deviation
hierarchical subdivision path: 6C

## Example 2

8.C'a53

can be interpreted as:

base coordinate: 8.C
downward microtonal deviation
hierarchical subdivision path: 53

# Why Zero is Excluded

The notation intentionally avoids the symbol 0.

Reason:

zero introduces ambiguity in cyclic harmonic space
harmonic identity is treated as continuous cyclic structure
the system prioritizes structural continuity over decimal indexing conventions

This also simplifies visual interpretation of rotational harmonic relationships.

# Relation to Spiral Geometry

The notation directly maps into spiral coordinates.

Example mapping:

pitch class → angular position
octave → radial level
microtonal refinement → local displacement

Thus:

notation = spatial coordinate

rather than a simple label.

# Continuous Spiral Principle

The duodecimal coordinate system is not intended as a set of isolated discrete points.

Each coordinate is treated as part of a continuous resonance trajectory inside spiral space.

This means that:

coordinate ≠ isolated vector

Instead:

coordinate = position on a continuous spiral curve

rather than as a sequence of isolated sample values.

# Relation to Analog Behavior

This approach is conceptually closer to analog wave behavior because:

resonance evolves continuously
harmonic interaction forms trajectories
phase relationships remain spatially connected
temporal evolution is preserved geometrically

The spiral representation therefore acts as:

a resonance-space model
a structural continuity framework
a geometric interpretation of signal evolution

rather than a purely discrete sampling grid.

# Important Clarification

The system still operates on digitally acquired signals.

However, the interpretation layer is fundamentally different from conventional PCM-centered analysis.

The goal is not merely:

frequency estimation
amplitude tracking

but reconstruction of:

resonance continuity
harmonic geometry
structural evolution in time

# General Principle

The model assumes that resonance systems are better represented as:

continuous evolving structures

rather than collections of isolated numerical samples.

---

# Why This Matters

Conventional PCM (Pulse-Code Modulation) representations treat signals as:

- discrete amplitude samples
- separated temporal measurements
- point-wise numerical states

This is highly effective for storage and transmission, but it does not explicitly preserve resonance continuity.

---

# Spiral-Continuous Interpretation

In the present model:

- harmonic evolution is continuous
- resonance structures are trajectory-based
- microtonal positioning is geometrically connected
- neighboring states remain structurally linked

The system therefore treats sound as:

continuous resonance motion

# Why Octaves Start Near ~1 Hz

The octave numbering system in this project intentionally differs from conventional musical octave numbering.

The system is not centered around historical musical notation standards.

Instead, octave indexing is tied to resonance-space scaling beginning near:

~1 Hz

---

# Reason

The project treats sound as part of a broader class of resonance phenomena.

This includes not only:

- musical sound
- instrumental acoustics

but also:

- low-frequency oscillations
- structural resonance
- geophysical waves
- mechanical vibration systems
- other resonance-based signals

For this reason, the octave structure is anchored to a physically scalable frequency space rather than to traditional piano notation.

---

# Consequence

Octave numbering becomes:

frequency-relative

rather than:

instrument-relative

This allows the same coordinate logic to be applied across very different frequency domains.

Example

In standard musical notation:

A4 = 440 Hz

In this system:

9.A'-
≈ 440 Hz

because the octave indexing begins near ~1 Hz and scales upward through powers of 2.

Why This Matters

Using a low-frequency physical anchor provides:

consistent logarithmic scaling
compatibility with non-musical resonance systems
unified treatment of wave structures
direct relation between octave growth and physical frequency doubling

This makes the notation more suitable for general resonance analysis rather than only musical labeling.

General Principle

The notation system is designed as:

a universal resonance coordinate framework

not merely as a musical naming convention.

# Time-60 Principle

## Why Time is Not Decimal-Centered

The project intentionally avoids strict decimal temporal logic.

Instead, temporal subdivision is based on division by 60.

## Reason

Many natural rhythmic and oscillatory systems divide more coherently through:

60

than through powers of 10.

60 supports clean subdivision into:

2
3
4
5
6
10
12

This makes it highly compatible with:

rhythm
wave interaction
harmonic timing
periodic resonance structures

## Practical Consequence

Instead of forcing time into decimal fractions:

0.1
0.01
0.001

the model prefers structures naturally compatible with:

musical meter
oscillatory division
harmonic periodicity

## Relation to Signal Analysis

The time-60 approach improves consistency between:

temporal segmentation
harmonic evolution
resonance persistence
cyclic structures

This is especially important in:

polyphonic analysis
overlapping resonance systems
temporal harmonic tracking

## General Principle

The notation system is not intended merely as an alternative musical naming convention.

It is a coordinate framework for describing:

resonance structures
harmonic geometry
temporal evolution
microtonal positioning

within a unified interpretable space.

## Relation to the Project

This notation is a core component of:

Duodecimal Resonant Numeration

A resonance-oriented framework for interpretable analysis of complex signals.