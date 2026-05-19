# Duodecimal Resonant Numeration

## A resonance-first research framework for interpretable sound structure

This project studies sound not as a flat stream of amplitudes, but as a
structured field of resonant entities evolving through time.

The current research focus is musical audio because it gives a controllable and
verifiable environment for testing the method. The longer-term goal is broader:
an interpretable framework for resonance structure in complex signals.

## Core research ideas

- duodecimal harmonic coordinates
- spiral-time geometry
- resonance entities and event lifecycles
- note vs body vs field distinction
- chain-based note identity instead of single-peak detection
- instrument behavior described through event roles, not only static timbre

## Why this differs from conventional DSP-only pipelines

A conventional pipeline often centers on:

- spectral peaks
- energy statistics
- frame-local classification

This project asks a different set of questions:

- what was born as a new excitation
- what is a continuation of the same entity
- what belongs to the instrument body
- what belongs to field or hall response
- which instrument owns attack, sustain, support, or residual presence

## Key concepts

### Duodecimal notation

The project uses a custom 12-radix pitch and microstructure notation such as:

- `9.A'-`
- `8.C'a53`
- `9.A'i6C`

This notation is tied to the project’s harmonic and spiral geometry rather than
to historical note naming alone.

### Spiral-time representation

Signals can be mapped into a 3D spiral model where:

- angle represents pitch class
- radius represents octave or harmonic level
- height represents temporal evolution

This makes resonance motion and continuity visible.

### Note box principle

The framework separates:

- the note excitation
- the resonant response of the instrument body

This is one of the foundations for later instrument differentiation.

## Repository philosophy

This repository should function as an emerging research lab, not as a raw
storage dump.

The public-facing structure should prioritize:

- clarity
- reproducibility
- compact examples
- interpretable outputs

Large archives, caches, full scans, and heavy generated reports should stay out
of GitHub and be described through manifests instead.

## Main directories

- [docs](E:/Duodecimal_resonant_numeration/docs) — architecture, public release notes, research memory
- [py](E:/Duodecimal_resonant_numeration/py) — core code
- [tools](E:/Duodecimal_resonant_numeration/tools) — external runners, audits, and experimental layers
- [examples](E:/Duodecimal_resonant_numeration/examples) — minimal reproducible public examples
- [images](E:/Duodecimal_resonant_numeration/images) — publication-oriented visuals
- [manifests](E:/Duodecimal_resonant_numeration/manifests) — descriptions of large offline datasets
- [papers](E:/Duodecimal_resonant_numeration/papers) — whitepapers and publication material
- [links](E:/Duodecimal_resonant_numeration/links) — cloud/archive/public links

## Good first public demonstrations

The strongest initial public release is likely not the whole project at once,
but a few compact cases:

1. same note played by different instruments
2. short polyphonic fragment with event-count reasoning
3. one ensemble case with layered instrument-role assignment

## Next public-release documents

- [docs/GITHUB_PUBLIC_RELEASE_GUIDE.md](E:/Duodecimal_resonant_numeration/docs/GITHUB_PUBLIC_RELEASE_GUIDE.md)
- [manifests/PUBLIC_RELEASE_SCOPE_v1.md](E:/Duodecimal_resonant_numeration/manifests/PUBLIC_RELEASE_SCOPE_v1.md)
- [links/CLOUD_AND_ARCHIVE_LINKS.md](E:/Duodecimal_resonant_numeration/links/CLOUD_AND_ARCHIVE_LINKS.md)

## Existing public records

- [Zenodo record 10.5281/zenodo.20076382](https://doi.org/10.5281/zenodo.20076382)
- [Zenodo record 10.5281/zenodo.18431048](https://doi.org/10.5281/zenodo.18431048)

## Intended license

Creative Commons Attribution-NonCommercial 4.0 International (`CC BY-NC 4.0`)
