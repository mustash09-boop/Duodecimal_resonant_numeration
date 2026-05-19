# GitHub Public Release Guide

## Public objective

The public repository should show:

- what the method is
- why it differs from conventional DSP-only thinking
- how to run a minimal example
- what kinds of outputs the system produces

It should not attempt to expose every private experiment at once.

## Recommended first repository shape

```text
Duodecimal_resonant_numeration/
├── README.md
├── docs/
├── py/
├── tools/
├── examples/
├── images/
├── manifests/
├── papers/
└── links/
```

## First public story

Keep the first release narrow and strong.

Suggested public storyline:

1. resonance entities and event lifecycle
2. note vs body vs field distinction
3. one compact instrument-comparison case
4. one compact polyphonic case

## Practical rule

If a file is large, noisy, or only useful for local iteration, describe it in a
manifest instead of uploading it.

