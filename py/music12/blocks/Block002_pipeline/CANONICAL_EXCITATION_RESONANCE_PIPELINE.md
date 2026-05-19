# Canonical Excitation-Resonance Pipeline

This document fixes the main task of `Block002` in the most restrictive form:

`find exciters in a continuous audio flow, trace the primary resonance that belongs to each exciter, then separate the secondary resonance layer that the resonant body inherits from the note`

The pipeline must not optimize for decorative structure. It must optimize for causal readability.

## Primary question

For each local musical event in streaming audio:

1. What acted as the exciter.
2. Over which time span that exciter was actually active.
3. Which resonance belongs to that exciter as the primary note-bearing chain.
4. When the note starts transferring energy into the resonant body.
5. Which later structures are already secondary resonance and must not be mistaken for the note itself.

## Input contract

The canonical starting point is `_audio_probe`, not the final framewise note table.

Minimal input layer:
- dense probe rows or probe matrix derived from `_audio_probe`
- frame time metadata
- note-token mapping in the project's native duodecimal system

Optional weak prior:
- isolated-note evidence from `Block004_data/piano_midi1`

Ground truth for evaluation:
- score/MIDI event references when available

## Canonical stages

### 0. Probe Field

Input:
- `_audio_probe`

Output:
- dense framewise probe field

Task:
- preserve raw resonant evidence without deciding what the note is yet.

Forbidden shortcut:
- declaring a note from a single strong frame or strongest peak.

### 1. Excitation Seed Extraction

Input:
- probe field

Output:
- `excitation_seed` objects

Task:
- identify compact local births of energy that plausibly begin a note event;
- allow the exciter to be distributed across a short time interval, not a single instant.

Must capture:
- first rise
- local attack spread
- earliest coherent harmonic support

### 2. Proto-Exciter Consolidation

Input:
- excitation seeds

Output:
- `proto_exciter` objects

Task:
- merge seeds that belong to the same birth process;
- reject seeds that are only body shimmer or late reflections.

Key question:
- is this energy source initiating resonance, or only reacting to prior resonance.

### 3. Early Branch Classification

Input:
- proto-exciters
- early family field evidence

Output:
- `pitched`
- `event`
- `unresolved`

Task:
- decide whether the exciter is already showing a stable root-bearing continuation or whether it behaves more like an event/resonance field without mandatory note identity.

Law:
- branch by causal continuation behavior, not by instrument label.

Safe routing rule:
- `pitched + unresolved` may continue into the note-chain branch
- `event` must be preserved for a separate gesture / resonance-field branch

Research law:
- exact MIDI meta such as event count, onset-group count, and max polyphony must be tracked as a separate target-alignment layer;
- local causal improvements are not enough if the emitted entity population drifts far away from the known event structure.

### 4. Primary Resonance Family Formation

Input:
- proto-exciters
- probe field

Output:
- `primary_family` candidates

Task:
- build the early resonance family that is causally tied to the exciter;
- let the family accumulate over time instead of requiring all harmonics at once.

Must treat:
- `1/2/3` harmonics as birth support
- `5/7` harmonics as identity reinforcement

Forbidden shortcut:
- requiring static full harmonic presence in one frame.

### 4E. Event Resonance-Field Mapping

Input:
- `event_only` proto-exciters
- probe-derived family field

Output:
- `event_field` entities and framewise phase rows

Task:
- preserve non-note-like exciters as explicit event/gesture persistence instead of forcing them into note-chain identity;
- describe their phases as:
  - `ATTACK`
  - `RESONANCE_FIELD`
  - `DECAY_FIELD`

Law:
- event-field mapping is parallel to, not subordinate to, the note-chain branch.

### 5. Note-Bearing Chain Stabilization

Input:
- primary families

Output:
- `note_chain` hypotheses

Task:
- decide when a resonance family is stable enough to count as the note-bearing chain;
- allow temporary harmonic dropout if causal continuity remains alive.

This is the stage where the note becomes more than a proto-event.

### 6. Ownership Split

Input:
- note-chain hypotheses
- surrounding field evidence

Output:
- ownership-labelled structures

Ownership labels:
- `exciter_core`
- `note_chain`
- `box_body`
- `secondary_resonance`
- `unresolved`

Task:
- separate what belongs to the note from what belongs to the resonant body and what is already inherited field activity.

### 7. Box Transfer Mapping

Input:
- ownership-labelled structures

Output:
- transfer intervals

Task:
- detect the moment when note energy starts living more in the resonant body than in the exciter-led chain.

This is not note death yet. It is the transfer boundary.

### 8. Secondary Resonance Layer

Input:
- transfer intervals
- post-transfer field

Output:
- `secondary_tail` structures

Task:
- map the later resonance that is caused by the note but is no longer the note itself;
- preserve it for scene understanding without feeding it back into note identity.

### 9. Event Emission

Input:
- exciter, note-chain, transfer, secondary-tail structures

Output:
- canonical note events with explicit phases:
  - `birth`
  - `stabilization`
  - `transfer`
  - `tail`

Task:
- emit note events as phaseful entities, not flat framewise labels.

## What the pipeline must stop doing

The pipeline must stop collapsing the task into:
- decorative graph complexity
- framewise winner picking
- note identity built mostly from structural companions
- late bridge activity mistaken for the main note
- secondary resonance counted as fresh note evidence

## Evaluation law

Evaluation must happen on three aligned layers:

1. `Birth`
- did the pipeline detect the true event onset closely enough

2. `Sustain`
- did it keep the real note alive across the event window

3. `Tail`
- did it avoid confusing body carry and secondary resonance for continued note identity

## Current strongest diagnostic frontier

For Bach baseline the most important proven error class is:

`EXACT_BIRTH + MISSED_SUSTAIN`

This means the present pipeline can sometimes find the birth but then loses the note-bearing chain.

Therefore the next highest-value investigations are:
- same-degree drift after correct birth
- octave/root confusion after correct birth
- bridge takeover after correct birth
- companion takeover after correct birth
- early tail capture after correct birth

## Canonical implementation direction

The clean implementation direction for `Block002_pipeline` is:

1. keep `_audio_probe` as the canonical source
2. create exciter objects first
3. classify early continuation as `pitched / event / unresolved`
4. grow note-bearing chains only where a root-bearing branch actually emerges
5. split ownership before final note emission
6. emit note phases explicitly
7. evaluate changes by event-level audit, not by decorative structural richness

This document should be treated as the architectural law for the next rewrite steps.
