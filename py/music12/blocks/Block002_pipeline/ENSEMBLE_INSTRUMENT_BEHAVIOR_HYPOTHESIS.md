# Ensemble Instrument Behavior Hypothesis

## Main conclusion

Static instrument passports are not enough for separation inside a shared musical flow.

What fails:
- direct waveform subtraction;
- note-for-note substitution from a real single-note library;
- absolute affinity to one instrument passport;
- one-label ownership of every event.

What this means:
- the project does not primarily need `instrument fingerprints as fixed values`;
- it needs `instrument behavior laws inside a shared event`.

## Why the previous route failed

The `Ave Maria` experiments showed three different truths:

1. A note can be correct but live in the wrong local time.
2. A note can be in the right local time but still fail as subtraction because the wave, phase, room and attack are different.
3. A shared event often belongs to more than one instrument, but with different roles.

Because of this, the question is not:
- `which instrument is this exact spectrum value`

but:
- `which instrument role is being performed in this local event phase`

## Constructor reading

Using the architecture constructor logic:

### MirrorNeuronAxiom

Before any separation step, the system must classify the local phenomenon:
- new attack;
- carried sustain;
- instrument body return;
- hall / field return;
- shared event;
- unresolved field.

### LearnedPathway

The already proven useful pathways are:
- event lifecycle;
- separation of attack from sustain;
- separation of note from body / echo / secondary response;
- layered ownership instead of forced single winner.

These should not be discarded.

### PathwayRotor

The next useful synthesis is:
- `event lifecycle`
+ `instrument passport`
+ `role-in-event assignment`

not:
- `passport`
+ `wave subtraction`

### MaxwellDaemon

Direct waveform subtraction is now downgraded from main path to diagnostic path.

It can still be useful as a probe, but it is not the canonical route for ensemble separation.

### PersonalMemory

We must preserve this negative result:
- even with correct real piano notes and better time alignment, direct subtraction still fails.

This is a useful boundary, not a wasted experiment.

### ValueCore

The priority is:
1. truthful event-role modeling;
2. stable instrument behavior recognition;
3. only then reconstructed stems.

## New hypothesis

Each instrument must be represented not by one passport, but by a phased behavior passport.

For each instrument, the system should learn:
- how it is born;
- how it sustains;
- how its body answers;
- how it coexists with other instruments;
- how it leaves traces after the main note identity is gone.

## Proposed behavior passport

Each instrument passport should contain phase-specific behavior descriptors.

### 1. Attack law

Descriptors:
- onset compactness;
- birth sharpness;
- attack-to-body delay;
- early harmonic concentration;
- register-sensitive attack spread.

Meaning:
- piano often has compact causal birth;
- strings often enter less compactly and may share continuity with previous energy;
- organ often lacks a percussive attack and behaves more like immediate sustained field.

### 2. Sustain law

Descriptors:
- sustain stability;
- note-class persistence;
- octave drift tolerance;
- harmonic ladder persistence;
- half-life of exact identity.

Meaning:
- piano sustain decays and releases identity faster;
- cello and violin can keep identity through longer continuation;
- organ sustain can remain stable with less attack evidence.

### 3. Body-response law

Descriptors:
- body-return ratio;
- delayed resonance density;
- low / mid / high body profile;
- same-note near-rebirth tendency;
- internal-wave tendency.

Meaning:
- piano body returns strongly;
- strings may have less box-return but longer sustained continuity;
- organ may have broad stable field rather than percussive body-return.

### 4. Field / hall law

Descriptors:
- likely hall-trace delay;
- far-field persistence;
- non-causal repeated tail tendency;
- low-information late return density.

Meaning:
- this must be separated from the instrument body itself.

### 5. Co-presence law

Descriptors:
- common shared partners;
- attack-owner vs sustain-owner tendency;
- support-role tendency;
- mixed-window tolerance.

Meaning:
- piano can be attack-owner while cello is sustain-owner;
- organ often coexists in mixed windows rather than isolated windows;
- some events should remain multi-owned.

## New event model

Instead of:
- `event -> one instrument`

the model should become:
- `event -> one or more instrument roles`

Roles:
- `attack_owner`
- `sustain_owner`
- `body_owner`
- `field_owner`
- `support_owner`
- `unresolved`

This is the key hypothesis.

It explains:
- why cello leaks into piano masks;
- why organ lives with piano;
- why direct subtraction is weak;
- why layered assignment was more honest than hard classification.

## Canonical next algorithm

### Stage A. Event-role decomposition

For each event or event-fragment:
- split early attack frames;
- split sustain frames;
- split body-return frames;
- split field-return frames.

### Stage B. Behavior scoring

For each phase fragment:
- compute behavior descriptors;
- compare them to instrument behavior passports;
- produce role scores, not only instrument scores.

### Stage C. Layered ownership map

For each frame or family:
- assign one or more roles:
  - `piano.attack_owner`
  - `cello.sustain_owner`
  - `organ.support_owner`
  - `hall.field_owner`

### Stage D. Stem reconstruction

Only after role assignment:
- build frame masks;
- build spectral masks;
- reconstruct stems.

## Practical immediate version

The first implementable version does not need full spectral intelligence.

It can start with event-level role inference using already existing artifacts:
- lifecycle tags;
- layered assignment;
- body / wave / hall audit;
- real piano / violin / cello / organ passports.

Immediate role rules:
- `piano attack owner`
  when birth is compact and target-window support is strong;
- `cello sustain owner`
  when exact piano overlap is weak but continuation is long and stable;
- `organ support owner`
  when mixed-window presence is broad and attack evidence is weak;
- `body owner`
  when near-rebirth is same-note and body-return-like;
- `field owner`
  when weak delayed trace survives without local causal birth.

## Expected benefit

This route should:
- reduce false cello capture inside piano;
- preserve real shared events instead of forcing them apart;
- make piano+organ coexistence explicit;
- give better masks than direct passport-only affinity;
- make WAV reconstruction secondary and more realistic.

## What to build next

The next concrete module should be:

- `instrument_role_behavior_mapper`

Input:
- cleaned event stream;
- layered instrument assignment;
- lifecycle summaries;
- body / hall / re-excitation audit;
- instrument behavior passports.

Output:
- per-event role map;
- per-frame role priors;
- stem-ready layered mask hints.
