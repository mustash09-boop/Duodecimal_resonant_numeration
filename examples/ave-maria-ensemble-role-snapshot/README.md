# Ave Maria Ensemble Role Snapshot

Status: `ongoing research`

This example shows a short ensemble window from *Ave Maria* where the method is
used not to force a single instrument label, but to describe event roles inside
a shared stream:

- dominant instrument by layered affinity
- support instruments
- attack owner
- sustain owner
- body return
- unresolved or field-like residue

The goal of this example is not final source separation. It is a compact public
snapshot of how the project currently models mixed instrumental behavior.

## Window

- audio window: `80.0s -> 90.0s`
- source clip: [audio/ave_maria_fragment_80s_90s.wav](audio/ave_maria_fragment_80s_90s.wav)

## Reference vs current interpretation

From MIDI reference inside the same window:

- `142` MIDI note events
- active parts:
  - `Piano-Treble: 80`
  - `Organ-Treble: 22`
  - `Cello: 15`
  - `Violin: 15`
  - `Cello 2: 8`
  - `Organ-Bass: 2`

From current layered event interpretation:

- `108` layered events
- dominant instrument counts:
  - `cello: 36`
  - `organ: 31`
  - `piano: 23`
  - `UNRESOLVED_FIELD: 18`

From current role-behavior interpretation:

- `108` role-mapped events
- strongest role classes:
  - `INTERNAL_WAVE_EVENT: 46`
  - `BODY_RETURN_EVENT: 44`
  - `PRIMARY_WITH_SUPPORT_EVENT: 6`

This makes the example useful as a public snapshot of the current problem:
real ensemble sound is not only "which instrument is this note", but also
"which instrument owns attack, sustain, body return, or shared support".

## Included files

- [reference/ave_maria_fragment_midi_events.csv](reference/ave_maria_fragment_midi_events.csv)
- [pipeline/ave_maria_fragment_layered_assignment.csv](pipeline/ave_maria_fragment_layered_assignment.csv)
- [pipeline/ave_maria_fragment_role_behavior_map.csv](pipeline/ave_maria_fragment_role_behavior_map.csv)
- [pipeline/ave_maria_fragment_summary.txt](pipeline/ave_maria_fragment_summary.txt)
- [HOW_TO_RUN.md](HOW_TO_RUN.md)
