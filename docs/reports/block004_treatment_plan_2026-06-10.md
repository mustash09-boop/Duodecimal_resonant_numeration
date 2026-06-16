# Block004 Treatment Plan — 2026-06-10

## Audit Basis

- per-note audit CSV: `block004_spiral_audit_notes_2026-06-10.csv`
- per-instrument audit CSV: `block004_spiral_audit_instruments_2026-06-10.csv`
- summary: `block004_spiral_audit_summary_2026-06-10.txt`

Important caveat:
- for many legacy note folders the expected note is not recoverable from the folder name alone, so `root mismatch 999` in the audit is not always a real pitch error;
- the decisive signals are therefore:
  - old `spiral12_clean_points.csv` schema without harmonic-marker fields,
  - old `note_box_profile.csv` schema without `freq_ratio / early_ratio / late_ratio`,
  - late `note_box` / `dense_other` vortex inflation,
  - and outdated `spiral3d` HTML geometry/autoscale behavior.

## Possible Reference Notes

These notes are no longer in the catastrophic category and can serve as working references:

- `RealPiano_1_1 / 022_piano_real_7.7-`
  - status: `PARTIAL_FIX`
  - fixed: root, `spiral12`, `note_box_profile`, tail cleanup
  - still pending: modern `spiral3d` HTML autoscale behavior

- `Bass_guitar / 15_7.7-_bass-guitar_4string`
  - status: `PARTIAL_FIX`
  - fixed: root, `spiral12`, `note_box_profile`, geometry lock
  - still pending: residual dense tail inflation, modern legend-driven autoscale

- `guitar2 / 032_8.8-_guitar2_5string`
  - status: `PARTIAL_FIX`
  - fixed: root, `spiral12`, geometry lock
  - still pending: new `note_box_profile` schema, residual dense tail inflation, modern legend-driven autoscale

- `banjo / banjo_G3_very-long_forte_normal`
  - status: keep as visual/sonic reference only with caution
  - root and `spiral12` are repaired, but the note still has a real tail `note_box` vortex and is not yet clean enough to count as partial-safe

## Full Heal First

These instruments still need a full therapeutic pass, not cosmetic patching:

- `bass_clarinet`
- `bassoon`
- `cello`
- `cello2`
- `clarinet`
- `contrabassoon`
- `cor_anglais`
- `double_bass`
- `double-bass2`
- `flute`
- `french_horn`
- `guitar`
- `mandolin`
- `oboe`
- `piano_midi1`
- `saxophone`
- `viola`
- `violin`
- `violin2`

And also the bulk of these four instruments still needs full healing even though one reference note was repaired:

- `RealPiano_1_1`
- `Bass_guitar`
- `banjo`
- `guitar2`

## What “Full Heal” Means

For each affected instrument the healing pass should rebuild, in order:

1. `10_reports`
   - correct root selection
   - harmonic-marker `spiral12_clean_points.csv`
   - `__spiral12_clean.png` as harmonic markers, not a dense dumb spiral

2. `30_note_box_profiles`
   - new schema with `freq_ratio / early_ratio / late_ratio`
   - suppression of fake late `note_box` growth

3. `50_spiral3d`
   - equal `x/y` geometry
   - no visually squashed spiral
   - removal or thinning of late vortex noise

4. compare/public layers
   - only after the note-level layers are corrected

## Current Global Conclusion

- `NO_FIX`: none yet at the raw Block004 note-report level
- `PARTIAL_FIX`: exactly the three repaired reference notes above
- `FULL_FIX`: everything else

This means the mass rebuild from `2026-06-05` should currently be treated as structurally unreliable for almost the whole Block004, except for the explicitly repaired reference notes.
