# Block 4 Public Reproducibility Guide

## Reproducing the isolated-note research pipeline

This guide is meant for outside researchers who want to verify that Block 4 is
not just a visual idea, but a working pipeline for analyzing real isolated
instrument notes.

It covers:

- what data are needed;
- which commands to run;
- which outputs to expect;
- how to interpret the result;
- how to confirm that the model is producing structured resonance behavior
  rather than a trivial noise tail.

---

## 1. What Block 4 tries to show

Block 4 studies isolated real notes as multi-layer causal resonance systems.

The working claim is not:

> "this note has one clean tone and then some unimportant remainder"

but rather:

> "this note contains harmonic cores, note-box growth, residual resonance,
> instrument-body response, and unresolved but meaningful causal continuations"

The pipeline is therefore designed to expose:

- harmonic chain structure;
- root stability;
- note-specific box behavior;
- 3D spiral geometry;
- harmonic-lineage structure;
- harmonic morphology across instruments;
- instrument passport summaries.

---

## 2. Required data layout

The public code assumes a dataset layout like:

```text
Block004_data/<instrument_name>/
  00_sources/
    audio_notes_wav/
      *.wav
  01_manifest12/
    <instrument>__manifest12.csv
  10_reports/
  20_range_research/
  30_note_box_profiles/
  50_spiral3d/
  55_harmonic_chain_spiral3d/
```

Typical real examples already present in this project:

- [RealPiano_1_1](../Block004_data/RealPiano_1_1)
- [cello](../Block004_data/cello)
- [violin](../Block004_data/violin)
- [piano_midi1](../Block004_data/piano_midi1)

If you are working from the public repository alone, use the compact public
example first, then connect it to the full offline corpora described in
[links/CLOUD_AND_ARCHIVE_LINKS.md](../links/CLOUD_AND_ARCHIVE_LINKS.md)
and the Zenodo records linked from the main README.

---

## 3. Minimal public entry point

For a compact reproducible path, use:

- [examples/block004-isolated-note-lineage-snapshot/README.md](../examples/block004-isolated-note-lineage-snapshot/README.md)

That example shows:

- one real isolated piano note;
- its harmonic-lineage 3D view;
- a cross-instrument harmonic-amplitude comparison;
- morphology summaries;
- the final interpretation of `unassigned resonance`.

---

## 4. One-command runner

This repository now includes a lightweight wrapper:

- [run_block004_public_research_snapshot.py](../tools/run_block004_public_research_snapshot.py)

It helps outside researchers run the existing Block 4 runner without manually
reassembling every path.

Example:

```powershell
python E:\Duodecimal_resonant_numeration\tools\run_block004_public_research_snapshot.py `
  --instrument-name RealPiano_1_1 `
  --dataset-dir E:\Duodecimal_resonant_numeration\Block004_data\RealPiano_1_1 `
  --refresh-lineage-passport
```

This wrapper auto-detects:

- the audio directory;
- the manifest;
- the reports directory;
- default Block 4 stage outputs.

---

## 5. What the wrapper runs

Internally it calls:

- [instrument_pipeline_runner_cli.py](../py/music12/blocks/Block004_real_instruments/instrument_pipeline_runner_cli.py)

The default stage list is:

```text
dense,chain,root,box,box_split,clean_box,dense_vs_theory,spiral12,note_box_profile,spiral3d,harmonic_chain_spiral3d,relation,passport
```

This means the public reproduction path is not a reduced toy workflow. It is a
compact way to invoke the real Block 4 pipeline.

---

## 6. Refreshing lineage passport summaries

After harmonic-lineage outputs exist, refresh the passport layer with:

```powershell
python E:\Duodecimal_resonant_numeration\tools\augment_block004_passports_with_harmonic_lineage.py
```

The wrapper can do this automatically when called with:

```text
--refresh-lineage-passport
```

This adds the `harmonic_chain_spiral3d` block and its summary fields to each
tonal instrument passport.

---

## 7. Optional morphology comparison stage

For cross-instrument harmonic behavior, use:

- [harmonic_morphology_compare_cli.py](../py/music12/blocks/Block004_real_instruments/harmonic_morphology_compare_cli.py)

or call it through the same wrapper with:

```text
--harmonic-morphology-html <path-to-plotly-html>
```

This produces:

- raw harmonic points;
- time-normalized harmonic curves;
- morphology features;
- pairwise harmonic distances;
- pairwise instrument morphology summaries.

---

## 8. Expected outputs

After a successful run, a third-party researcher should be able to inspect:

1. Per-note report folders in `10_reports/`
2. Note-box profiles in `30_note_box_profiles/`
3. Spiral3D note geometry in `50_spiral3d/`
4. Harmonic-lineage geometry in `55_harmonic_chain_spiral3d/`
5. Instrument passport files in `20_range_research/`

The most important final files are usually:

- `*__instrument_passport.json`
- `*__instrument_passport.md`
- `*__harmonic_chain_spiral3d_summary.csv`
- `*__box_harmonic_relation.csv`

---

## 9. What counts as a successful scientific check

The main verification target is not "perfect instrument separation".

The main scientific check is whether the outputs support the following claims:

1. The note contains stable harmonic cores, not only a noisy blur.
2. The surrounding box and residual layers are structured.
3. Different harmonics generate different resonance branches.
4. Cross-instrument morphology differs in systematic ways.
5. `unassigned resonance` remains structured even in controlled isolated-note recordings.

If these five claims are visible in the outputs, then the model is functioning
as a meaningful research model, even if later ensemble tasks remain unfinished.

---

## 10. How to interpret `unassigned resonance`

In Block 4, `unassigned resonance` should **not** be read as "empty tail".

It should be treated as a frontier class that may contain:

- late instrument-body returns;
- secondary unresolved branches;
- peripheral field responses;
- very high-order distant resonance responses.

This is one of the most important public conclusions of Block 4.

---

## 11. Recommended public reading order

For an outside researcher, the best order is:

1. [README.md](/E:/Duodecimal_resonant_numeration/README.md)
2. [examples/block004-isolated-note-lineage-snapshot/README.md](../examples/block004-isolated-note-lineage-snapshot/README.md)
3. [BLOCK004_REAL_INSTRUMENTS_RESEARCH_EN.md](BLOCK004_REAL_INSTRUMENTS_RESEARCH_EN.md)
4. this reproducibility guide
5. the wrapper script and example commands

That sequence should be enough to verify that Block 4 is a working research
pipeline and not just a collection of disconnected plots.
