# How To Run

This example is meant to be reproducible from the real Block 4 pipeline, not by
hand-editing the results.

## Minimal public path

Run the public wrapper on a full Block 4 tonal dataset:

```powershell
python E:\Duodecimal_resonant_numeration\tools\run_block004_public_research_snapshot.py `
  --instrument-name RealPiano_1_1 `
  --dataset-dir E:\Duodecimal_resonant_numeration\Block004_data\RealPiano_1_1 `
  --refresh-lineage-passport
```

This reproduces the normal Block 4 pipeline stages:

- `dense`
- `chain`
- `root`
- `box`
- `box_split`
- `clean_box`
- `dense_vs_theory`
- `spiral12`
- `note_box_profile`
- `spiral3d`
- `harmonic_chain_spiral3d`
- `relation`
- `passport`

## If you want only the harmonic-lineage layer

If the earlier Block 4 outputs already exist, you can run the runner with a
reduced stage list:

```powershell
python E:\Duodecimal_resonant_numeration\tools\run_block004_public_research_snapshot.py `
  --instrument-name RealPiano_1_1 `
  --dataset-dir E:\Duodecimal_resonant_numeration\Block004_data\RealPiano_1_1 `
  --stages harmonic_chain_spiral3d,passport `
  --refresh-lineage-passport
```

## If you want cross-instrument morphology from existing compare HTML

```powershell
python E:\Duodecimal_resonant_numeration\py\music12\blocks\Block004_real_instruments\instrument_pipeline_runner_cli.py `
  --instrument_name compare `
  --audio_dir E:\Duodecimal_resonant_numeration\Block004_data\RealPiano_1_1\00_sources\audio_notes_wav `
  --manifest_csv E:\Duodecimal_resonant_numeration\Block004_data\RealPiano_1_1\01_manifest12\RealPiano_1_1__manifest12.csv `
  --reports_root E:\Duodecimal_resonant_numeration\Block004_data\RealPiano_1_1\10_reports `
  --stages harmonic_morphology_compare `
  --harmonic_morphology_html E:\Duodecimal_resonant_numeration\Block004_data\_multi_instrument_compare\90_public_compare\harmonic_amplitude_compare__9.5-__3d.html `
  --harmonic_morphology_out_dir E:\Duodecimal_resonant_numeration\Block004_data\_multi_instrument_compare\90_public_compare\harmonic_morphology_compare__9.5-
```

## Outputs to verify

After a successful run, verify that these exist:

- `55_harmonic_chain_spiral3d/*__harmonic_chain_spiral3d.png`
- `55_harmonic_chain_spiral3d/*__harmonic_chain_spiral3d.html`
- `55_harmonic_chain_spiral3d/*__harmonic_chain_spiral3d_summary.csv`
- `20_range_research/*__instrument_passport.json`
- `20_range_research/*__instrument_passport.md`

And confirm that the passport contains:

- `harmonic_chain_spiral3d`
- `harmonic_chain_notes_built`
- `harmonic_chain_unassigned_ratio`

## Scientific check

The run is useful if it lets you confirm that:

1. the note contains stable harmonic core structure;
2. note-box and residual layers are not random tails;
3. different harmonics appear to generate different resonance continuations;
4. morphology differs systematically across instruments;
5. `unassigned resonance` remains structured enough to deserve interpretation.

