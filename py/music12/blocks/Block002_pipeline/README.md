# Block002 Pipeline

This package is the clean-room assembly area for the canonical Block002
streaming note pipeline.

Primary architectural law:
- [CANONICAL_EXCITATION_RESONANCE_PIPELINE.md](/E:/Duodecimal_resonant_numeration/py/music12/blocks/Block002_pipeline/CANONICAL_EXCITATION_RESONANCE_PIPELINE.md)

Current migrated stages:
- `excitation_seed_extractor_cli.py`
- `proto_exciter_builder_cli.py`
- `exciter_branch_classifier_cli.py`
- `event_resonance_field_mapper_cli.py`
- `event_field_onset_group_merger_cli.py`
- `cross_branch_event_fusion_cli.py`
- `primary_note_chain_builder_cli.py`
- `controlled_sustain_transfer_mapper_cli.py`
- `pipeline_target_alignment_audit_cli.py`
- `resonance_candidate_inference_core.py`
- `resonance_candidate_inference_cli.py`
- `micro_candidate_cluster_cli.py`
- `micro_harmonic_family_builder_cli.py`
- `resonance_event_lifecycle_tracker_v2_cli.py`
- `resonance_entity_builder_cli.py`
- `resonance_entity_persistence_stabilizer_cli.py`
- `resonance_entity_trajectory_persistence_cli.py`
- `resonance_influence_graph_cli.py`
- `resonance_causality_flow_tracker_cli.py`
- `resonance_field_dynamics_cli.py`
- `resonance_ownership_resolution_cli.py`
- `micro_directed_causality_graph_cli.py`
- `micro_causal_role_decomposition_cli.py`
- `micro_simultaneous_note_disentangler_cli.py`
- `micro_voice_continuity_tracker_cli.py`
- `voice_identity_stabilizer_cli.py`
- `causal_note_hypothesis_resolver_cli.py`
- `tempo_aligned_polyphony_vs_midi_cli.py`
- `polyphony_error_diagnostics_cli.py`

Current order:
1. `excitation_seed_extractor_cli.py`
2. `proto_exciter_builder_cli.py`
3. `exciter_branch_classifier_cli.py`
4. `event_resonance_field_mapper_cli.py`
5. `event_field_onset_group_merger_cli.py`
6. `primary_note_chain_builder_cli.py`
7. `controlled_sustain_transfer_mapper_cli.py`
8. `cross_branch_event_fusion_cli.py`
9. `pipeline_target_alignment_audit_cli.py`
10. `resonance_candidate_inference_cli.py`
11. `micro_candidate_cluster_cli.py`
12. `micro_harmonic_family_builder_cli.py`
13. `resonance_event_lifecycle_tracker_v2_cli.py`
14. `resonance_entity_builder_cli.py`
15. `resonance_entity_persistence_stabilizer_cli.py`
16. `resonance_entity_trajectory_persistence_cli.py`
17. `resonance_influence_graph_cli.py`
18. `resonance_causality_flow_tracker_cli.py`
19. `resonance_field_dynamics_cli.py`
20. `resonance_ownership_resolution_cli.py`
21. `micro_directed_causality_graph_cli.py`
22. `micro_causal_role_decomposition_cli.py`
23. `micro_simultaneous_note_disentangler_cli.py`
24. `micro_voice_continuity_tracker_cli.py`
25. `voice_identity_stabilizer_cli.py`
26. `causal_note_hypothesis_resolver_cli.py`
27. `tempo_aligned_polyphony_vs_midi_cli.py`
28. `polyphony_error_diagnostics_cli.py`

Near-term planned migrations:
- `resonance_structure_assembler_cli.py`
- `resonance_body_decomposition_cli.py`
- `resonance_field_persistence_mapper_cli.py`
- `resonance_attractor_resolver_cli.py`
- `composition_note_box_scene_builder_cli.py`
- `scene_consistent_note_resolution_cli.py`
- `adaptive_attentional_resonance_focus_cli.py`

State-machine target:
`EXCITATION_BIRTH -> ROOT_HYPOTHESIS -> CHAIN_STABILIZATION -> OWNERSHIP_SPLIT -> BOX_TRANSFER -> SECONDARY_RESONANCE_TAIL`

Current branch law:
- keep one universal excitation-first entrypoint from `_audio_probe`
- branch after proto-exciter emergence:
  - `pitched / unresolved -> note-chain branch`
  - `event -> gesture/resonance-field branch`
- the current runner routes `pitched + unresolved` into `primary_note_chain`
- short and weak event-like fallback sparks are promoted to `event_field_candidate`
- `event_only + event_field_candidate` exciters go through `event_resonance_field_mapper_cli.py`
- raw event-field entities must then be merged into onset groups before target-alignment counting
- cross-branch fusion must use onset-local evidence only, so long sustained instruments are not broken by lifetime-overlap heuristics
- global progress must also be read against exact MIDI target counts from `bach_invention_1_midi_events_meta_v1.json`

Current causality-flow refinements:
- `EXCITATION_TO_CHAIN`
- `NOTE_TO_BOX_TRANSFER`
- `BOX_TO_SECONDARY_RESONANCE`

Current tuning notes:
- `micro_directed_causality_graph_cli.py` now keeps a weaker same-degree sustain layer
  instead of discarding it completely.
- `micro_causal_role_decomposition_cli.py` separates `bridge_resonator` from true
  note centers and requires asymmetry before a bridge can become a center.
- `micro_simultaneous_note_disentangler_cli.py` allows structurally strong
  companions beside centers, using isolated-note chain experience from
  `Block004_data/piano_midi1` as a reference signal, not as an absolute oracle.
- `causal_note_hypothesis_resolver_cli.py` lifts note identity from framewise
  selections into longer stable-voice hypotheses, rewarding early exciter
  support and sustained local runs over late bridge-like resonance tails.
