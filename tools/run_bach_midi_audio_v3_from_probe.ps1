$ErrorActionPreference = "Stop"

$ProjectRoot = "E:\Duodecimal_resonant_numeration"
$PythonExe = "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.10_3.10.3056.0_x64__qbz5n2kfra8p0\python3.10.exe"
$ReportDir = Join-Path $ProjectRoot "Block001_data\Bach_Invention_1\10_reports_midi_audio_v3"
$ReferenceEventsCsv = Join-Path $ProjectRoot "Block001_data\Bach_Invention_1\00_sources\midi\bach_invention_1_midi_events_v1.csv"
$ProbeMetaJson = Join-Path $ReportDir "bach_midi_audio_probe_meta_micro_full.json"
$ProbeTimesCsv = Join-Path $ReportDir "bach_midi_audio_probe_times_micro_full.csv"
$Prefix = "bach_midi_audio"

$env:PYTHONPATH = Join-Path $ProjectRoot "py"
Set-Location $ProjectRoot

function Run-Step {
    param(
        [string]$Title,
        [string]$Module,
        [string]$Tag,
        [string[]]$ModuleArgs
    )

    Write-Host ""
    Write-Host ("=" * 80)
    Write-Host $Title
    Write-Host ("=" * 80)

    $cmd = @(
        $PythonExe,
        "-m", "music12.demons.demon_maxwell_cli",
        "-m", $Module,
        "--task-class", "module_run",
        "--project-root", $ProjectRoot,
        "--logdir", "_demon_logs",
        "--tag", $Tag,
        "--"
    ) + $ModuleArgs

    Write-Host ($cmd -join " ")
    & $PythonExe -m music12.demons.demon_maxwell_cli `
        -m $Module `
        --task-class module_run `
        --project-root $ProjectRoot `
        --logdir "_demon_logs" `
        --tag $Tag `
        -- @ModuleArgs

    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Title"
    }
}

$probeMeta = Get-Content -Raw $ProbeMetaJson | ConvertFrom-Json
$detectedDurationSec = [double]$probeMeta.time_slice.effective_duration_seconds
if ($detectedDurationSec -le 0) {
    $lastTime = Import-Csv $ProbeTimesCsv | Select-Object -Last 1
    $detectedDurationSec = [double]$lastTime.time_seconds
}

$referenceLast = Import-Csv $ReferenceEventsCsv | Select-Object -Last 1
$referenceDurationSec = [double]$referenceLast.end_sec

$matrix = Join-Path $ReportDir "${Prefix}_probe_matrix_micro_full.csv"
$times = Join-Path $ReportDir "${Prefix}_probe_times_micro_full.csv"
$coords = Join-Path $ReportDir "${Prefix}_probe_coords_micro_full.csv"

$framewise = Join-Path $ReportDir "${Prefix}_framewise_candidates_micro_v1.csv"
$framewiseReadable = Join-Path $ReportDir "${Prefix}_framewise_candidates_micro_v1_readable.csv"

$clusters = Join-Path $ReportDir "${Prefix}_micro_clusters_v1.csv"
$clustersReadable = Join-Path $ReportDir "${Prefix}_micro_clusters_v1_readable.csv"

$families = Join-Path $ReportDir "${Prefix}_micro_families_v1.csv"
$familyFrame = Join-Path $ReportDir "${Prefix}_micro_family_frame_summary_v1.csv"

$directedEdges = Join-Path $ReportDir "${Prefix}_micro_directed_edges_v1.csv"
$directedNodes = Join-Path $ReportDir "${Prefix}_micro_directed_nodes_v1.csv"

$causalRoles = Join-Path $ReportDir "${Prefix}_micro_causal_roles_v1.csv"
$causalCenters = Join-Path $ReportDir "${Prefix}_micro_causal_note_centers_v1.csv"

$simulFrames = Join-Path $ReportDir "${Prefix}_micro_simul_frame_notes_v1.csv"
$simulReadable = Join-Path $ReportDir "${Prefix}_micro_simul_readable_v1.csv"

$voiceEvents = Join-Path $ReportDir "${Prefix}_micro_voice_events_v1.csv"
$voiceSummary = Join-Path $ReportDir "${Prefix}_micro_voice_summary_v1.csv"
$frameVoice = Join-Path $ReportDir "${Prefix}_micro_frame_voice_v1.csv"

$stableVoices = Join-Path $ReportDir "${Prefix}_stable_voices_v1.csv"
$stableMapping = Join-Path $ReportDir "${Prefix}_stable_voice_mapping_v1.csv"

$compareCsv = Join-Path $ReportDir "${Prefix}_tempo_aligned_polyphony_vs_midi_v1.csv"

Run-Step "2. Candidate inference" `
    "music12.blocks.Block002_audio_recogn.resonance_candidate_inference_cli" `
    "${Prefix}_candidate_inference_v3" `
    @(
        "--matrix_csv", $matrix,
        "--times_csv", $times,
        "--coords_csv", $coords,
        "--out_framewise_csv", $framewise,
        "--out_framewise_readable_csv", $framewiseReadable,
        "--out_meta_json", (Join-Path $ReportDir "${Prefix}_framewise_candidates_micro_v1_meta.json"),
        "--energy_threshold", "0.003",
        "--top_n_candidates", "96",
        "--max_polyphonic_candidates", "48",
        "--analysis_min_hz", "30",
        "--analysis_max_hz", "12000"
    )

Run-Step "3. Micro candidate clustering" `
    "music12.blocks.Block002_audio_recogn.micro_candidate_cluster_cli" `
    "${Prefix}_micro_clusters_v3" `
    @(
        "--framewise_csv", $framewise,
        "--out_cluster_csv", $clusters,
        "--out_cluster_readable_csv", $clustersReadable,
        "--out_meta_json", (Join-Path $ReportDir "${Prefix}_micro_clusters_meta_v1.json"),
        "--out_summary_txt", (Join-Path $ReportDir "${Prefix}_micro_clusters_summary_v1.txt")
    )

Run-Step "4. Micro harmonic families" `
    "music12.blocks.Block002_audio_recogn.micro_harmonic_family_builder_cli" `
    "${Prefix}_micro_families_v3" `
    @(
        "--micro_clusters_csv", $clusters,
        "--out_family_csv", $families,
        "--out_frame_summary_csv", $familyFrame,
        "--out_meta_json", (Join-Path $ReportDir "${Prefix}_micro_family_meta_v1.json"),
        "--out_summary_txt", (Join-Path $ReportDir "${Prefix}_micro_family_summary_v1.txt"),
        "--anchor_token", "9.A'-",
        "--anchor_hz", "440",
        "--max_harmonic", "8",
        "--tolerance_cents", "35",
        "--max_families_per_frame", "12"
    )

Run-Step "5. Directed causality graph" `
    "music12.blocks.Block002_audio_recogn.micro_directed_causality_graph_cli" `
    "${Prefix}_micro_directed_v3" `
    @(
        "--micro_family_csv", $families,
        "--out_directed_edges_csv", $directedEdges,
        "--out_nodes_csv", $directedNodes,
        "--out_meta_json", (Join-Path $ReportDir "${Prefix}_micro_directed_meta_v1.json"),
        "--out_summary_txt", (Join-Path $ReportDir "${Prefix}_micro_directed_summary_v1.txt"),
        "--max_nodes_per_frame", "12",
        "--lag_min_frames", "1",
        "--lag_max_frames", "6",
        "--min_causal_frames", "5",
        "--min_causal_weight", "0.015"
    )

Run-Step "6. Causal role decomposition" `
    "music12.blocks.Block002_audio_recogn.micro_causal_role_decomposition_cli" `
    "${Prefix}_micro_causal_roles_v3" `
    @(
        "--directed_edges_csv", $directedEdges,
        "--out_roles_csv", $causalRoles,
        "--out_note_centers_csv", $causalCenters,
        "--out_meta_json", (Join-Path $ReportDir "${Prefix}_micro_causal_roles_meta_v1.json"),
        "--out_summary_txt", (Join-Path $ReportDir "${Prefix}_micro_causal_roles_summary_v1.txt"),
        "--min_center_score", "0.015"
    )

Run-Step "7. Simultaneous note disentanglement" `
    "music12.blocks.Block002_audio_recogn.micro_simultaneous_note_disentangler_cli" `
    "${Prefix}_micro_simul_v3" `
    @(
        "--micro_family_csv", $families,
        "--causal_centers_csv", $causalCenters,
        "--out_frame_notes_csv", $simulFrames,
        "--out_readable_csv", $simulReadable,
        "--out_meta_json", (Join-Path $ReportDir "${Prefix}_micro_simul_meta_v1.json"),
        "--out_summary_txt", (Join-Path $ReportDir "${Prefix}_micro_simul_summary_v1.txt"),
        "--min_center_score", "0.015",
        "--min_family_score", "0.20",
        "--max_notes_per_frame", "8",
        "--max_per_degree", "1"
    )

Run-Step "8. Voice continuity" `
    "music12.blocks.Block002_audio_recogn.micro_voice_continuity_tracker_cli" `
    "${Prefix}_micro_voice_v3" `
    @(
        "--frame_notes_csv", $simulFrames,
        "--out_voice_events_csv", $voiceEvents,
        "--out_voice_summary_csv", $voiceSummary,
        "--out_frame_voice_csv", $frameVoice,
        "--out_meta_json", (Join-Path $ReportDir "${Prefix}_micro_voice_meta_v1.json"),
        "--out_summary_txt", (Join-Path $ReportDir "${Prefix}_micro_voice_summary_v1.txt"),
        "--max_pitch_jump", "5",
        "--max_gap_frames", "3",
        "--min_voice_len_frames", "6"
    )

Run-Step "9. Voice identity stabilization" `
    "music12.blocks.Block002_audio_recogn.voice_identity_stabilizer_cli" `
    "${Prefix}_stable_voices_v3" `
    @(
        "--voice_events_csv", $voiceEvents,
        "--out_stable_voices_csv", $stableVoices,
        "--out_mapping_csv", $stableMapping,
        "--out_meta_json", (Join-Path $ReportDir "${Prefix}_stable_voices_meta_v1.json"),
        "--out_summary_txt", (Join-Path $ReportDir "${Prefix}_stable_voices_summary_v1.txt"),
        "--max_merge_gap_frames", "12",
        "--max_merge_pitch_jump", "4",
        "--min_stable_duration_frames", "12"
    )

Run-Step "10. Tempo-aligned MIDI comparison" `
    "music12.blocks.Block002_audio_recogn.tempo_aligned_polyphony_vs_midi_cli" `
    "${Prefix}_tempo_aligned_vs_midi_v3" `
    @(
        "--detected_frame_notes_csv", $simulFrames,
        "--reference_events_csv", $ReferenceEventsCsv,
        "--out_frame_compare_csv", $compareCsv,
        "--out_summary_json", (Join-Path $ReportDir "${Prefix}_tempo_aligned_polyphony_vs_midi_meta_v1.json"),
        "--out_summary_txt", (Join-Path $ReportDir "${Prefix}_tempo_aligned_polyphony_vs_midi_summary_v1.txt"),
        "--detected_duration_sec", ([string]::Format([System.Globalization.CultureInfo]::InvariantCulture, "{0}", $detectedDurationSec)),
        "--reference_duration_sec", ([string]::Format([System.Globalization.CultureInfo]::InvariantCulture, "{0}", $referenceDurationSec)),
        "--fps", "60"
    )

Run-Step "11. Polyphony diagnostics" `
    "music12.blocks.Block002_audio_recogn.polyphony_error_diagnostics_cli" `
    "${Prefix}_polyphony_diagnostics_v3" `
    @(
        "--frame_compare_csv", $compareCsv,
        "--out_error_summary_csv", (Join-Path $ReportDir "${Prefix}_polyphony_error_summary_v1.csv"),
        "--out_readable_csv", (Join-Path $ReportDir "${Prefix}_polyphony_readable_compare_v1.csv"),
        "--out_problem_windows_csv", (Join-Path $ReportDir "${Prefix}_polyphony_problem_windows_v1.csv"),
        "--out_meta_json", (Join-Path $ReportDir "${Prefix}_polyphony_error_diagnostics_meta_v1.json"),
        "--out_summary_txt", (Join-Path $ReportDir "${Prefix}_polyphony_error_diagnostics_summary_v1.txt"),
        "--problem_min_error", "4"
    )

Write-Host ""
Write-Host "FULL BACH MIDI AUDIO V3 PIPELINE FROM PROBE COMPLETE"
Write-Host "report_dir = $ReportDir"
Write-Host "detected_duration_sec = $detectedDurationSec"
Write-Host "reference_duration_sec = $referenceDurationSec"
