param(
    [string]$ProjectRoot = "E:\Duodecimal_resonant_numeration",
    [string]$AnchorToken = "9.A-",
    [double]$AnchorHz = 440.0
)

$ErrorActionPreference = "Stop"

$pyRoot = Join-Path $ProjectRoot "py"
$blockRoot = Join-Path $ProjectRoot "Block004_data"
$reportsDir = Join-Path $ProjectRoot "docs\reports"
$logPath = Join-Path $reportsDir "block004_all_pitched_pdf_spiral_rebuild_2026-06-04.log"
$statePath = Join-Path $reportsDir "block004_all_pitched_pdf_spiral_rebuild_2026-06-04_state.json"

New-Item -ItemType Directory -Force -Path $reportsDir | Out-Null

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $logPath -Append
}

function Save-State {
    param(
        [string]$Status,
        [string]$CurrentInstrument,
        [int]$InstrumentIndex,
        [int]$InstrumentCount,
        [array]$Completed,
        [array]$Skipped,
        [array]$Failed
    )

    $state = [ordered]@{
        status = $Status
        current_instrument = $CurrentInstrument
        instrument_index = $InstrumentIndex
        instrument_count = $InstrumentCount
        completed = $Completed
        skipped = $Skipped
        failed = $Failed
        updated_at = (Get-Date).ToString("s")
        log_path = $logPath
    }

    $state | ConvertTo-Json -Depth 6 | Set-Content -Path $statePath -Encoding UTF8
}

function Resolve-AudioDir {
    param([string]$InstrumentRoot)

    $candidates = @(
        (Join-Path $InstrumentRoot "00_sources\audio_notes_wav"),
        (Join-Path $InstrumentRoot "00_sources\audio_notes")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    return $null
}

function Resolve-ManifestCsv {
    param(
        [string]$InstrumentRoot,
        [string]$InstrumentName
    )

    $manifestDir = Join-Path $InstrumentRoot "20_manifest"
    $exact = Join-Path $manifestDir ("{0}_manifest_12.csv" -f $InstrumentName)
    if (Test-Path $exact) {
        return $exact
    }

    $preferred = Get-ChildItem -Path $manifestDir -Filter "*manifest_12.csv" -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -notmatch "subset|fixed" } |
        Sort-Object Name

    if ($preferred) {
        return $preferred[0].FullName
    }

    $any = Get-ChildItem -Path $manifestDir -Filter "*.csv" -File -ErrorAction SilentlyContinue |
        Sort-Object Name

    if ($any) {
        return $any[0].FullName
    }

    return $null
}

function Run-Module {
    param([string[]]$Args)

    $env:PYTHONPATH = $pyRoot
    & python @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

$instrumentDirs = Get-ChildItem -Path $blockRoot -Directory |
    Where-Object {
        $_.Name -ne "percussion" -and
        $_.Name -ne "_multi_instrument_compare"
    } |
    Sort-Object Name

$completed = @()
$skipped = @()
$failed = @()

Write-Log "BLOCK004 pitched rebuild started"
Write-Log ("ProjectRoot: {0}" -f $ProjectRoot)
Write-Log ("Anchor: {0} @ {1}" -f $AnchorToken, $AnchorHz)
Write-Log ("Instruments: {0}" -f ($instrumentDirs.Name -join ", "))

Save-State -Status "running" -CurrentInstrument "" -InstrumentIndex 0 -InstrumentCount $instrumentDirs.Count -Completed $completed -Skipped $skipped -Failed $failed

for ($i = 0; $i -lt $instrumentDirs.Count; $i++) {
    $dir = $instrumentDirs[$i]
    $instrument = $dir.Name
    $instrumentRoot = $dir.FullName
    $reportsRoot = Join-Path $instrumentRoot "10_reports"
    $audioDir = Resolve-AudioDir -InstrumentRoot $instrumentRoot
    $manifestCsv = Resolve-ManifestCsv -InstrumentRoot $instrumentRoot -InstrumentName $instrument

    Save-State -Status "running" -CurrentInstrument $instrument -InstrumentIndex ($i + 1) -InstrumentCount $instrumentDirs.Count -Completed $completed -Skipped $skipped -Failed $failed

    if (-not (Test-Path $reportsRoot) -or -not $audioDir -or -not $manifestCsv) {
        $reason = [ordered]@{
            instrument = $instrument
            reports_root = $reportsRoot
            audio_dir = $audioDir
            manifest_csv = $manifestCsv
            reason = "missing_required_path"
        }
        $skipped += $reason
        Write-Log ("SKIP {0}: missing path(s)" -f $instrument)
        continue
    }

    try {
        Write-Log ("START {0} ({1}/{2})" -f $instrument, ($i + 1), $instrumentDirs.Count)
        Write-Log ("reports_root={0}" -f $reportsRoot)
        Write-Log ("audio_dir={0}" -f $audioDir)
        Write-Log ("manifest_csv={0}" -f $manifestCsv)

        Run-Module -Args @(
            "-m", "music12.blocks.Block004_real_instruments.reports_from_existing_dense_cli",
            "--reports_root", $reportsRoot,
            "--anchor_token", $AnchorToken,
            "--anchor_hz", "$AnchorHz"
        )

        Run-Module -Args @(
            "-m", "music12.blocks.Block004_real_instruments.instrument_pipeline_runner_cli",
            "--instrument_name", $instrument,
            "--audio_dir", $audioDir,
            "--manifest_csv", $manifestCsv,
            "--reports_root", $reportsRoot,
            "--stages", "box,box_split,note_box_profile,spiral3d,harmonic_chain_spiral3d,relation,passport",
            "--anchor_token", $AnchorToken,
            "--anchor_hz", "$AnchorHz"
        )

        $completed += [ordered]@{
            instrument = $instrument
            reports_root = $reportsRoot
            manifest_csv = $manifestCsv
            completed_at = (Get-Date).ToString("s")
        }
        Write-Log ("DONE {0}" -f $instrument)
    }
    catch {
        $failed += [ordered]@{
            instrument = $instrument
            reports_root = $reportsRoot
            manifest_csv = $manifestCsv
            error = $_.Exception.Message
            failed_at = (Get-Date).ToString("s")
        }
        Write-Log ("FAIL {0}: {1}" -f $instrument, $_.Exception.Message)
    }
}

$finalStatus = if ($failed.Count -gt 0) { "completed_with_failures" } else { "completed" }
Save-State -Status $finalStatus -CurrentInstrument "" -InstrumentIndex $instrumentDirs.Count -InstrumentCount $instrumentDirs.Count -Completed $completed -Skipped $skipped -Failed $failed
Write-Log ("FINISH status={0}; completed={1}; skipped={2}; failed={3}" -f $finalStatus, $completed.Count, $skipped.Count, $failed.Count)
