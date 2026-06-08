param(
    [Parameter(Mandatory = $true)]
    [string]$ReportDir,

    [int]$IntervalSeconds = 20
)

$ErrorActionPreference = "Stop"

function Test-RunComplete {
    param([string]$Dir)

    $required = @(
        "ave_maria_framewise_candidates_micro_v1.csv",
        "ave_maria_micro_clusters_v1.csv",
        "ave_maria_micro_families_v1.csv"
    )

    foreach ($name in $required) {
        if (-not (Test-Path -LiteralPath (Join-Path $Dir $name))) {
            return $false
        }
    }
    return $true
}

function Send-AlertBeep {
    Add-Type -AssemblyName System.Windows.Forms | Out-Null
    1..4 | ForEach-Object {
        [System.Media.SystemSounds]::Exclamation.Play()
        Start-Sleep -Milliseconds 500
    }
}

$statePath = Join-Path $ReportDir "clean_probe_rerun_state.json"
$alertFlag = Join-Path $ReportDir "clean_probe_rerun_alerted.flag"
$watchLog = Join-Path $ReportDir "clean_probe_rerun_watch.log"

if (-not (Test-Path -LiteralPath $statePath)) {
    "state file not found: $statePath" | Set-Content -LiteralPath $watchLog -Encoding UTF8
    exit 1
}

"watcher started $(Get-Date -Format s)" | Set-Content -LiteralPath $watchLog -Encoding UTF8

while ($true) {
    $raw = Get-Content -LiteralPath $statePath -Raw
    $state = $raw | ConvertFrom-Json
    $runPid = [int]$state.pid
    $runAlive = $null -ne (Get-Process -Id $runPid -ErrorAction SilentlyContinue)
    $runComplete = Test-RunComplete -Dir $ReportDir

    $line = "{0} pid={1} alive={2} complete={3}" -f (Get-Date -Format s), $runPid, $runAlive, $runComplete
    Add-Content -LiteralPath $watchLog -Value $line -Encoding UTF8

    if ($runComplete) {
        Add-Content -LiteralPath $watchLog -Value "completed, watcher exiting" -Encoding UTF8
        break
    }

    if (-not $runAlive) {
        if (-not (Test-Path -LiteralPath $alertFlag)) {
            Send-AlertBeep
            "alerted $(Get-Date -Format s)" | Set-Content -LiteralPath $alertFlag -Encoding UTF8
            Add-Content -LiteralPath $watchLog -Value "run died before completion, alert sent" -Encoding UTF8
        }
        break
    }

    Start-Sleep -Seconds $IntervalSeconds
}
