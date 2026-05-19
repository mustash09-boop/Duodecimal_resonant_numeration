param(
    [string]$DestinationRoot = "E:\Duodecimal_resonant_numeration\ops\jim_memory\codex_state_backups"
)

$ErrorActionPreference = "Stop"

function New-DirectoryIfMissing {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

function Invoke-RobocopySafe {
    param(
        [string]$Source,
        [string]$Destination
    )

    if (-not (Test-Path -LiteralPath $Source)) {
        return [pscustomobject]@{
            Source = $Source
            Destination = $Destination
            Status = "missing"
            ExitCode = $null
        }
    }

    New-DirectoryIfMissing -Path $Destination

    $null = & robocopy $Source $Destination /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP `
        /XD "Cache" "Code Cache" "GPUCache" "DawnGraphiteCache" "DawnWebGPUCache" `
        /XF "lockfile" "LOCK" "Cookies" "Cookies-journal"
    $exitCode = $LASTEXITCODE

    if ($exitCode -gt 7) {
        throw "Robocopy failed for '$Source' with exit code $exitCode"
    }

    return [pscustomobject]@{
        Source = $Source
        Destination = $Destination
        Status = "copied"
        ExitCode = $exitCode
    }
}

$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$snapshotDir = Join-Path $DestinationRoot $timestamp

New-DirectoryIfMissing -Path $DestinationRoot
New-DirectoryIfMissing -Path $snapshotDir

$sources = @(
    @{
        Name = "Roaming_Codex"
        Path = "C:\Users\Alex\AppData\Roaming\Codex"
    },
    @{
        Name = "Local_Codex_Logs"
        Path = "C:\Users\Alex\AppData\Local\Codex\Logs"
    }
)

$results = @()
foreach ($source in $sources) {
    $destination = Join-Path $snapshotDir $source.Name
    $results += Invoke-RobocopySafe -Source $source.Path -Destination $destination
}

$metadata = [pscustomobject]@{
    created_at = (Get-Date).ToString("s")
    snapshot_dir = $snapshotDir
    machine_user = $env:USERNAME
    note = "Backup of local Codex state into persistent Jim memory on E:"
    items = $results
}

$metadata | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $snapshotDir "snapshot_meta.json") -Encoding UTF8

$summary = @(
    "Jim memory Codex backup"
    "Created: $($metadata.created_at)"
    "Snapshot: $snapshotDir"
    ""
)

foreach ($item in $results) {
    $summary += "[$($item.Status)] $($item.Source) -> $($item.Destination)"
}

$summary | Set-Content -LiteralPath (Join-Path $snapshotDir "snapshot_summary.txt") -Encoding UTF8

Write-Output "Backup created at: $snapshotDir"
