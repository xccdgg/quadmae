param(
    [string]$Root = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Root)) {
    $Root = $PSScriptRoot
}

$ReproDir = Join-Path $Root "repro_dmae"
$PretrainDir = Join-Path $ReproDir "cifar_pretrain_vitb_sigma025"
$FinetuneDir = Join-Path $ReproDir "finetune_sigma025"
$PauseLog = Join-Path $ReproDir "pause.log"

function Write-PauseLog {
    param([string]$Message)
    $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -Path $PauseLog -Value $line -Encoding UTF8
}

function Get-LatestCheckpointEpoch {
    param([string]$Dir)

    if (-not (Test-Path $Dir)) {
        return $null
    }

    $latest = Get-ChildItem -Path $Dir -Filter "checkpoint-*.pth" -File -ErrorAction SilentlyContinue |
        Where-Object { $_.BaseName -match '^checkpoint-(\d+)$' } |
        Sort-Object { [int]([regex]::Match($_.BaseName, '^checkpoint-(\d+)$').Groups[1].Value) } -Descending |
        Select-Object -First 1

    if ($null -eq $latest) {
        return $null
    }

    return [int]([regex]::Match($latest.BaseName, '^checkpoint-(\d+)$').Groups[1].Value)
}

function Get-ActiveStage {
    $proc = Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -match 'pretrain_cifar10.py|finetune_cifar10.py|certify_cifar10.py'
    } | Select-Object -First 1

    if ($null -eq $proc) {
        return $null
    }
    if ($proc.CommandLine -match 'pretrain_cifar10.py') {
        return "pretrain"
    }
    if ($proc.CommandLine -match 'finetune_cifar10.py') {
        return "finetune"
    }
    if ($proc.CommandLine -match 'certify_cifar10.py') {
        return "certify"
    }
    return $null
}

$stage = Get-ActiveStage
if ($null -eq $stage) {
    Write-PauseLog "no active stage found; nothing to pause"
    exit 0
}

if ($stage -eq "certify") {
    Write-PauseLog "active stage is certify; pause-after-epoch request ignored"
    exit 0
}

$stageDir = if ($stage -eq "pretrain") { $PretrainDir } else { $FinetuneDir }
$currentEpoch = Get-LatestCheckpointEpoch -Dir $stageDir
if ($null -eq $currentEpoch) {
    Write-PauseLog "stage=$stage no checkpoint found yet; waiting for first checkpoint"
    $targetEpoch = 0
}
else {
    $targetEpoch = $currentEpoch + 1
    Write-PauseLog "armed for stage=$stage current_checkpoint=$currentEpoch target_checkpoint=$targetEpoch"
}

while ($true) {
    Start-Sleep -Seconds 15

    $latestEpoch = Get-LatestCheckpointEpoch -Dir $stageDir
    if ($null -ne $latestEpoch -and $latestEpoch -ge $targetEpoch) {
        Write-PauseLog "target checkpoint reached stage=$stage checkpoint=$latestEpoch; stopping pipeline"

        $toStop = Get-CimInstance Win32_Process | Where-Object {
            $_.CommandLine -match 'run_repro_dmae_cifar10_sigma025.ps1|pretrain_cifar10.py|finetune_cifar10.py|certify_cifar10.py'
        } | Select-Object -ExpandProperty ProcessId

        if ($toStop) {
            Stop-Process -Id $toStop -Force
        }

        Write-PauseLog "pipeline paused"
        break
    }
}
