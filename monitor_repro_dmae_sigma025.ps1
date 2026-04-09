param(
    [string]$Root = "",
    [int]$PollSeconds = 120
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Root)) {
    $Root = $PSScriptRoot
}

$ReproDir = Join-Path $Root "repro_dmae"
$PretrainDir = Join-Path $ReproDir "cifar_pretrain_vitb_sigma025"
$FinetuneDir = Join-Path $ReproDir "finetune_sigma025"
$CertDir = Join-Path $ReproDir "cert_sigma025_num10000"
$MonitorLog = Join-Path $ReproDir "monitor.log"

function Write-MonitorLog {
    param([string]$Message)
    $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -Path $MonitorLog -Value $line -Encoding UTF8
}

function Get-LatestCheckpointName {
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

    return $latest.Name
}

function Get-Stage {
    $proc = Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -match 'pretrain_cifar10.py|finetune_cifar10.py|certify_cifar10.py'
    } | Select-Object -First 1

    if ($null -eq $proc) {
        return "idle"
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
    return "idle"
}

New-Item -ItemType Directory -Force -Path $ReproDir | Out-Null
Write-MonitorLog "monitor started"

$lastStage = $null
$lastPretrainCkpt = $null
$lastFinetuneCkpt = $null
$lastCertState = $null

while ($true) {
    $stage = Get-Stage
    if ($stage -ne $lastStage) {
        Write-MonitorLog "stage=$stage"
        $lastStage = $stage
    }

    $pretrainCkpt = Get-LatestCheckpointName -Dir $PretrainDir
    if ($pretrainCkpt -and $pretrainCkpt -ne $lastPretrainCkpt) {
        Write-MonitorLog "pretrain latest checkpoint=$pretrainCkpt"
        $lastPretrainCkpt = $pretrainCkpt
    }

    $finetuneCkpt = Get-LatestCheckpointName -Dir $FinetuneDir
    if ($finetuneCkpt -and $finetuneCkpt -ne $lastFinetuneCkpt) {
        Write-MonitorLog "finetune latest checkpoint=$finetuneCkpt"
        $lastFinetuneCkpt = $finetuneCkpt
    }

    $certCsv = Join-Path $CertDir "certified_accuracy.csv"
    $certState = if (Test-Path $certCsv) { "ready" } else { "pending" }
    if ($certState -ne $lastCertState) {
        Write-MonitorLog "certified_accuracy=$certState"
        $lastCertState = $certState
    }

    if ($certState -eq "ready" -and $stage -eq "idle") {
        Write-MonitorLog "monitor finished"
        break
    }

    Start-Sleep -Seconds $PollSeconds
}
