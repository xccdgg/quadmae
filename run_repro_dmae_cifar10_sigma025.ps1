param(
    [ValidateSet("all", "pretrain", "finetune", "certify")]
    [string]$Stage = "all",
    [string]$PythonExe = "D:\anaconda3\envs\dmae38\python.exe",
    [string]$Root = "",
    [string]$Data = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Root)) {
    $Root = $PSScriptRoot
}
if ([string]::IsNullOrWhiteSpace($Data)) {
    $Data = Join-Path $Root "data"
}

$PretrainDir = Join-Path $Root "repro_dmae\cifar_pretrain_vitb_sigma025"
$FinetuneDir = Join-Path $Root "repro_dmae\finetune_sigma_uniform_0_075"
$CertDir025 = Join-Path $Root "repro_dmae\eval_sigma025_num10000"
$CertDir05 = Join-Path $Root "repro_dmae\eval_sigma05_num10000"
$InitCkpt = Join-Path $Root "dmae_base_sigma_0.25_mask_0.75_1100e.pth"

function Get-LatestCheckpoint {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Dir
    )

    if (-not (Test-Path $Dir)) {
        return $null
    }

    $latest = Get-ChildItem -Path $Dir -Filter "checkpoint-*.pth" -File -ErrorAction SilentlyContinue |
        Where-Object { $_.BaseName -match '^checkpoint-(\d+)$' } |
        Sort-Object { [int]([regex]::Match($_.BaseName, '^checkpoint-(\d+)$').Groups[1].Value) } -Descending |
        Select-Object -First 1

    if ($null -ne $latest) {
        return $latest.FullName
    }

    return $null
}

function Get-BestR0Checkpoint {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Dir
    )

    $best = Join-Path $Dir "best-r0.pth"
    if (Test-Path $best) {
        return $best
    }

    return $null
}

New-Item -ItemType Directory -Force -Path $PretrainDir, $FinetuneDir, $CertDir025, $CertDir05, $Data | Out-Null

Push-Location $Root
try {
    if ($Stage -in @("all", "pretrain")) {
        $PretrainResume = Get-LatestCheckpoint -Dir $PretrainDir
        if ([string]::IsNullOrWhiteSpace($PretrainResume)) {
            $PretrainResume = $InitCkpt
        }

        & $PythonExe pretrain_cifar10.py `
            --data_path $Data `
            --output_dir $PretrainDir `
            --log_dir $PretrainDir `
            --resume $PretrainResume `
            --batch_size 64 `
            --accum_iter 8 `
            --model dmae_vit_base_patch16 `
            --norm_pix_loss `
            --mask_ratio 0.75 `
            --sigma 0.25 `
            --start_epoch 0 `
            --epochs 50 `
            --warmup_epochs 10 `
            --blr 5e-5 `
            --weight_decay 0.05 `
            --device cuda
    }

    if ($Stage -in @("all", "finetune")) {
        $LatestFinetune = Get-LatestCheckpoint -Dir $FinetuneDir
        $PretrainCkpt = Get-LatestCheckpoint -Dir $PretrainDir

        if ([string]::IsNullOrWhiteSpace($LatestFinetune)) {
            if ([string]::IsNullOrWhiteSpace($PretrainCkpt)) {
                throw "No pretrain checkpoint found in $PretrainDir"
            }

            & $PythonExe finetune_cifar10.py `
                --data_path $Data `
                --finetune $PretrainCkpt `
                --output_dir $FinetuneDir `
                --log_dir $FinetuneDir `
                --batch_size 64 `
                --accum_iter 4 `
                --model vit_base_patch16 `
                --sigma "[0,0.75]" `
                --use_quaternion_noise false `
                --con_reg `
                --reg_lbd 2.0 `
                --reg_eta 0.5 `
                --epochs 50 `
                --blr 5e-4 `
                --layer_decay 0.65 `
                --weight_decay 0.05 `
                --drop_path 0.1 `
                --device cuda
        }
        else {
            & $PythonExe finetune_cifar10.py `
                --data_path $Data `
                --resume $LatestFinetune `
                --output_dir $FinetuneDir `
                --log_dir $FinetuneDir `
                --batch_size 64 `
                --accum_iter 4 `
                --model vit_base_patch16 `
                --sigma "[0,0.75]" `
                --use_quaternion_noise false `
                --con_reg `
                --reg_lbd 2.0 `
                --reg_eta 0.5 `
                --epochs 50 `
                --blr 5e-4 `
                --layer_decay 0.65 `
                --weight_decay 0.05 `
                --drop_path 0.1 `
                --device cuda
        }
    }

    if ($Stage -in @("all", "certify")) {
        $FinetuneCkpt = Get-BestR0Checkpoint -Dir $FinetuneDir
        if ([string]::IsNullOrWhiteSpace($FinetuneCkpt)) {
            $FinetuneCkpt = Get-LatestCheckpoint -Dir $FinetuneDir
        }
        if ([string]::IsNullOrWhiteSpace($FinetuneCkpt)) {
            throw "No finetune checkpoint found in $FinetuneDir"
        }

        foreach ($Eval in @(
            @{ Sigma = "0.25"; Dir = $CertDir025 },
            @{ Sigma = "0.5"; Dir = $CertDir05 }
        )) {
            & $PythonExe certify_cifar10.py `
                --resume $FinetuneCkpt `
                --model vit_base_patch16 `
                --data_path $Data `
                --output_dir $Eval["Dir"] `
                --sigma $Eval["Sigma"] `
                --sample_interval 1 `
                --num 10000 `
                --nb_classes 10 `
                --use_quaternion_noise false `
                --device cuda:0
        }
    }
}
finally {
    Pop-Location
}
