#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="all"
MODE="gauss"
TORCHRUN_BIN="torchrun"
ROOT="${SCRIPT_DIR}"
DATA=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)
      STAGE="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --torchrun)
      TORCHRUN_BIN="$2"
      shift 2
      ;;
    --root)
      ROOT="$2"
      shift 2
      ;;
    --data)
      DATA="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${DATA}" ]]; then
  DATA="${ROOT}/data"
fi

case "${STAGE}" in
  all|pretrain|finetune|certify) ;;
  *)
    echo "Invalid --stage: ${STAGE}" >&2
    exit 1
    ;;
esac

case "${MODE}" in
  gauss)
    USE_QWT="false"
    PRETRAIN_DIR="${ROOT}/repro_dmae_gauss_uniform_0_075/cifar_pretrain_vitb_sigma025"
    FINETUNE_DIR="${ROOT}/repro_dmae_gauss_uniform_0_075/finetune_sigma_uniform_0_075"
    CERT_DIR_025="${ROOT}/repro_dmae_gauss_uniform_0_075/eval_sigma025_num10000"
    CERT_DIR_05="${ROOT}/repro_dmae_gauss_uniform_0_075/eval_sigma05_num10000"
    ;;
  qwt)
    USE_QWT="true"
    PRETRAIN_DIR="${ROOT}/repro_dmae_qwt_uniform_0_075/cifar_pretrain_vitb_sigma025_qwt"
    FINETUNE_DIR="${ROOT}/repro_dmae_qwt_uniform_0_075/finetune_sigma_uniform_0_075_qwt"
    CERT_DIR_025="${ROOT}/repro_dmae_qwt_uniform_0_075/eval_sigma025_qwt_num10000"
    CERT_DIR_05="${ROOT}/repro_dmae_qwt_uniform_0_075/eval_sigma05_qwt_num10000"
    ;;
  *)
    echo "Invalid --mode: ${MODE}" >&2
    exit 1
    ;;
esac

INIT_CKPT="${ROOT}/dmae_base_sigma_0.25_mask_0.75_1100e.pth"
LEVELS="1"
RATIO="3.0"

latest_checkpoint() {
  local dir="$1"
  if [[ ! -d "${dir}" ]]; then
    return 0
  fi
  find "${dir}" -maxdepth 1 -name 'checkpoint-*.pth' -type f | sort -V | tail -n 1
}

best_r0_checkpoint() {
  local dir="$1"
  local best="${dir}/best-r0.pth"
  if [[ -f "${best}" ]]; then
    printf '%s\n' "${best}"
  fi
}

mkdir -p "${PRETRAIN_DIR}" "${FINETUNE_DIR}" "${CERT_DIR_025}" "${CERT_DIR_05}" "${DATA}"
cd "${ROOT}"

if [[ "${STAGE}" == "all" || "${STAGE}" == "pretrain" ]]; then
  PRETRAIN_RESUME="$(latest_checkpoint "${PRETRAIN_DIR}")"
  if [[ -z "${PRETRAIN_RESUME}" ]]; then
    PRETRAIN_RESUME="${INIT_CKPT}"
  fi

  "${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc_per_node=1 pretrain_cifar10.py \
    --data_path "${DATA}" \
    --output_dir "${PRETRAIN_DIR}" \
    --log_dir "${PRETRAIN_DIR}" \
    --resume "${PRETRAIN_RESUME}" \
    --batch_size 64 \
    --accum_iter 8 \
    --model dmae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --sigma 0.25 \
    --use_quaternion_noise "${USE_QWT}" \
    --levels "${LEVELS}" \
    --ratio "${RATIO}" \
    --start_epoch 0 \
    --epochs 50 \
    --warmup_epochs 10 \
    --blr 5e-5 \
    --weight_decay 0.05 \
    --device cuda
fi

if [[ "${STAGE}" == "all" || "${STAGE}" == "finetune" ]]; then
  LATEST_FINETUNE="$(latest_checkpoint "${FINETUNE_DIR}")"
  PRETRAIN_CKPT="$(latest_checkpoint "${PRETRAIN_DIR}")"

  if [[ -z "${LATEST_FINETUNE}" ]]; then
    if [[ -z "${PRETRAIN_CKPT}" ]]; then
      echo "No pretrain checkpoint found in ${PRETRAIN_DIR}" >&2
      exit 1
    fi

    "${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc_per_node=1 finetune_cifar10.py \
      --data_path "${DATA}" \
      --finetune "${PRETRAIN_CKPT}" \
      --output_dir "${FINETUNE_DIR}" \
      --log_dir "${FINETUNE_DIR}" \
      --batch_size 64 \
      --accum_iter 4 \
      --model vit_base_patch16 \
      --sigma "[0,0.75]" \
      --use_quaternion_noise "${USE_QWT}" \
      --levels "${LEVELS}" \
      --ratio "${RATIO}" \
      --con_reg \
      --reg_lbd 2.0 \
      --reg_eta 0.5 \
      --epochs 50 \
      --blr 5e-4 \
      --layer_decay 0.65 \
      --weight_decay 0.05 \
      --drop_path 0.1 \
      --device cuda
  else
    "${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc_per_node=1 finetune_cifar10.py \
      --data_path "${DATA}" \
      --resume "${LATEST_FINETUNE}" \
      --output_dir "${FINETUNE_DIR}" \
      --log_dir "${FINETUNE_DIR}" \
      --batch_size 64 \
      --accum_iter 4 \
      --model vit_base_patch16 \
      --sigma "[0,0.75]" \
      --use_quaternion_noise "${USE_QWT}" \
      --levels "${LEVELS}" \
      --ratio "${RATIO}" \
      --con_reg \
      --reg_lbd 2.0 \
      --reg_eta 0.5 \
      --epochs 50 \
      --blr 5e-4 \
      --layer_decay 0.65 \
      --weight_decay 0.05 \
      --drop_path 0.1 \
      --device cuda
  fi
fi

if [[ "${STAGE}" == "all" || "${STAGE}" == "certify" ]]; then
  FINETUNE_CKPT="$(best_r0_checkpoint "${FINETUNE_DIR}")"
  if [[ -z "${FINETUNE_CKPT}" ]]; then
    FINETUNE_CKPT="$(latest_checkpoint "${FINETUNE_DIR}")"
  fi
  if [[ -z "${FINETUNE_CKPT}" ]]; then
    echo "No finetune checkpoint found in ${FINETUNE_DIR}" >&2
    exit 1
  fi

  for EVAL_SIGMA in 0.25 0.5; do
    if [[ "${EVAL_SIGMA}" == "0.25" ]]; then
      CERT_DIR="${CERT_DIR_025}"
    else
      CERT_DIR="${CERT_DIR_05}"
    fi

    "${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc_per_node=1 certify_cifar10.py \
      --resume "${FINETUNE_CKPT}" \
      --model vit_base_patch16 \
      --data_path "${DATA}" \
      --output_dir "${CERT_DIR}" \
      --sigma "${EVAL_SIGMA}" \
      --sample_interval 1 \
      --num 10000 \
      --nb_classes 10 \
      --use_quaternion_noise "${USE_QWT}" \
      --levels "${LEVELS}" \
      --ratio "${RATIO}" \
      --device cuda:0
  done
fi
