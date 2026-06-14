#!/bin/bash
# train.sh — Launch training on Vast.ai
# Targets: SSIM>0.6 | PSNR>18dB | MSE<0.1 | 50 epochs | convergence detection
set -e

echo "============================================"
echo "  Sketch Diffusion v3 — Target Training"
echo "  SSIM>0.6 | PSNR>18 | MSE<0.1 | 50 epochs"
echo "============================================"

export CSV_PATH="/workspace/captions.csv"
export IMAGE_ROOT="/workspace/flickr30k-images"
export OUTPUT_DIR="/workspace/outputs"
export NUM_IMAGES="${NUM_IMAGES:-10000}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export NUM_EPOCHS="${NUM_EPOCHS:-50}"
export LR="${LR:-5e-5}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export TIMESTEP_BIAS="${TIMESTEP_BIAS:-0.8}"
export EMA_DECAY="${EMA_DECAY:-0.9995}"
export SNR_GAMMA="${SNR_GAMMA:-5.0}"
export EARLY_STOP_PAT="${EARLY_STOP_PAT:-8}"
export RESUME_CKPT="${RESUME_CKPT:-}"

mkdir -p /workspace/outputs/checkpoints /workspace/outputs/inference

echo "Config:"
echo "  NUM_IMAGES=$NUM_IMAGES | BATCH_SIZE=$BATCH_SIZE"
echo "  NUM_EPOCHS=$NUM_EPOCHS | LR=$LR | GRAD_ACCUM=$GRAD_ACCUM_STEPS"
echo "  EMA_DECAY=$EMA_DECAY | SNR_GAMMA=$SNR_GAMMA"
echo "  RESUME_CKPT=${RESUME_CKPT:-none}"
echo ""

nohup python /train.py > /workspace/outputs/training_v3.log 2>&1 &
PID=$!
echo $PID > /workspace/outputs/train.pid
echo "Training started — PID $PID"
echo "Monitor: tail -f /workspace/outputs/training_v3.log"
