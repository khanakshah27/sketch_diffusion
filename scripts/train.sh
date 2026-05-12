#!/bin/bash
# train.sh — Launch training on Vast.ai instance
set -e

echo "============================================"
echo "  Starting Training"
echo "============================================"

export CSV_PATH="/root/captions.csv"
export IMAGE_ROOT="/root/flickr30k-images"
export OUTPUT_DIR="/root/outputs"
export NUM_IMAGES="${NUM_IMAGES:-10000}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export NUM_EPOCHS="${NUM_EPOCHS:-5}"
export LR="${LR:-1e-4}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export TIMESTEP_BIAS="${TIMESTEP_BIAS:-0.7}"

mkdir -p /root/outputs/checkpoints /root/outputs/inference

echo "Config:"
echo "  NUM_IMAGES=$NUM_IMAGES | BATCH_SIZE=$BATCH_SIZE"
echo "  NUM_EPOCHS=$NUM_EPOCHS | LR=$LR"
echo ""

nohup python /root/train.py > /root/outputs/training.log 2>&1 &
PID=$!
echo $PID > /root/outputs/train.pid
echo "Training started — PID $PID"
echo "Monitor: tail -f /root/outputs/training.log"
