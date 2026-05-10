#!/bin/bash
# train.sh — Launch training on GCP VM
# Usage: bash scripts/train.sh YOUR_BUCKET_NAME
# Optional env overrides: NUM_IMAGES, BATCH_SIZE, NUM_EPOCHS, LR

set -e
BUCKET=${1:-"your-bucket-name"}

echo "============================================"
echo "  Starting Training Run"
echo "  Bucket: $BUCKET"
echo "============================================"

# Export paths pointing at mounted bucket
export CSV_PATH="/workspace/captions.csv"
export IMAGE_ROOT="/workspace/flickr30k-images"
export OUTPUT_DIR="/workspace/outputs"

# Training config — override via environment or edit here
export NUM_IMAGES="${NUM_IMAGES:-10000}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export NUM_EPOCHS="${NUM_EPOCHS:-5}"
export LR="${LR:-1e-4}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export TIMESTEP_BIAS="${TIMESTEP_BIAS:-0.7}"

echo "Config:"
echo "  NUM_IMAGES=$NUM_IMAGES  BATCH_SIZE=$BATCH_SIZE"
echo "  NUM_EPOCHS=$NUM_EPOCHS  LR=$LR"
echo ""

# Run training in background so SSH disconnect doesn't kill it
nohup python src/train.py > /workspace/outputs/training.log 2>&1 &
PID=$!
echo "Training started with PID $PID"
echo "Monitor logs: tail -f /workspace/outputs/training.log"
echo "Or watch via: gcloud compute ssh YOUR_VM -- 'tail -f /workspace/outputs/training.log'"
echo ""
echo "PID saved to /workspace/outputs/train.pid"
echo $PID > /workspace/outputs/train.pid
