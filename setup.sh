#!/bin/bash
# setup.sh — Run this once on your GCP VM after SSH-ing in
# Usage: bash scripts/setup.sh YOUR_BUCKET_NAME

set -e
BUCKET=${1:-"your-bucket-name"}

echo "============================================"
echo "  GCP VM Setup for Sketch Diffusion"
echo "  Bucket: $BUCKET"
echo "============================================"

# 1. Install gcsfuse to mount GCS bucket
echo "[1/5] Installing gcsfuse..."
export GCSFUSE_REPO=gcsfuse-$(lsb_release -c -s)
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" \
    | sudo tee /etc/apt/sources.list.d/gcsfuse.list
sudo apt-get update -qq
sudo apt-get install -y gcsfuse

# 2. Mount GCS bucket
echo "[2/5] Mounting gs://$BUCKET → /workspace..."
sudo mkdir -p /workspace
sudo chmod 777 /workspace
gcsfuse --implicit-dirs "$BUCKET" /workspace
echo "  Mounted. Contents:"
ls /workspace | head -10

# 3. Install Python deps
echo "[3/5] Installing Python dependencies..."
pip install -r requirements.txt --quiet

# 4. Verify GPU
echo "[4/5] GPU check..."
python -c "import torch; print(f'  CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# 5. Run convert.py if captions.csv doesn't exist
if [ ! -f /workspace/captions.csv ]; then
    echo "[5/5] Running convert.py to generate captions.csv..."
    python src/convert.py
else
    echo "[5/5] captions.csv already exists, skipping convert.py"
fi

echo ""
echo "============================================"
echo "  Setup complete! Now run:"
echo "  bash scripts/train.sh $BUCKET"
echo "============================================"
