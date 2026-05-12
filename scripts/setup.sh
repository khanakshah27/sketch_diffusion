#!/bin/bash
# setup.sh — Run once on your Vast.ai instance after SSH-ing in
set -e

echo "============================================"
echo "  Vast.ai Instance Setup"
echo "============================================"

# 1. Install dependencies
echo "[1/3] Installing Python dependencies..."
pip install -r /root/requirements.txt --quiet

# 2. Verify GPU
echo "[2/3] GPU check..."
python -c "import torch; print(f'  CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0)}')"

# 3. Run convert.py if captions.csv doesn't exist
if [ ! -f /root/captions.csv ]; then
    echo "[3/3] Generating captions.csv..."
    cd /root && python convert.py
else
    echo "[3/3] captions.csv already exists, skipping."
fi

echo ""
echo "============================================"
echo "  Setup complete! Now run:"
echo "  bash /root/train.sh"
echo "============================================"
