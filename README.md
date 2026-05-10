# Sketch-to-Image Diffusion — GCP Deployment Guide

## Repo Structure

```
sketch_diffusion/
├── src/
│   ├── train.py       ← main training script
│   ├── convert.py     ← Flickr30k caption preprocessor
│   └── inference.py   ← standalone image generation
├── scripts/
│   ├── setup.sh       ← one-time VM setup
│   └── train.sh       ← training launcher
├── requirements.txt
└── README.md
```

---

## Prerequisites (you already have these)

- Google Cloud project with billing enabled ✅
- Flickr30k images uploaded to a GCS bucket ✅
- Vertex AI API enabled ✅

---

## Step 1 — Upload this repo to GCS

On your local machine, run:

```bash
# Replace with your actual bucket name everywhere below
BUCKET="your-bucket-name"

# Upload the code
gsutil -m cp -r sketch_diffusion/ gs://$BUCKET/code/

# Upload the captions token file if not already there
gsutil cp results_20130124.token gs://$BUCKET/
```

---

## Step 2 — Create a GPU VM on GCP

Go to **Google Cloud Console → Compute Engine → VM Instances → Create Instance**

Or use this gcloud command (run in Cloud Shell):

```bash
gcloud compute instances create sketch-diffusion-vm \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-a100,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True"
```

**Why these specs:**
- `n1-standard-8` = 8 vCPUs, 30GB RAM — handles large DataLoader workers
- `nvidia-tesla-a100` = 40GB VRAM — needed for SD1.5 + ControlNet in float32
- `200GB` disk = model weights (~15GB) + dataset + outputs
- `pytorch-latest-gpu` image = PyTorch pre-installed, no CUDA setup needed

**Cost: ~$3.50/hour. A full 5-epoch run = ~$10-15 total.**

---

## Step 3 — SSH into the VM

```bash
gcloud compute ssh sketch-diffusion-vm --zone=us-central1-a
```

---

## Step 4 — Download and set up the code

Once inside the VM terminal:

```bash
# Download code from GCS
gsutil -m cp -r gs://your-bucket-name/code/sketch_diffusion/ .
cd sketch_diffusion

# Run setup (installs gcsfuse, mounts bucket, installs deps)
bash scripts/setup.sh your-bucket-name
```

This does everything automatically:
- Installs gcsfuse
- Mounts your GCS bucket to `/workspace/` so all your Flickr images are accessible
- Installs all Python dependencies
- Runs `convert.py` to generate `captions.csv` from your token file

After setup, verify the mount:
```bash
ls /workspace/flickr30k-images/ | head -5
# Should show .jpg files like: 1000092795.jpg
cat /workspace/captions.csv | head -3
# Should show: image,caption rows
```

---

## Step 5 — Start training

```bash
bash scripts/train.sh your-bucket-name
```

Training runs in the **background** so if your SSH drops, it keeps going.

Monitor progress:
```bash
tail -f /workspace/outputs/training.log
```

You'll see output like:
```
  Ep 1/5 | Loss 0.1243 | MSE 0.1243 | PSNR 18.12dB | SSIM 0.4821 | LR 1.00e-04 | 847s
  Ep 2/5 | Loss 0.0981 | MSE 0.0981 | PSNR 20.11dB | SSIM 0.5234 | LR 8.50e-05 | 1694s
  ...
```

**Expected results:**
- Epoch 1: MSE ~0.12-0.14
- Epoch 3: MSE ~0.09-0.10
- Epoch 5: MSE < 0.09

Checkpoints save automatically after each epoch to `/workspace/outputs/checkpoints/`.

---

## Step 6 — Run inference (generate an image)

After training finishes:

```bash
python src/inference.py \
  --checkpoint /workspace/outputs/checkpoints/checkpoint_epoch_5.pt \
  --image_path /workspace/flickr30k-images/1000092795.jpg \
  --prompt "a man in a blue shirt standing near a tree" \
  --output /workspace/outputs/inference/result.png
```

Download the result to your local machine:
```bash
# From your LOCAL terminal (not VM):
gcloud compute scp sketch-diffusion-vm:/workspace/outputs/inference/result.png ./result.png --zone=us-central1-a
```

---

## Step 7 — Stop the VM when done

**IMPORTANT: Always stop the VM when not using it or you'll keep getting charged.**

```bash
# From Cloud Shell or local terminal:
gcloud compute instances stop sketch-diffusion-vm --zone=us-central1-a
```

---

## Answers to your questions

**Q: Which code to use — Kaggle notebook or this repo?**
Use this repo for GCP. The Kaggle notebook was a single-file PoC. This is the proper project structure with separate files for training, preprocessing, and inference.

**Q: How does Flickr30k get incorporated?**
Your images are already in GCS. `gcsfuse` mounts the bucket as a local folder — the training code reads images from `/workspace/flickr30k-images/` exactly like a normal directory. Zero changes needed to code paths.

**Q: Will GCP handle this?**
Yes. The A100 (40GB VRAM) handles SD1.5 + ControlNet in float32 comfortably. With 10,000 images and batch size 8, each epoch takes ~25-30 minutes. Full 5-epoch run = ~2.5 hours = ~$8-10.

---

## Troubleshooting

**"No space left on device"**
```bash
df -h  # check disk usage
# If full, delete HuggingFace cache:
rm -rf ~/.cache/huggingface/hub/models--runwayml*
```

**"CUDA out of memory"**
Reduce batch size: edit `scripts/train.sh` and set `BATCH_SIZE=4`

**"gcsfuse: permission denied"**
```bash
gcloud auth application-default login
```

**Training stopped (SSH dropped)**
```bash
# Check if still running:
cat /workspace/outputs/train.pid
ps aux | grep train.py

# If stopped, restart:
bash scripts/train.sh your-bucket-name
# It will pick up from the last checkpoint automatically if you add resume logic
```
