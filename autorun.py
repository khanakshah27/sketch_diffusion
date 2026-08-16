#!/usr/bin/env python3
"""
autorun.py -- Full automation script for Vast.ai
=================================================
Runs everything automatically:
  1. Installs gcloud if missing
  2. Authenticates with GCS bucket
  3. Downloads dataset from bucket
  4. Installs Python dependencies
  5. Runs convert.py to generate captions.csv
  6. Starts training in background
  7. Tails the log live

Usage:
  python autorun.py

Optional env vars to override defaults:
  BUCKET_NAME   — GCS bucket name (default: diffm_bucket1)
  PROJECT_ID    — GCP project ID (default: diffusionmodel-492408)
  NUM_EPOCHS    — number of training epochs (default: 50)
  RESUME        — set to 'yes' to auto-resume from latest checkpoint
  SKIP_DOWNLOAD — set to 'yes' if dataset already on disk

Just run: python autorun.py
And it handles everything else.
"""

import os
import sys
import subprocess
import shutil
import time
from pathlib import Path

# ── Config — change these if needed ───────────────────────────────────────
BUCKET_NAME   = os.environ.get("BUCKET_NAME",   "diffm_bucket1")
PROJECT_ID    = os.environ.get("PROJECT_ID",    "diffusionmodel-492408")
WORKSPACE     = Path("/workspace")
IMAGE_DIR     = WORKSPACE / "flickr30k-images"
TOKEN_FILE    = WORKSPACE / "results_20130124.token"
CAPTIONS_CSV  = WORKSPACE / "captions.csv"
OUTPUT_DIR    = WORKSPACE / "outputs"
LOG_FILE      = OUTPUT_DIR / "training_v6.log"
TRAIN_PID     = OUTPUT_DIR / "train.pid"

NUM_EPOCHS    = os.environ.get("NUM_EPOCHS",  "50")
BATCH_SIZE    = os.environ.get("BATCH_SIZE",  "16")
LR            = os.environ.get("LR",          "3e-5")
RESUME        = os.environ.get("RESUME",      "yes")   # auto-resume by default
SKIP_DOWNLOAD = os.environ.get("SKIP_DOWNLOAD", "no")

GCLOUD_PATH   = Path("/root/google-cloud-sdk/bin/gcloud")
GCLOUD_INIT   = Path("/root/google-cloud-sdk/path.bash.inc")

# ── Helpers ────────────────────────────────────────────────────────────────

def run(cmd, check=True, shell=True, capture=False):
    """Run a shell command, print it, handle errors."""
    print(f"\n>>> {cmd}")
    if capture:
        result = subprocess.run(cmd, shell=shell, capture_output=True, text=True)
        return result.stdout.strip()
    result = subprocess.run(cmd, shell=shell)
    if check and result.returncode != 0:
        print(f"[ERROR] Command failed with code {result.returncode}")
        sys.exit(1)
    return result.returncode == 0


def gcloud(cmd, check=True, capture=False):
    """Run a gcloud command using the correct path."""
    if GCLOUD_PATH.exists():
        return run(f"{GCLOUD_PATH} {cmd}", check=check, capture=capture)
    else:
        return run(f"gcloud {cmd}", check=check, capture=capture)


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check_disk():
    stat = shutil.disk_usage("/workspace")
    free_gb = stat.free / (1024**3)
    total_gb = stat.total / (1024**3)
    print(f"[DISK] Free: {free_gb:.1f}GB / Total: {total_gb:.1f}GB")
    if free_gb < 15:
        print("[WARN] Less than 15GB free. Consider clearing space.")
        print("       Run: rm -rf /workspace/outputs/checkpoints/checkpoint_epoch_*.pt")
    return free_gb


# ── Step 1: Install gcloud if not present ─────────────────────────────────

def install_gcloud():
    section("STEP 1: Checking gcloud")

    if GCLOUD_PATH.exists():
        print(f"[OK] gcloud found at {GCLOUD_PATH}")
        # Source the path
        run(f"source {GCLOUD_INIT} 2>/dev/null || true", check=False)
        return

    print("[INFO] gcloud not found. Installing...")
    run("curl https://sdk.cloud.google.com | bash -s -- --disable-prompts --install-dir=/root")
    if not GCLOUD_PATH.exists():
        print("[ERROR] gcloud installation failed.")
        sys.exit(1)
    print("[OK] gcloud installed.")


# ── Step 2: Authenticate GCP ──────────────────────────────────────────────

def authenticate_gcp():
    section("STEP 2: GCP Authentication")

    # Check if already authenticated
    result = run(f"{GCLOUD_PATH} auth list --format='value(account)'",
                 check=False, capture=True)
    if result and "@" in result:
        print(f"[OK] Already authenticated as: {result}")
        run(f"{GCLOUD_PATH} config set project {PROJECT_ID}", check=False)
        run(f"{GCLOUD_PATH} auth application-default set-quota-project {PROJECT_ID}", check=False)
        return

    print("[INFO] Not authenticated. Starting auth flow...")
    print("\n" + "="*50)
    print("  ACTION NEEDED: Follow the prompts below")
    print("  1. Click the URL that appears")
    print("  2. Sign in with your Google account")
    print("  3. Copy the verification code")
    print("  4. Paste it here and press Enter")
    print("="*50 + "\n")

    run(f"{GCLOUD_PATH} auth login --no-launch-browser")
    run(f"{GCLOUD_PATH} auth application-default login --no-launch-browser")
    run(f"{GCLOUD_PATH} config set project {PROJECT_ID}")
    run(f"{GCLOUD_PATH} auth application-default set-quota-project {PROJECT_ID}")
    print("[OK] Authentication complete.")


# ── Step 3: Download dataset ───────────────────────────────────────────────

def download_dataset():
    section("STEP 3: Dataset Download")

    if SKIP_DOWNLOAD.lower() == "yes":
        print("[SKIP] SKIP_DOWNLOAD=yes — skipping dataset download")
        return

    # Check if already downloaded
    if IMAGE_DIR.exists():
        n_images = len(list(IMAGE_DIR.glob("*.jpg")))
        if n_images > 100:
            print(f"[OK] Images already present: {n_images} jpg files found")
            if TOKEN_FILE.exists():
                print(f"[OK] Token file already present: {TOKEN_FILE}")
                return
        else:
            print(f"[WARN] Only {n_images} images found — re-downloading")

    # List bucket contents first
    print(f"[INFO] Checking bucket gs://{BUCKET_NAME}/...")
    bucket_contents = run(
        f"{GCLOUD_PATH} storage ls gs://{BUCKET_NAME}/",
        check=False, capture=True
    )
    print(f"[INFO] Bucket contents:\n{bucket_contents}")

    # Download token file
    if not TOKEN_FILE.exists():
        print(f"[INFO] Downloading token file...")
        success = run(
            f"{GCLOUD_PATH} storage cp gs://{BUCKET_NAME}/results_20130124.token {WORKSPACE}/",
            check=False
        )
        if not success or not TOKEN_FILE.exists():
            print("[WARN] Token file not in bucket. Checking for captions.csv directly...")
            run(
                f"{GCLOUD_PATH} storage cp gs://{BUCKET_NAME}/captions.csv {WORKSPACE}/",
                check=False
            )

    # Download images — check bucket path
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading Flickr30k images (this takes 20-40 min)...")

    # Try different possible bucket structures
    for bucket_path in [
        f"gs://{BUCKET_NAME}/flickr30k-images",
        f"gs://{BUCKET_NAME}/flickr30k-images/",
        f"gs://{BUCKET_NAME}/"
    ]:
        result = run(
            f"{GCLOUD_PATH} storage ls {bucket_path} 2>/dev/null | head -5",
            check=False, capture=True
        )
        if result and ".jpg" in result:
            print(f"[INFO] Images found at: {bucket_path}")
            run(f"{GCLOUD_PATH} storage cp -r {bucket_path} {WORKSPACE}/")
            break
    else:
        print("[ERROR] Could not find images in bucket.")
        print("        Please upload them manually or check BUCKET_NAME.")
        sys.exit(1)

    n_images = len(list(IMAGE_DIR.glob("*.jpg")))
    print(f"[OK] Downloaded {n_images} images to {IMAGE_DIR}")


# ── Step 4: Install Python dependencies ───────────────────────────────────

def install_dependencies():
    section("STEP 4: Installing Python Dependencies")

    # Check if already installed
    result = run("python -c 'import diffusers; print(diffusers.__version__)'",
                 check=False, capture=True)
    if result and result.strip():
        print(f"[OK] diffusers already installed: {result}")
        # Still install skimage if missing
        run("pip install scikit-image --quiet", check=False)
        return

    print("[INFO] Installing dependencies...")
    run("pip install torch>=2.0.0 torchvision>=0.15.0 --quiet")
    run("pip install diffusers==0.27.2 transformers==4.40.0 --quiet")
    run("pip install accelerate einops pandas Pillow --quiet")
    run("pip install opencv-python-headless scikit-image --quiet")
    run("pip install xformers --quiet", check=False)  # optional

    # Verify
    run("python -c \"import torch; print('PyTorch:', torch.__version__)\"")
    run("python -c \"import torch; print('CUDA:', torch.cuda.is_available(), "
        "torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')\"")
    print("[OK] Dependencies installed.")


# ── Step 5: Run convert.py ────────────────────────────────────────────────

def run_convert():
    section("STEP 5: Generating captions.csv")

    if CAPTIONS_CSV.exists():
        n_lines = sum(1 for _ in open(CAPTIONS_CSV)) - 1
        print(f"[OK] captions.csv already exists with {n_lines} pairs")
        return

    if not TOKEN_FILE.exists():
        print(f"[SKIP] No token file found at {TOKEN_FILE}")
        print("       If captions.csv is already in the bucket, downloading it...")
        run(
            f"{GCLOUD_PATH} storage cp gs://{BUCKET_NAME}/captions.csv {WORKSPACE}/",
            check=False
        )
        if CAPTIONS_CSV.exists():
            print("[OK] captions.csv downloaded from bucket.")
            return
        print("[ERROR] Neither token file nor captions.csv found.")
        sys.exit(1)

    print("[INFO] Running convert.py...")
    env = {
        **os.environ,
        "CAPTION_FILE": str(TOKEN_FILE),
        "IMAGE_FOLDER":  str(IMAGE_DIR),
        "OUTPUT_CSV":    str(CAPTIONS_CSV),
    }

    # Find convert.py
    for path in ["/train.py", "/convert.py", "convert.py"]:
        if os.path.exists(path.replace("train", "convert")):
            convert_path = path.replace("train", "convert")
            break
    else:
        convert_path = "/convert.py"

    result = subprocess.run(
        ["python", convert_path],
        env=env
    )
    if result.returncode != 0 or not CAPTIONS_CSV.exists():
        print("[ERROR] convert.py failed.")
        sys.exit(1)

    n_lines = sum(1 for _ in open(CAPTIONS_CSV)) - 1
    print(f"[OK] captions.csv created with {n_lines} caption-image pairs")


# ── Step 6: Start training ────────────────────────────────────────────────

def start_training():
    section("STEP 6: Starting Training")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "inference").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "validation").mkdir(parents=True, exist_ok=True)

    check_disk()

    # Find train.py
    train_path = None
    for p in ["/train.py", "train.py", "src/train.py"]:
        if os.path.exists(p):
            train_path = p
            break
    if train_path is None:
        print("[ERROR] train.py not found. Upload it to /train.py on the instance.")
        sys.exit(1)
    print(f"[OK] Found train.py at: {train_path}")

    # Build environment
    env_vars = {
        "CSV_PATH":        str(CAPTIONS_CSV),
        "IMAGE_ROOT":      str(IMAGE_DIR),
        "OUTPUT_DIR":      str(OUTPUT_DIR),
        "NUM_IMAGES":      "10000",
        "BATCH_SIZE":      BATCH_SIZE,
        "NUM_EPOCHS":      NUM_EPOCHS,
        "LR":              LR,
        "GRAD_ACCUM_STEPS":"2",
        "NUM_WORKERS":     "4",
        "TIMESTEP_BIAS":   "0.8",
        "EMA_DECAY":       "0.9995",
        "SNR_GAMMA":       "5.0",
        "VAL_EVERY":       "3",
        "VAL_IMAGES":      "32",
        "VAL_STEPS":       "20",
        "EARLY_STOP_PAT":  "5",
        "RESUME_CKPT":     "auto" if RESUME.lower() == "yes" else "",
    }

    env_export = " ".join(f"{k}={v}" for k, v in env_vars.items())

    print("\n[CONFIG] Training configuration:")
    for k, v in env_vars.items():
        print(f"  {k:<22} = {v}")

    # Kill any existing training process
    run("pkill -f 'python.*train.py' 2>/dev/null || true", check=False)
    time.sleep(2)

    # Build nohup command
    cmd = (
        f"env {env_export} "
        f"nohup python {train_path} "
        f"> {LOG_FILE} 2>&1 & "
        f"echo $! > {TRAIN_PID} && "
        f"echo 'Training started with PID:' $(cat {TRAIN_PID})"
    )

    print(f"\n[INFO] Launching training in background...")
    run(cmd)

    time.sleep(3)

    # Verify it started
    if TRAIN_PID.exists():
        pid = open(TRAIN_PID).read().strip()
        check = run(f"ps -p {pid} > /dev/null 2>&1", check=False)
        if check:
            print(f"[OK] Training process running (PID: {pid})")
        else:
            print(f"[WARN] Process {pid} not found. Checking log...")

    print(f"\n[INFO] Log file: {LOG_FILE}")


# ── Step 7: Tail log ──────────────────────────────────────────────────────

def tail_log():
    section("STEP 7: Live Training Log")
    print(f"[INFO] Tailing {LOG_FILE}")
    print("[INFO] Press Ctrl+C to stop watching (training continues in background)")
    print("-" * 60)

    # Wait for log to appear
    for _ in range(30):
        if LOG_FILE.exists() and LOG_FILE.stat().st_size > 0:
            break
        print(".", end="", flush=True)
        time.sleep(1)
    print()

    if not LOG_FILE.exists():
        print("[ERROR] Log file not created. Training may have failed.")
        print(f"        Check: cat {OUTPUT_DIR}/training_v6.log")
        return

    try:
        run(f"tail -f {LOG_FILE}")
    except KeyboardInterrupt:
        print(f"\n\n[INFO] Stopped watching log. Training continues in background.")
        print(f"[INFO] To watch again: tail -f {LOG_FILE}")
        print(f"[INFO] To check status: ps aux | grep train.py")
        print(f"[INFO] To download results when done:")
        print(f"         scp -P PORT root@ssh.vast.ai:{OUTPUT_DIR}/inference/generated_epoch*.png .")
        print(f"         scp -P PORT root@ssh.vast.ai:{OUTPUT_DIR}/training_v6.log .")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  SKETCH DIFFUSION — Full Automation Script")
    print(f"  Bucket: gs://{BUCKET_NAME}")
    print(f"  Project: {PROJECT_ID}")
    print(f"  Epochs: {NUM_EPOCHS} | LR: {LR} | Resume: {RESUME}")
    print("=" * 60)

    install_gcloud()
    authenticate_gcp()
    download_dataset()
    install_dependencies()
    run_convert()
    start_training()
    tail_log()


if __name__ == "__main__":
    main()
