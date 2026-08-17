#!/usr/bin/env python3
"""
autorun.py -- Full automation script for Vast.ai
=================================================
Runs everything automatically:
  1. Installs gcloud if missing
  2. Authenticates with GCS bucket
  3. Downloads dataset from bucket (handles tar.gz)
  4. Installs Python dependencies
  5. Runs convert.py to generate captions.csv
  6. Starts training in background
  7. Tails the log live

Usage:
  python /autorun.py

Optional env vars to override defaults:
  BUCKET_NAME   — GCS bucket name (default: diffm_bucket1)
  PROJECT_ID    — GCP project ID (default: diffusionmodel-492408)
  NUM_EPOCHS    — number of training epochs (default: 50)
  RESUME        — set to 'yes' to auto-resume from latest checkpoint
  SKIP_DOWNLOAD — set to 'yes' if dataset already on disk
"""

import os
import sys
import subprocess
import shutil
import time
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────
BUCKET_NAME   = os.environ.get("BUCKET_NAME",   "diffm_bucket1")
PROJECT_ID    = os.environ.get("PROJECT_ID",    "diffusionmodel-492408")
WORKSPACE     = Path("/workspace")
IMAGE_DIR     = WORKSPACE / "flickr30k-images"
TOKEN_FILE    = WORKSPACE / "results_20130124.token"
CAPTIONS_CSV  = WORKSPACE / "captions.csv"
OUTPUT_DIR    = WORKSPACE / "outputs"
LOG_FILE      = OUTPUT_DIR / "training_v6.log"
TRAIN_PID     = OUTPUT_DIR / "train.pid"

NUM_EPOCHS    = os.environ.get("NUM_EPOCHS",    "50")
BATCH_SIZE    = os.environ.get("BATCH_SIZE",    "16")
LR            = os.environ.get("LR",            "3e-5")
RESUME        = os.environ.get("RESUME",        "yes")
SKIP_DOWNLOAD = os.environ.get("SKIP_DOWNLOAD", "no")

GCLOUD_PATH   = Path("/root/google-cloud-sdk/bin/gcloud")
GCLOUD_INIT   = Path("/root/google-cloud-sdk/path.bash.inc")


# ── Helpers ────────────────────────────────────────────────────────────────

def run(cmd, check=True, shell=True, capture=False):
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
    free_gb  = stat.free  / (1024**3)
    total_gb = stat.total / (1024**3)
    print(f"[DISK] Free: {free_gb:.1f}GB / Total: {total_gb:.1f}GB")
    if free_gb < 15:
        print("[WARN] Less than 15GB free. Consider clearing space.")
        print("       Run: rm -rf /workspace/outputs/checkpoints/checkpoint_epoch_*.pt")
    return free_gb


# ── Step 1: Install gcloud ─────────────────────────────────────────────────

def install_gcloud():
    section("STEP 1: Checking gcloud")

    if GCLOUD_PATH.exists():
        print(f"[OK] gcloud found at {GCLOUD_PATH}")
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

    result = run(
        f"{GCLOUD_PATH} auth list --format='value(account)'",
        check=False, capture=True
    )
    if result and "@" in result:
        print(f"[OK] Already authenticated as: {result}")
        run(f"{GCLOUD_PATH} config set project {PROJECT_ID}", check=False)
        run(f"{GCLOUD_PATH} auth application-default set-quota-project {PROJECT_ID}", check=False)
        return

    print("[INFO] Not authenticated. Starting auth flow...")
    print("\n" + "="*50)
    print("  ACTION NEEDED:")
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

    # Check if images already downloaded and extracted
    if IMAGE_DIR.exists():
        n_images = len(list(IMAGE_DIR.glob("*.jpg")))
        if n_images > 100:
            print(f"[OK] Images already present: {n_images} jpg files found")
            if TOKEN_FILE.exists():
                print(f"[OK] Token file already present: {TOKEN_FILE}")
                return

    # ── Download token file ────────────────────────────────────────────────
    if not TOKEN_FILE.exists():
        print("[INFO] Downloading caption token file...")
        success = run(
            f"{GCLOUD_PATH} storage cp "
            f"gs://{BUCKET_NAME}/results_20130124.token {WORKSPACE}/",
            check=False
        )
        if not success:
            print("[WARN] Token file not found in bucket — will try captions.csv directly later")

    # ── Download images ────────────────────────────────────────────────────
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    print("[INFO] Checking bucket for images...")

    # Show bucket contents for debugging
    bucket_ls = run(
        f"{GCLOUD_PATH} storage ls gs://{BUCKET_NAME}/",
        check=False, capture=True
    )
    print(f"[INFO] Bucket contents:\n{bucket_ls}")

    # FIX: Check for tar.gz first (this is what's in your bucket)
    tar_path  = f"gs://{BUCKET_NAME}/flickr30k-images.tar.gz"
    tar_check = run(
        f"{GCLOUD_PATH} storage ls {tar_path} 2>/dev/null",
        check=False, capture=True
    )

    if tar_check and "flickr30k-images.tar.gz" in tar_check:
        # ── tar.gz path ───────────────────────────────────────────────────
        local_tar = WORKSPACE / "flickr30k-images.tar.gz"

        if local_tar.exists():
            print(f"[INFO] tar.gz already downloaded locally. Extracting...")
        else:
            print(f"[INFO] Downloading flickr30k-images.tar.gz from bucket...")
            run(f"{GCLOUD_PATH} storage cp {tar_path} {local_tar}")

        print(f"[INFO] Extracting archive to {WORKSPACE}/ (5-10 min)...")
        run(f"tar -xzf {local_tar} -C {WORKSPACE}/")

        # Free disk space after extraction
        print(f"[INFO] Removing tar.gz to free disk space...")
        local_tar.unlink()
        print(f"[OK] tar.gz extracted and removed.")

    else:
        # ── Fallback: direct folder copy ──────────────────────────────────
        print("[INFO] No tar.gz found. Trying direct folder download...")
        found = False
        for bucket_path in [
            f"gs://{BUCKET_NAME}/flickr30k-images",
            f"gs://{BUCKET_NAME}/flickr30k-images/",
        ]:
            result = run(
                f"{GCLOUD_PATH} storage ls {bucket_path} 2>/dev/null | head -5",
                check=False, capture=True
            )
            if result and ".jpg" in result:
                print(f"[INFO] Images found at: {bucket_path}")
                run(f"{GCLOUD_PATH} storage cp -r {bucket_path} {WORKSPACE}/")
                found = True
                break

        if not found:
            print("[ERROR] Could not find images in bucket.")
            print("        Expected either:")
            print(f"          gs://{BUCKET_NAME}/flickr30k-images.tar.gz")
            print(f"          gs://{BUCKET_NAME}/flickr30k-images/*.jpg")
            print("        Full bucket listing:")
            run(f"{GCLOUD_PATH} storage ls gs://{BUCKET_NAME}/", check=False)
            sys.exit(1)

    n_images = len(list(IMAGE_DIR.glob("*.jpg")))
    print(f"[OK] {n_images} images ready at {IMAGE_DIR}")
    if n_images < 100:
        print("[WARN] Very few images found — extraction may have put them in a subfolder.")
        print(f"       Check: ls {WORKSPACE}/")
        # Try to find where they actually extracted to
        result = run(f"find {WORKSPACE} -name '*.jpg' | head -3", check=False, capture=True)
        if result:
            print(f"[INFO] Found jpg files at: {result}")


# ── Step 4: Install Python dependencies ───────────────────────────────────

def install_dependencies():
    section("STEP 4: Installing Python Dependencies")

    result = run(
        "python -c 'import diffusers; print(diffusers.__version__)'",
        check=False, capture=True
    )
    if result and result.strip():
        print(f"[OK] diffusers already installed: {result}")
        # Ensure skimage is present
        run("pip install scikit-image --quiet", check=False)
        return

    print("[INFO] Installing dependencies...")
    run("pip install 'torch>=2.0.0' 'torchvision>=0.15.0' --quiet")
    run("pip install 'diffusers==0.27.2' 'transformers==4.40.0' --quiet")
    run("pip install accelerate einops pandas Pillow --quiet")
    run("pip install opencv-python-headless scikit-image --quiet")
    run("pip install xformers --quiet", check=False)

    run("python -c \"import torch; print('PyTorch:', torch.__version__)\"")
    run(
        "python -c \"import torch; print('CUDA:', torch.cuda.is_available(), "
        "torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')\""
    )
    print("[OK] Dependencies installed.")


# ── Step 5: Run convert.py ────────────────────────────────────────────────

def run_convert():
    section("STEP 5: Generating captions.csv")

    if CAPTIONS_CSV.exists():
        n_lines = sum(1 for _ in open(CAPTIONS_CSV)) - 1
        print(f"[OK] captions.csv already exists with {n_lines} pairs")
        return

    if not TOKEN_FILE.exists():
        print(f"[INFO] No token file. Trying to download captions.csv from bucket...")
        run(
            f"{GCLOUD_PATH} storage cp "
            f"gs://{BUCKET_NAME}/captions.csv {WORKSPACE}/",
            check=False
        )
        if CAPTIONS_CSV.exists():
            print("[OK] captions.csv downloaded from bucket.")
            return
        print("[ERROR] Neither token file nor captions.csv found.")
        sys.exit(1)

    # Find convert.py
    convert_path = None
    for p in ["/convert.py", "convert.py", "src/convert.py"]:
        if os.path.exists(p):
            convert_path = p
            break
    if convert_path is None:
        print("[ERROR] convert.py not found at /convert.py")
        sys.exit(1)

    print(f"[INFO] Running {convert_path}...")
    env = {
        **os.environ,
        "CAPTION_FILE": str(TOKEN_FILE),
        "IMAGE_FOLDER":  str(IMAGE_DIR),
        "OUTPUT_CSV":    str(CAPTIONS_CSV),
    }
    result = subprocess.run(["python", convert_path], env=env)
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
        print("[ERROR] train.py not found. Upload it to /train.py")
        sys.exit(1)
    print(f"[OK] Found train.py at: {train_path}")

    env_vars = {
        "CSV_PATH":         str(CAPTIONS_CSV),
        "IMAGE_ROOT":       str(IMAGE_DIR),
        "OUTPUT_DIR":       str(OUTPUT_DIR),
        "NUM_IMAGES":       "10000",
        "BATCH_SIZE":       BATCH_SIZE,
        "NUM_EPOCHS":       NUM_EPOCHS,
        "LR":               LR,
        "GRAD_ACCUM_STEPS": "2",
        "NUM_WORKERS":      "4",
        "TIMESTEP_BIAS":    "0.8",
        "EMA_DECAY":        "0.9995",
        "SNR_GAMMA":        "5.0",
        "VAL_EVERY":        "3",
        "VAL_IMAGES":       "32",
        "VAL_STEPS":        "20",
        "EARLY_STOP_PAT":   "5",
        "RESUME_CKPT":      "auto" if RESUME.lower() == "yes" else "",
    }

    print("\n[CONFIG] Training configuration:")
    for k, v in env_vars.items():
        print(f"  {k:<22} = {v}")

    # Kill any existing training
    run("pkill -f 'python.*train.py' 2>/dev/null || true", check=False)
    time.sleep(2)

    env_export = " ".join(f'{k}="{v}"' for k, v in env_vars.items())
    cmd = (
        f"env {env_export} "
        f"nohup python {train_path} "
        f"> {LOG_FILE} 2>&1 & "
        f"echo $! > {TRAIN_PID} && "
        f"echo 'Training PID:' $(cat {TRAIN_PID})"
    )

    print(f"\n[INFO] Launching training in background...")
    run(cmd)
    time.sleep(3)

    if TRAIN_PID.exists():
        pid = open(TRAIN_PID).read().strip()
        alive = run(f"ps -p {pid} > /dev/null 2>&1", check=False)
        if alive:
            print(f"[OK] Training running (PID: {pid})")
        else:
            print(f"[WARN] Process {pid} not found. Check log for errors.")

    print(f"\n[INFO] Log: {LOG_FILE}")


# ── Step 7: Tail log ──────────────────────────────────────────────────────

def tail_log():
    section("STEP 7: Live Training Log")
    print(f"[INFO] Watching {LOG_FILE}")
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
        print(f"[ERROR] Log not created. Check: cat {LOG_FILE}")
        return

    try:
        run(f"tail -f {LOG_FILE}")
    except KeyboardInterrupt:
        print(f"\n\n[INFO] Stopped watching. Training continues in background.")
        print(f"[INFO] Watch again:    tail -f {LOG_FILE}")
        print(f"[INFO] Check running:  ps aux | grep train.py")
        print(f"\n[INFO] When done, download results:")
        print(f"  scp -P PORT root@ssh.vast.ai:{OUTPUT_DIR}/inference/generated_epoch*.png .")
        print(f"  scp -P PORT root@ssh.vast.ai:{OUTPUT_DIR}/training_v6.log .")
        print(f"  scp -P PORT root@ssh.vast.ai:{CKPT_DIR}/checkpoint_best.pt .")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  SKETCH DIFFUSION v6 — Full Automation Script")
    print(f"  Bucket:  gs://{BUCKET_NAME}")
    print(f"  Project: {PROJECT_ID}")
    print(f"  Epochs:  {NUM_EPOCHS} | LR: {LR} | Resume: {RESUME}")
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
