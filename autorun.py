#!/usr/bin/env python3
"""
autorun.py -- Full automation script for Vast.ai
=================================================
Runs everything automatically:
  1. Installs gcloud if missing
  2. Authenticates with GCS bucket
  3. Downloads dataset from bucket (handles tar.gz)
  4. Installs Python dependencies (with huggingface_hub version fix)
  5. Runs convert.py to generate captions.csv
  6. Starts training in background
  7. Tails the log live

Usage:
  python /autorun.py                           # 10k dataset, full run
  DATASET_MODE=30k python /autorun.py          # 30k dataset, full run
  ABLATION_MODE=batch python /autorun.py       # batch size ablation on 10k
  ABLATION_MODE=batch DATASET_MODE=30k python /autorun.py  # ablation on 30k
  RESUME=yes python /autorun.py                # resume from latest checkpoint
  SKIP_DOWNLOAD=yes python /autorun.py         # skip dataset download

Ablation mode runs batch sizes [4, 8, 16] for 10 epochs each,
saves separate logs, then prints a comparison table at the end.
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
TRAIN_PID     = OUTPUT_DIR / "train.pid"

DATASET_MODE  = os.environ.get("DATASET_MODE",  "10k")
ABLATION_MODE = os.environ.get("ABLATION_MODE", "")    # set to 'batch' for ablation
RESUME        = os.environ.get("RESUME",        "no")
SKIP_DOWNLOAD = os.environ.get("SKIP_DOWNLOAD", "no")

GCLOUD_PATH   = Path("/root/google-cloud-sdk/bin/gcloud")
GCLOUD_INIT   = Path("/root/google-cloud-sdk/path.bash.inc")

# ── GitHub repo — code files downloaded automatically ─────────────────────
GITHUB_RAW    = "https://raw.githubusercontent.com/khanakshah27/sketch_diffusion/main"
GITHUB_FILES  = {
    "/train.py":     f"{GITHUB_RAW}/src/train.py",
    "/convert.py":   f"{GITHUB_RAW}/src/convert.py",
    "/inference.py": f"{GITHUB_RAW}/src/inference.py",
}

# ── Per-dataset config ─────────────────────────────────────────────────────
if DATASET_MODE == "30k":
    LOG_FILE    = OUTPUT_DIR / "training_30k.log"
    NUM_IMAGES  = "31783"
    BATCH_SIZE  = "16"
    NUM_EPOCHS  = "15"
    LR          = "2e-5"
    GRAD_ACCUM  = "4"
    SNR_GAMMA   = "5.0"
    DESCRIPTION = "30k Flickr — mid_block + down_blocks.3 (~28M params)"
    ABLATION_BATCH_SIZES = ["4", "8", "16"]
    ABLATION_EPOCHS      = "10"
else:
    LOG_FILE    = OUTPUT_DIR / "training_10k.log"
    NUM_IMAGES  = "10000"
    BATCH_SIZE  = "8"
    NUM_EPOCHS  = "25"
    LR          = "1e-5"
    GRAD_ACCUM  = "4"
    SNR_GAMMA   = "3.0"
    DESCRIPTION = "10k Flickr — mid_block only (~5M params)"
    ABLATION_BATCH_SIZES = ["4", "8", "16"]
    ABLATION_EPOCHS      = "10"


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


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check_disk():
    stat     = shutil.disk_usage("/workspace")
    free_gb  = stat.free  / (1024**3)
    total_gb = stat.total / (1024**3)
    print(f"[DISK] Free: {free_gb:.1f}GB / Total: {total_gb:.1f}GB")
    if free_gb < 15:
        print("[WARN] Less than 15GB free.")
    return free_gb


def extract_final_metrics(log_path: Path) -> dict:
    """Parse the last epoch metrics from a training log."""
    metrics = {
        "noise_MSE": "N/A", "noise_MAE": "N/A",
        "img_MSE": "N/A", "img_PSNR": "N/A", "img_SSIM": "N/A"
    }
    if not log_path.exists():
        return metrics
    try:
        lines = log_path.read_text().splitlines()
        for line in reversed(lines):
            if "nMSE=" in line and "nMAE=" in line:
                # Parse: Ep X/Y | nMSE=0.8500 | nMAE=0.7200 | ...
                parts = line.split("|")
                for p in parts:
                    p = p.strip()
                    if p.startswith("nMSE="):
                        metrics["noise_MSE"] = p.replace("nMSE=", "").strip()
                    elif p.startswith("nMAE="):
                        metrics["noise_MAE"] = p.replace("nMAE=", "").strip()
                    elif p.startswith("iMSE="):
                        metrics["img_MSE"] = p.replace("iMSE=", "").strip()
                    elif p.startswith("iPSNR="):
                        metrics["img_PSNR"] = p.replace("iPSNR=", "").strip()
                    elif p.startswith("iSSIM="):
                        metrics["img_SSIM"] = p.replace("iSSIM=", "").strip()
                break
    except Exception:
        pass
    return metrics


def print_ablation_table(results: list):
    """Print a formatted comparison table of batch size ablation results."""
    W = 80
    print("\n" + "=" * W)
    print(f"{'BATCH SIZE ABLATION RESULTS — ' + DATASET_MODE + ' Dataset':^{W}}")
    print(f"{'Loss: SNR-weighted Huber | Epochs: ' + ABLATION_EPOCHS + ' | LR: ' + LR:^{W}}")
    print("=" * W)
    print(f"\n{'Batch':>8} {'Eff.Batch':>10} {'Noise MSE':>12} {'Noise MAE':>12} "
          f"{'img MSE':>10} {'img PSNR':>10} {'img SSIM':>10}")
    print("-" * W)
    for r in results:
        eff = int(r["batch"]) * int(GRAD_ACCUM)
        print(f"  {r['batch']:>6} {str(eff):>10} "
              f"{r['metrics']['noise_MSE']:>12} "
              f"{r['metrics']['noise_MAE']:>12} "
              f"{r['metrics']['img_MSE']:>10} "
              f"{r['metrics']['img_PSNR']:>10} "
              f"{r['metrics']['img_SSIM']:>10}")
    print("\n" + "=" * W)
    print("  Best batch size = lowest img_MSE / highest img_PSNR / highest img_SSIM")
    print(f"  Selected for final full run: batch size {BATCH_SIZE} "
          f"(effective {int(BATCH_SIZE)*int(GRAD_ACCUM)})")
    print("=" * W)


# ── Step 0: Download code files from GitHub ───────────────────────────────

def download_code_from_github():
    section("STEP 0: Downloading code files from GitHub")
    print(f"[INFO] Repo: https://github.com/khanakshah27/sketch_diffusion")

    # Check if wget or curl is available
    has_wget = run("which wget > /dev/null 2>&1", check=False)

    all_ok = True
    for dest_path, url in GITHUB_FILES.items():
        print(f"\n[INFO] Downloading {dest_path} ...")
        if has_wget:
            ok = run(f"wget -q -O {dest_path} '{url}'", check=False)
        else:
            ok = run(f"curl -sL -o {dest_path} '{url}'", check=False)

        if ok and Path(dest_path).exists() and Path(dest_path).stat().st_size > 100:
            size_kb = Path(dest_path).stat().st_size / 1024
            print(f"[OK] {dest_path} ({size_kb:.0f}KB)")
        else:
            print(f"[WARN] Failed to download {dest_path} from GitHub")
            print(f"       URL tried: {url}")
            print(f"       Please upload manually via Jupyter file browser")
            all_ok = False

    if all_ok:
        print(f"\n[OK] All code files downloaded from GitHub successfully")
        print(f"     train.py, convert.py, inference.py ready at /")
    else:
        print(f"\n[WARN] Some files failed. Check your repo is public and")
        print(f"       paths are correct: src/train.py, src/convert.py, src/inference.py")


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
        run(f"{GCLOUD_PATH} auth application-default set-quota-project {PROJECT_ID}",
            check=False)
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

    if IMAGE_DIR.exists():
        n_images = len(list(IMAGE_DIR.glob("*.jpg")))
        if n_images > 100:
            print(f"[OK] Images already present: {n_images} jpg files")
            if TOKEN_FILE.exists():
                print(f"[OK] Token file already present")
                return

    if not TOKEN_FILE.exists():
        print("[INFO] Downloading caption token file...")
        run(
            f"{GCLOUD_PATH} storage cp "
            f"gs://{BUCKET_NAME}/results_20130124.token {WORKSPACE}/",
            check=False
        )

    print(f"[INFO] Bucket contents:")
    run(f"{GCLOUD_PATH} storage ls gs://{BUCKET_NAME}/", check=False)

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    tar_path  = f"gs://{BUCKET_NAME}/flickr30k-images.tar.gz"
    tar_check = run(
        f"{GCLOUD_PATH} storage ls {tar_path} 2>/dev/null",
        check=False, capture=True
    )

    if tar_check and "flickr30k-images.tar.gz" in tar_check:
        local_tar = WORKSPACE / "flickr30k-images.tar.gz"
        if local_tar.exists():
            print(f"[INFO] tar.gz already on disk. Extracting...")
        else:
            print(f"[INFO] Downloading flickr30k-images.tar.gz...")
            run(f"{GCLOUD_PATH} storage cp {tar_path} {local_tar}")

        print(f"[INFO] Extracting (5-10 min)...")
        run(f"tar -xzf {local_tar} -C {WORKSPACE}/")
        local_tar.unlink()
        print(f"[OK] Extraction complete.")
    else:
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
                run(f"{GCLOUD_PATH} storage cp -r {bucket_path} {WORKSPACE}/")
                found = True
                break
        if not found:
            print("[ERROR] Could not find images in bucket.")
            sys.exit(1)

    n_images = len(list(IMAGE_DIR.glob("*.jpg")))
    print(f"[OK] {n_images} images ready at {IMAGE_DIR}")


# ── Step 4: Install dependencies ──────────────────────────────────────────

def install_dependencies():
    section("STEP 4: Installing Python Dependencies")

    print("[INFO] Pinning huggingface_hub==0.23.2 (fixes diffusers ImportError)...")
    run("pip install 'huggingface_hub==0.23.2' --quiet --force-reinstall")

    result = run(
        "python -c 'import diffusers; print(diffusers.__version__)'",
        check=False, capture=True
    )
    if result and result.strip():
        print(f"[OK] diffusers working: {result}")
        run("pip install scikit-image --quiet", check=False)
        return

    print("[INFO] Installing all dependencies...")
    run("pip install 'torch>=2.0.0' 'torchvision>=0.15.0' --quiet")
    run("pip install 'diffusers==0.27.2' 'transformers==4.40.0' "
        "'huggingface_hub==0.23.2' --quiet")
    run("pip install accelerate einops pandas Pillow --quiet")
    run("pip install opencv-python-headless scikit-image --quiet")
    run("pip install xformers --quiet", check=False)
    run("python -c \"import torch; print('PyTorch:', torch.__version__)\"")
    run("python -c \"import diffusers; print('diffusers:', diffusers.__version__)\"")
    print("[OK] All dependencies installed.")


# ── Step 5: Run convert.py ────────────────────────────────────────────────

def run_convert():
    section("STEP 5: Generating captions.csv")

    if CAPTIONS_CSV.exists():
        n_lines = sum(1 for _ in open(CAPTIONS_CSV)) - 1
        print(f"[OK] captions.csv already exists with {n_lines} pairs")
        return

    if not TOKEN_FILE.exists():
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

    convert_path = None
    for p in ["/convert.py", "convert.py", "src/convert.py"]:
        if os.path.exists(p):
            convert_path = p
            break
    if convert_path is None:
        print("[ERROR] convert.py not found.")
        sys.exit(1)

    env = {**os.environ, "CAPTION_FILE": str(TOKEN_FILE),
           "IMAGE_FOLDER": str(IMAGE_DIR), "OUTPUT_CSV": str(CAPTIONS_CSV)}
    result = subprocess.run(["python", convert_path], env=env)
    if result.returncode != 0 or not CAPTIONS_CSV.exists():
        print("[ERROR] convert.py failed.")
        sys.exit(1)

    n_lines = sum(1 for _ in open(CAPTIONS_CSV)) - 1
    print(f"[OK] captions.csv: {n_lines} pairs")


# ── Training launcher (shared) ─────────────────────────────────────────────

def launch_training(batch_size, num_epochs, log_file,
                    resume="no", block_until_done=False):
    """Launch one training run. If block_until_done=True, waits for completion."""

    train_path = None
    for p in ["/train.py", "train.py", "src/train.py"]:
        if os.path.exists(p):
            train_path = p
            break
    if train_path is None:
        print("[ERROR] train.py not found.")
        sys.exit(1)

    env_vars = {
        "DATASET_MODE":     DATASET_MODE,
        "CSV_PATH":         str(CAPTIONS_CSV),
        "IMAGE_ROOT":       str(IMAGE_DIR),
        "OUTPUT_DIR":       str(OUTPUT_DIR),
        "NUM_IMAGES":       NUM_IMAGES,
        "BATCH_SIZE":       str(batch_size),
        "NUM_EPOCHS":       str(num_epochs),
        "LR":               LR,
        "GRAD_ACCUM_STEPS": GRAD_ACCUM,
        "NUM_WORKERS":      "4",
        "TIMESTEP_BIAS":    "0.85",
        "EMA_DECAY":        "0.9995",
        "SNR_GAMMA":        SNR_GAMMA,
        "VAL_EVERY":        "3",
        "VAL_IMAGES":       "32",
        "VAL_STEPS":        "20",
        "RESUME_CKPT":      "auto" if resume == "yes" else "",
    }

    env_export = " ".join(f'{k}="{v}"' for k, v in env_vars.items())

    if block_until_done:
        # Run synchronously — wait for it to finish (used in ablation)
        cmd = f"env {env_export} python {train_path} > {log_file} 2>&1"
        run(cmd)
    else:
        # Run in background
        run("pkill -f 'python.*train.py' 2>/dev/null || true", check=False)
        time.sleep(2)
        cmd = (
            f"env {env_export} "
            f"nohup python {train_path} "
            f"> {log_file} 2>&1 & "
            f"echo $! > {TRAIN_PID} && "
            f"echo 'Training PID:' $(cat {TRAIN_PID})"
        )
        run(cmd)
        time.sleep(4)
        if TRAIN_PID.exists():
            pid = open(TRAIN_PID).read().strip()
            alive = run(f"ps -p {pid} > /dev/null 2>&1", check=False)
            if alive:
                print(f"[OK] Training running (PID: {pid})")
            else:
                print("[WARN] Process not running. Check log:")
                run(f"tail -20 {log_file}", check=False)


# ── Step 6a: Batch size ablation ──────────────────────────────────────────

def run_batch_ablation():
    section(f"STEP 6: Batch Size Ablation — {DATASET_MODE} Dataset")
    print(f"  Running batch sizes: {ABLATION_BATCH_SIZES}")
    print(f"  Epochs per run: {ABLATION_EPOCHS}")
    print(f"  Loss: SNR-weighted Huber | LR: {LR}")
    print(f"  Each run saved to separate log\n")

    ablation_dir = OUTPUT_DIR / f"ablation_{DATASET_MODE}"
    ablation_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for bs in ABLATION_BATCH_SIZES:
        eff_batch = int(bs) * int(GRAD_ACCUM)
        log_file  = ablation_dir / f"ablation_batch{bs}.log"
        print(f"\n{'─'*60}")
        print(f"  Running batch_size={bs} (effective={eff_batch})...")
        print(f"  Log: {log_file}")
        print(f"{'─'*60}")

        # Clear previous checkpoint so each ablation starts fresh
        for ckpt in (OUTPUT_DIR / "checkpoints").glob("*.pt"):
            ckpt.unlink()

        launch_training(
            batch_size=bs,
            num_epochs=ABLATION_EPOCHS,
            log_file=log_file,
            resume="no",
            block_until_done=True,  # wait for this run to finish before next
        )

        metrics = extract_final_metrics(log_file)
        results.append({"batch": bs, "effective": eff_batch, "metrics": metrics})
        print(f"\n[RESULT] batch={bs}: "
              f"noise_MSE={metrics['noise_MSE']} | "
              f"img_MSE={metrics['img_MSE']} | "
              f"img_PSNR={metrics['img_PSNR']} | "
              f"img_SSIM={metrics['img_SSIM']}")

    # Print comparison table
    print_ablation_table(results)

    # Save table to file for the paper
    table_path = ablation_dir / "ablation_results.txt"
    with open(table_path, "w") as f:
        f.write(f"BATCH SIZE ABLATION — {DATASET_MODE} Dataset\n")
        f.write(f"Loss: SNR-weighted Huber | Epochs: {ABLATION_EPOCHS} | LR: {LR}\n\n")
        f.write(f"{'Batch':>8} {'Eff.Batch':>10} {'Noise MSE':>12} "
                f"{'Noise MAE':>12} {'img MSE':>10} {'img PSNR':>10} {'img SSIM':>10}\n")
        f.write("-" * 70 + "\n")
        for r in results:
            f.write(f"  {r['batch']:>6} {str(r['effective']):>10} "
                    f"{r['metrics']['noise_MSE']:>12} "
                    f"{r['metrics']['noise_MAE']:>12} "
                    f"{r['metrics']['img_MSE']:>10} "
                    f"{r['metrics']['img_PSNR']:>10} "
                    f"{r['metrics']['img_SSIM']:>10}\n")
    print(f"\n[OK] Ablation results saved to: {table_path}")
    print(f"[INFO] Now run the full training with the best batch size:")
    print(f"       python /autorun.py  (uses batch={BATCH_SIZE} by default)")


# ── Step 6b: Full training run ────────────────────────────────────────────

def start_training():
    section(f"STEP 6: Starting Full Training — {DATASET_MODE} mode")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "inference").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "validation").mkdir(parents=True, exist_ok=True)

    check_disk()

    print(f"\n[CONFIG] Dataset mode: {DATASET_MODE}")
    print(f"         {DESCRIPTION}")
    print(f"         Loss: SNR-weighted Huber (delta=1.0, gamma={SNR_GAMMA})")
    print(f"         Batch: {BATCH_SIZE} x {GRAD_ACCUM} accum = "
          f"{int(BATCH_SIZE)*int(GRAD_ACCUM)} effective")
    print(f"         Epochs: {NUM_EPOCHS} | LR: {LR}")
    print(f"         Resume: {RESUME}")

    launch_training(
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        log_file=LOG_FILE,
        resume=RESUME,
        block_until_done=False,
    )
    print(f"\n[INFO] Log: {LOG_FILE}")


# ── Step 7: Tail log ──────────────────────────────────────────────────────

def tail_log():
    section("STEP 7: Live Training Log")
    print(f"[INFO] Watching {LOG_FILE}")
    print("[INFO] Press Ctrl+C to stop watching (training continues in background)")
    print("-" * 60)

    for _ in range(30):
        if LOG_FILE.exists() and LOG_FILE.stat().st_size > 0:
            break
        print(".", end="", flush=True)
        time.sleep(1)
    print()

    if not LOG_FILE.exists():
        print("[ERROR] Log not created.")
        return

    try:
        run(f"tail -f {LOG_FILE}")
    except KeyboardInterrupt:
        print(f"\n\n[INFO] Stopped watching. Training continues in background.")
        print(f"[INFO] Watch again:   tail -f {LOG_FILE}")
        print(f"[INFO] Check status:  ps aux | grep train.py")
        print(f"\n[INFO] Download results (from LOCAL terminal):")
        print(f"  scp -P PORT root@ssh.vast.ai:"
              f"{OUTPUT_DIR}/inference/generated_epoch*.png .")
        print(f"  scp -P PORT root@ssh.vast.ai:{LOG_FILE} .")
        print(f"  scp -P PORT root@ssh.vast.ai:"
              f"{OUTPUT_DIR}/checkpoints/checkpoint_best.pt .")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  SKETCH DIFFUSION v6 — Automation Script")
    if ABLATION_MODE == "batch":
        print(f"  Mode:    BATCH SIZE ABLATION on {DATASET_MODE}")
        print(f"           Batch sizes: {ABLATION_BATCH_SIZES} x {ABLATION_EPOCHS} epochs each")
    else:
        print(f"  Mode:    Full training — {DATASET_MODE}")
        print(f"           {DESCRIPTION}")
    print(f"  Bucket:  gs://{BUCKET_NAME}")
    print(f"  Resume:  {RESUME}")
    print("=" * 60)
    print()
    print("  Commands:")
    print("    10k full run:     python /autorun.py")
    print("    30k full run:     DATASET_MODE=30k SKIP_DOWNLOAD=yes python /autorun.py")
    print("    10k ablation:     ABLATION_MODE=batch python /autorun.py")
    print("    30k ablation:     ABLATION_MODE=batch DATASET_MODE=30k python /autorun.py")
    print("    Resume run:       RESUME=yes SKIP_DOWNLOAD=yes python /autorun.py")
    print()

    download_code_from_github()   # Step 0 — always pull latest code from GitHub
    install_gcloud()
    authenticate_gcp()
    download_dataset()
    install_dependencies()
    run_convert()

    if ABLATION_MODE == "batch":
        run_batch_ablation()
    else:
        start_training()
        tail_log()


if __name__ == "__main__":
    main()
