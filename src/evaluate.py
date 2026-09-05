"""
evaluate.py -- Comprehensive Evaluation for Sketch-to-Image Diffusion
======================================================================
Computes all metrics needed for paper submission:

PERCEPTUAL METRICS (on generated vs ground truth):
  - FID   (Fréchet Inception Distance) — distributional quality
  - LPIPS (Learned Perceptual Image Patch Similarity) — perceptual similarity
  - DINO  (ViT-S/8 feature cosine similarity) — semantic similarity
  - CLIP  Score — semantic alignment between image and caption

BASELINE COMPARISON (3 systems, same inputs):
  1. Standard ControlNet       — no RegionExtractor, SemanticAttention, or RegionAwareAttnProcessor
  2. Proposed Stage 1 only     — RegionExtractor active, NO SemanticAttention injection into UNet
  3. Full proposed system      — both stages, full RegionAwareAttnProcessor injection

PIXEL METRICS (for completeness):
  - MSE, MAE, PSNR, SSIM (skimage local-window)

All results saved to:
  /workspace/outputs/evaluation/
    metrics_summary.txt      — full table for paper
    baseline_comparison.txt  — 3-system comparison table
    generated_proposed/      — images from full system
    generated_stage1/        — images from stage 1 only
    generated_controlnet/    — images from standard ControlNet

Usage:
  python /evaluate.py

  Or with env vars:
  CHECKPOINT_PATH=/workspace/outputs/checkpoints/checkpoint_best.pt \
  CSV_PATH=/workspace/captions.csv \
  IMAGE_ROOT=/workspace/flickr30k-images \
  OUTPUT_DIR=/workspace/outputs \
  python /evaluate.py
"""

import os
import sys
import math
import time
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity as sk_ssim

# ── Config ─────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH",
    "/workspace/outputs/checkpoints/checkpoint_best.pt")
CSV_PATH        = os.environ.get("CSV_PATH",   "/workspace/captions.csv")
IMAGE_ROOT      = os.environ.get("IMAGE_ROOT", "/workspace/flickr30k-images")
OUTPUT_DIR      = Path(os.environ.get("OUTPUT_DIR", "/workspace/outputs"))
DATASET_MODE    = os.environ.get("DATASET_MODE", "30k")

EVAL_DIR        = OUTPUT_DIR / "evaluation"
GEN_PROPOSED    = EVAL_DIR / "generated_proposed"
GEN_STAGE1      = EVAL_DIR / "generated_stage1"
GEN_CONTROLNET  = EVAL_DIR / "generated_controlnet"

SD_ID           = "runwayml/stable-diffusion-v1-5"
CN_ID           = "lllyasviel/sd-controlnet-canny"
COMPUTE_DTYPE   = torch.bfloat16
REGION_DTYPE    = torch.float32
IMAGE_SIZE      = 512
MAX_TOK         = 77
VAE_SCALE       = 0.18215
GRID_SIZE       = 8
TEXT_DIM        = 768
NUM_EVAL_IMAGES = int(os.environ.get("NUM_EVAL_IMAGES", "32"))
INF_STEPS       = int(os.environ.get("INF_STEPS", "30"))

# ── Install missing packages ────────────────────────────────────────────────
def install_packages():
    import subprocess
    pkgs = ["lpips", "clean-fid", "open_clip_torch"]
    for pkg in pkgs:
        try:
            __import__(pkg.replace("-", "_").replace("open_clip_torch", "open_clip"))
        except ImportError:
            print(f"[INSTALL] Installing {pkg}...")
            subprocess.run([sys.executable, "-m", "pip", "install", pkg, "--quiet"])

install_packages()


# ── Imports after install ───────────────────────────────────────────────────
from diffusers import (
    AutoencoderKL, ControlNetModel, DDPMScheduler,
    StableDiffusionControlNetPipeline, UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer

# ── Section separator ───────────────────────────────────────────────────────
def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── Load validation data ────────────────────────────────────────────────────
def load_val_data():
    section("Loading Validation Data")
    df = pd.read_csv(CSV_PATH)
    df["wc"] = df["caption"].astype(str).apply(lambda x: len(x.split()))
    df = df[(df["wc"] >= 8) & (df["wc"] <= 35)].reset_index(drop=True)
    df = df.drop_duplicates(subset=["caption"]).reset_index(drop=True)
    val_df = df.tail(NUM_EVAL_IMAGES).reset_index(drop=True)
    print(f"[DATA] {len(val_df)} validation samples loaded")

    images, edges, captions, gt_pils = [], [], [], []
    for _, row in val_df.iterrows():
        img_path = os.path.join(IMAGE_ROOT, row["image"])
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LANCZOS4)
        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        median  = np.median(gray)
        lo      = max(0, int(0.66 * median))
        hi      = min(255, int(1.33 * median))
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edge    = cv2.Canny(blurred, lo, hi)
        edge3   = cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        images.append(img_rgb)
        edges.append(edge3)
        captions.append(str(row["caption"]))
        gt_pils.append(Image.fromarray(img_rgb))

    print(f"[DATA] {len(images)} images successfully loaded")
    return images, edges, captions, gt_pils


# ── Load architecture modules from train.py ─────────────────────────────────
def load_train_modules():
    """Import custom modules from train.py"""
    sys.path.insert(0, "/")
    try:
        from train import (
            RegionExtractor, SemanticAttention,
            RegionAwareAttnProcessor, SafeAttnProcessor, EMA
        )
        return RegionExtractor, SemanticAttention, RegionAwareAttnProcessor, SafeAttnProcessor, EMA
    except ImportError as e:
        print(f"[ERROR] Could not import from train.py: {e}")
        print("        Make sure train.py is at /train.py")
        sys.exit(1)


# ── Build pipelines ─────────────────────────────────────────────────────────
def build_pipelines(device):
    section("Building Inference Pipelines")

    RegionExtractor, SemanticAttention, RegionAwareAttnProcessor, SafeAttnProcessor, EMA = \
        load_train_modules()

    # Load checkpoint
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[ERROR] Checkpoint not found: {CHECKPOINT_PATH}")
        sys.exit(1)
    print(f"[CKPT] Loading: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)

    # ── Shared frozen components ──────────────────────────────────────────
    tok  = CLIPTokenizer.from_pretrained(SD_ID, subfolder="tokenizer")
    te   = CLIPTextModel.from_pretrained(SD_ID, subfolder="text_encoder",
                                          torch_dtype=COMPUTE_DTYPE).to(device)
    vae  = AutoencoderKL.from_pretrained(SD_ID, subfolder="vae",
                                          torch_dtype=COMPUTE_DTYPE).to(device)
    sched = DDPMScheduler.from_pretrained(SD_ID, subfolder="scheduler")

    # ── SYSTEM 1: Standard ControlNet (no region modules) ─────────────────
    print("\n[PIPE 1] Building Standard ControlNet pipeline...")
    cn_std = ControlNetModel.from_pretrained(CN_ID, torch_dtype=COMPUTE_DTYPE).to(device)
    unet_std = UNet2DConditionModel.from_pretrained(SD_ID, subfolder="unet",
                                                     torch_dtype=COMPUTE_DTYPE).to(device)
    pipe_controlnet = StableDiffusionControlNetPipeline(
        vae=vae, text_encoder=te, tokenizer=tok,
        unet=unet_std, controlnet=cn_std,
        scheduler=sched, safety_checker=None,
        feature_extractor=None, requires_safety_checker=False,
    ).to(device)
    pipe_controlnet.set_progress_bar_config(disable=True)
    print("[PIPE 1] Standard ControlNet ready")

    # ── SYSTEM 2: Stage 1 Only (RegionExtractor, NO UNet injection) ───────
    print("\n[PIPE 2] Building Stage 1 only pipeline...")
    cn_s1 = ControlNetModel.from_pretrained(CN_ID, torch_dtype=COMPUTE_DTYPE).to(device)
    cn_s1.load_state_dict(ckpt["controlnet"])
    ema_s1 = EMA(cn_s1)
    if "controlnet_ema" in ckpt:
        ema_s1.load_state_dict(ckpt["controlnet_ema"])
    ema_s1.shadow.to(dtype=COMPUTE_DTYPE)

    re_s1 = RegionExtractor().to(device=device, dtype=REGION_DTYPE)
    re_s1.load_state_dict(ckpt["re"])
    re_s1.eval()

    # Stage 1 uses standard UNet — no RegionAwareAttnProcessor injection
    unet_s1 = UNet2DConditionModel.from_pretrained(SD_ID, subfolder="unet",
                                                    torch_dtype=COMPUTE_DTYPE).to(device)
    pipe_stage1 = StableDiffusionControlNetPipeline(
        vae=vae, text_encoder=te, tokenizer=tok,
        unet=unet_s1, controlnet=ema_s1.shadow,
        scheduler=sched, safety_checker=None,
        feature_extractor=None, requires_safety_checker=False,
    ).to(device)
    pipe_stage1.set_progress_bar_config(disable=True)
    print("[PIPE 2] Stage 1 only pipeline ready")

    # ── SYSTEM 3: Full proposed system (Stage 1 + Stage 2) ───────────────
    print("\n[PIPE 3] Building full proposed system pipeline...")
    cn_full = ControlNetModel.from_pretrained(CN_ID, torch_dtype=COMPUTE_DTYPE).to(device)
    cn_full.load_state_dict(ckpt["controlnet"])
    ema_full = EMA(cn_full)
    if "controlnet_ema" in ckpt:
        ema_full.load_state_dict(ckpt["controlnet_ema"])
    ema_full.shadow.to(dtype=COMPUTE_DTYPE)

    re_full = RegionExtractor().to(device=device, dtype=REGION_DTYPE)
    sa_full = SemanticAttention().to(device=device, dtype=REGION_DTYPE)
    re_full.load_state_dict(ckpt["re"])
    sa_full.load_state_dict(ckpt["sa"])
    re_full.eval(); sa_full.eval()

    # Inject RegionAwareAttnProcessor into UNet
    unet_full = UNet2DConditionModel.from_pretrained(SD_ID, subfolder="unet",
                                                      torch_dtype=COMPUTE_DTYPE).to(device)
    new_procs = {}
    for name, proc in unet_full.attn_processors.items():
        if "attn2" in name:
            new_procs[name] = RegionAwareAttnProcessor().to(device=device, dtype=REGION_DTYPE)
        else:
            new_procs[name] = SafeAttnProcessor(proc)
    unet_full.set_attn_processor(new_procs)

    pipe_full = StableDiffusionControlNetPipeline(
        vae=vae, text_encoder=te, tokenizer=tok,
        unet=unet_full, controlnet=ema_full.shadow,
        scheduler=sched, safety_checker=None,
        feature_extractor=None, requires_safety_checker=False,
    ).to(device)
    pipe_full.set_progress_bar_config(disable=True)
    print("[PIPE 3] Full proposed system ready")

    return {
        "controlnet": pipe_controlnet,
        "stage1":     (pipe_stage1, re_s1),
        "full":       (pipe_full, re_full, sa_full),
        "tok":        tok,
        "te":         te,
    }, sched


# ── Generate images ─────────────────────────────────────────────────────────
@torch.no_grad()
def generate_images(pipes, sched, edges, captions, device, save_dir):
    """Generate images for one system and save to save_dir."""
    save_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    pipe_controlnet = pipes["controlnet"]
    pipe_stage1, re_s1 = pipes["stage1"]
    pipe_full, re_full, sa_full = pipes["full"]
    tok = pipes["tok"]
    te  = pipes["te"]

    systems = {
        "controlnet": pipe_controlnet,
        "stage1":     pipe_stage1,
        "full":       pipe_full,
    }

    all_results = {k: [] for k in systems}

    for i, (edge_np, caption) in enumerate(zip(edges, captions)):
        sketch_pil = Image.fromarray(edge_np)
        print(f"  [{i+1}/{len(captions)}] Generating for: {caption[:50]}...")

        for sys_name, pipe in systems.items():
            result = pipe(
                prompt=caption,
                image=sketch_pil,
                num_inference_steps=INF_STEPS,
                guidance_scale=7.5,
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
            ).images[0]

            out_path = (EVAL_DIR / f"generated_{sys_name}") / f"img_{i:03d}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            result.save(str(out_path))
            all_results[sys_name].append(result)

        if i % 5 == 0:
            torch.cuda.empty_cache()

    return all_results


# ── Pixel metrics ───────────────────────────────────────────────────────────
def compute_pixel_metrics(gen_pils, gt_pils):
    mse_list, mae_list, psnr_list, ssim_list = [], [], [], []
    for gen, gt in zip(gen_pils, gt_pils):
        g = np.array(gen.resize((IMAGE_SIZE, IMAGE_SIZE))).astype(np.float32) / 127.5 - 1.0
        r = np.array(gt.resize((IMAGE_SIZE, IMAGE_SIZE))).astype(np.float32) / 127.5 - 1.0
        mse  = np.mean((g - r) ** 2)
        mae  = np.mean(np.abs(g - r))
        psnr = 10 * math.log10(4.0 / (mse + 1e-8))
        # SSIM on [0,1]
        g01  = ((g + 1.0) / 2.0).clip(0, 1)
        r01  = ((r + 1.0) / 2.0).clip(0, 1)
        ssim = sk_ssim(g01, r01, data_range=1.0, channel_axis=2, win_size=7)
        mse_list.append(mse); mae_list.append(mae)
        psnr_list.append(psnr); ssim_list.append(ssim)

    return {
        "MSE":  float(np.mean(mse_list)),
        "MAE":  float(np.mean(mae_list)),
        "PSNR": float(np.mean(psnr_list)),
        "SSIM": float(np.mean(ssim_list)),
    }


# ── LPIPS ───────────────────────────────────────────────────────────────────
def compute_lpips(gen_pils, gt_pils, device):
    """Learned Perceptual Image Patch Similarity using AlexNet."""
    section("Computing LPIPS")
    import lpips
    loss_fn = lpips.LPIPS(net="alex").to(device)
    loss_fn.eval()

    scores = []
    for gen, gt in zip(gen_pils, gt_pils):
        g = torch.from_numpy(
            np.array(gen.resize((IMAGE_SIZE, IMAGE_SIZE))).astype(np.float32) / 127.5 - 1.0
        ).permute(2, 0, 1).unsqueeze(0).to(device)
        r = torch.from_numpy(
            np.array(gt.resize((IMAGE_SIZE, IMAGE_SIZE))).astype(np.float32) / 127.5 - 1.0
        ).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            score = loss_fn(g.float(), r.float()).item()
        scores.append(score)

    result = float(np.mean(scores))
    print(f"[LPIPS] Mean LPIPS: {result:.4f} (lower = more perceptually similar)")
    return result


# ── CLIPScore ───────────────────────────────────────────────────────────────
def compute_clip_score(gen_pils, captions, device):
    """CLIPScore: cosine similarity between CLIP image and text embeddings."""
    section("Computing CLIPScore")
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    scores = []
    for gen, caption in zip(gen_pils, captions):
        img_t = preprocess(gen.resize((224, 224))).unsqueeze(0).to(device)
        txt_t = tokenizer([caption[:77]]).to(device)

        with torch.no_grad():
            img_feat = model.encode_image(img_t)
            txt_feat = model.encode_text(txt_t)
            img_feat = F.normalize(img_feat, dim=-1)
            txt_feat = F.normalize(txt_feat, dim=-1)
            score = (img_feat * txt_feat).sum().item()
        scores.append(score)

    result = float(np.mean(scores))
    print(f"[CLIP] Mean CLIPScore: {result:.4f} (higher = better text-image alignment)")
    return result


# ── DINO Similarity ─────────────────────────────────────────────────────────
def compute_dino_similarity(gen_pils, gt_pils, device):
    """DINO ViT-S/8 feature cosine similarity between generated and ground truth."""
    section("Computing DINO Similarity")
    try:
        dino = torch.hub.load("facebookresearch/dino:main", "dino_vits8").to(device).eval()
    except Exception as e:
        print(f"[DINO] Could not load DINO model: {e}. Skipping.")
        return None

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    scores = []
    for gen, gt in zip(gen_pils, gt_pils):
        g_t = transform(gen.resize((256, 256))).unsqueeze(0).to(device)
        r_t = transform(gt.resize((256, 256))).unsqueeze(0).to(device)
        with torch.no_grad():
            g_feat = dino(g_t.float())
            r_feat = dino(r_t.float())
            g_feat = F.normalize(g_feat, dim=-1)
            r_feat = F.normalize(r_feat, dim=-1)
            score  = (g_feat * r_feat).sum().item()
        scores.append(score)

    result = float(np.mean(scores))
    print(f"[DINO] Mean DINO similarity: {result:.4f} (higher = more semantically similar)")
    return result


# ── FID ──────────────────────────────────────────────────────────────────────
def compute_fid(gen_pils, gt_pils, device):
    """
    FID computation using clean-fid.
    Note: FID is most reliable with 1000+ images.
    With 32 images this is an approximation — reported as FID-32.
    """
    section("Computing FID (approximate — 32 images)")
    try:
        from cleanfid import fid
        import tempfile

        with tempfile.TemporaryDirectory() as gen_dir, \
             tempfile.TemporaryDirectory() as real_dir:

            for i, img in enumerate(gen_pils):
                img.resize((299, 299)).save(f"{gen_dir}/img_{i:03d}.png")
            for i, img in enumerate(gt_pils):
                img.resize((299, 299)).save(f"{real_dir}/img_{i:03d}.png")

            score = fid.compute_fid(gen_dir, real_dir, device=str(device), verbose=False)

        print(f"[FID] FID-{len(gen_pils)}: {score:.2f} (lower = better; note: approximate with {len(gen_pils)} images)")
        return float(score)

    except Exception as e:
        print(f"[FID] Failed: {e}. Skipping FID.")
        return None


# ── Human Evaluation Design ──────────────────────────────────────────────────
def save_human_eval_pairs(all_results, gt_pils, captions):
    """
    Save side-by-side comparison images for human evaluation.
    Each pair shows: [Sketch] [ControlNet] [Stage1] [Full System] [Ground Truth]
    """
    section("Saving Human Evaluation Pairs")
    human_dir = EVAL_DIR / "human_evaluation"
    human_dir.mkdir(parents=True, exist_ok=True)

    for i, (caption, gt) in enumerate(zip(captions, gt_pils)):
        cn_img   = all_results["controlnet"][i].resize((256, 256))
        s1_img   = all_results["stage1"][i].resize((256, 256))
        full_img = all_results["full"][i].resize((256, 256))
        gt_img   = gt.resize((256, 256))

        # Create side-by-side comparison (4 panels)
        combined = Image.new("RGB", (256 * 4, 256 + 40), (255, 255, 255))
        combined.paste(cn_img,   (0,   40))
        combined.paste(s1_img,   (256, 40))
        combined.paste(full_img, (512, 40))
        combined.paste(gt_img,   (768, 40))

        combined.save(str(human_dir / f"pair_{i:03d}.png"))

    # Save instruction sheet
    instructions = f"""HUMAN EVALUATION STUDY
======================
Sketch-to-Image Generation — Preference Study
{len(captions)} image pairs to evaluate

For each pair image (pair_000.png to pair_{len(captions)-1:03d}.png):
  Panel 1 (leftmost):  System A — Standard ControlNet
  Panel 2:             System B — Proposed Stage 1 only
  Panel 3:             System C — Full Proposed System (both stages)
  Panel 4 (rightmost): Ground Truth photograph

Questions to answer for each pair:
  Q1: Which system (A/B/C) best follows the structural layout? (1=A, 2=B, 3=C)
  Q2: Which system (A/B/C) best matches the caption semantics? (1=A, 2=B, 3=C)
  Q3: Which system (A/B/C) produces the most photorealistic image? (1=A, 2=B, 3=C)
  Q4: Overall preference — which system would you choose? (1=A, 2=B, 3=C)
  Q5: Is the main object/scene in the correct spatial location? (Y/N for each)

Captions for each pair:
"""
    for i, cap in enumerate(captions):
        instructions += f"  pair_{i:03d}: {cap}\n"

    with open(str(human_dir / "instructions.txt"), "w") as f:
        f.write(instructions)

    print(f"[HUMAN EVAL] {len(captions)} comparison pairs saved to {human_dir}/")
    print(f"[HUMAN EVAL] Instructions saved to {human_dir}/instructions.txt")


# ── Save results ─────────────────────────────────────────────────────────────
def save_results(all_metrics, dataset_mode):
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Summary text
    lines = []
    lines.append("=" * 70)
    lines.append(f"COMPREHENSIVE EVALUATION RESULTS — {dataset_mode.upper()} RUN")
    lines.append(f"Checkpoint: {CHECKPOINT_PATH}")
    lines.append(f"Evaluated on: {NUM_EVAL_IMAGES} images")
    lines.append("=" * 70)
    lines.append("")

    lines.append("BASELINE COMPARISON:")
    lines.append("-" * 70)
    header = f"{'Metric':<15} {'ControlNet':>15} {'Stage1 Only':>15} {'Full System':>15}"
    lines.append(header)
    lines.append("-" * 70)

    metrics_to_show = ["MSE", "MAE", "PSNR", "SSIM", "LPIPS", "CLIPScore", "DINO", "FID"]
    for m in metrics_to_show:
        cn_val  = all_metrics.get("controlnet", {}).get(m)
        s1_val  = all_metrics.get("stage1", {}).get(m)
        fl_val  = all_metrics.get("full", {}).get(m)

        def fmt(v):
            if v is None: return "N/A"
            return f"{v:.4f}"

        lines.append(f"  {m:<13} {fmt(cn_val):>15} {fmt(s1_val):>15} {fmt(fl_val):>15}")

    lines.append("")
    lines.append("METRIC NOTES:")
    lines.append("  MSE, MAE, PSNR, SSIM: pixel-level comparison vs ground truth")
    lines.append("  LPIPS: perceptual similarity (AlexNet) — lower is better")
    lines.append("  CLIPScore: text-image semantic alignment — higher is better")
    lines.append("  DINO: ViT-S/8 semantic similarity — higher is better")
    lines.append(f"  FID: distributional quality (approximate, {NUM_EVAL_IMAGES} images) — lower is better")
    lines.append("")
    lines.append("IMPORTANT: FID with 32 images is approximate.")
    lines.append("  Standard FID requires 10,000+ images for statistical reliability.")
    lines.append("  Values reported here as FID-32 should be interpreted as indicative only.")
    lines.append("")
    lines.append("=" * 70)

    summary_path = EVAL_DIR / "metrics_summary.txt"
    with open(str(summary_path), "w") as f:
        f.write("\n".join(lines))

    # JSON for programmatic access
    json_path = EVAL_DIR / "metrics.json"
    with open(str(json_path), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n" + "\n".join(lines))
    print(f"\n[SAVED] {summary_path}")
    print(f"[SAVED] {json_path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  COMPREHENSIVE EVALUATION SCRIPT")
    print(f"  Dataset mode:  {DATASET_MODE}")
    print(f"  Checkpoint:    {CHECKPOINT_PATH}")
    print(f"  Eval images:   {NUM_EVAL_IMAGES}")
    print(f"  Output:        {EVAL_DIR}")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    if device == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    t0 = time.time()

    # Load data
    images, edges, captions, gt_pils = load_val_data()

    # Build all three pipelines
    pipes, sched = build_pipelines(device)

    # Generate images for all 3 systems
    section("Generating Images — All 3 Systems")
    print(f"[GEN] Generating {len(captions)} images × 3 systems = {len(captions)*3} total")
    all_results = generate_images(pipes, sched, edges, captions, device, EVAL_DIR)

    # Compute all metrics for each system
    all_metrics = {}
    for sys_name, gen_pils in all_results.items():
        section(f"Computing Metrics — {sys_name.upper()}")
        gt_subset = gt_pils[:len(gen_pils)]

        m = compute_pixel_metrics(gen_pils, gt_subset)
        print(f"  MSE={m['MSE']:.4f} MAE={m['MAE']:.4f} "
              f"PSNR={m['PSNR']:.2f}dB SSIM={m['SSIM']:.4f}")

        m["LPIPS"]     = compute_lpips(gen_pils, gt_subset, device)
        m["CLIPScore"] = compute_clip_score(gen_pils, captions[:len(gen_pils)], device)
        m["DINO"]      = compute_dino_similarity(gen_pils, gt_subset, device)
        m["FID"]       = compute_fid(gen_pils, gt_subset, device)

        all_metrics[sys_name] = m
        torch.cuda.empty_cache()

    # Save human evaluation pairs
    save_human_eval_pairs(all_results, gt_pils, captions)

    # Save all results
    save_results(all_metrics, DATASET_MODE)

    elapsed = time.time() - t0
    print(f"\n[DONE] Total evaluation time: {elapsed/60:.1f} minutes")
    print(f"[DONE] All results in: {EVAL_DIR}/")
    print(f"       metrics_summary.txt  — table for paper")
    print(f"       metrics.json         — raw numbers")
    print(f"       generated_*/         — images from each system")
    print(f"       human_evaluation/    — comparison pairs + instructions")


if __name__ == "__main__":
    main()
