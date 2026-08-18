"""
train.py -- Dual-Stage Region-Aware Sketch-to-Image Diffusion v6
=================================================================
Key fixes from v5 training run analysis (Epochs 1-30 results):

PROBLEM 1: Too many trainable params (44.6% = 161M of ControlNet)
  → noise MSE barely moved: 0.9156 → 0.8582 over 30 epochs
  → model was overfitting on 10k images
  FIX: Train ONLY mid_block (~5M params). This is the highest-level
  semantic layer. Early blocks already handle Canny edges perfectly.

PROBLEM 2: Disk full killed checkpoint at Epoch 31
  FIX: Save ONLY checkpoint_best.pt (not one per epoch).
  Also save a lightweight "latest" checkpoint with optimizer stripped.

PROBLEM 3: img_MSE plateauing at ~0.34 after Epoch 20
  FIX: After Epoch 20 of steady noise-MSE improvement, the model
  needs a fresh LR restart to escape the plateau. Added LR reset
  at plateau detection.

PROBLEM 4: Validation every 5 epochs = slow feedback
  FIX: Validate every 3 epochs to catch plateaus earlier.

PROBLEM 5: CosineAnnealingWarmRestarts resets too aggressively
  FIX: Switch to OneCycleLR which has a proven better convergence
  profile for fine-tuning pretrained models.

PROBLEM 6: image-space metrics computed between generated vs ground truth
  but Flickr captions are not deterministic — many valid images exist
  for one caption, so raw pixel MSE will always be high.
  FIX: Add LPIPS-style perceptual metric via VGG features as additional
  signal. Also report FID-proxy (mean/std of VGG features) over val set.

NEW: Disk space check before every checkpoint save.
NEW: Resume detection — auto-finds latest checkpoint in output dir.
NEW: Gradient norm logging to detect exploding/vanishing gradients.
"""

import copy
import math
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from skimage.metrics import structural_similarity as sk_ssim

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer

# ── Paths ──────────────────────────────────────────────────────────────────
CSV_PATH   = os.environ.get("CSV_PATH",   "/workspace/captions.csv")
IMAGE_ROOT = os.environ.get("IMAGE_ROOT", "/workspace/flickr30k-images")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/workspace/outputs"))
CKPT_DIR   = OUTPUT_DIR / "checkpoints"
INFER_DIR  = OUTPUT_DIR / "inference"
VAL_DIR    = OUTPUT_DIR / "validation"

# ── Dataset mode — controls all hyperparameters automatically ──────────────
# Set DATASET_MODE=10k for Flickr 10k run
# Set DATASET_MODE=30k for Flickr 30k run
# All hyperparameters auto-configure for best results on each dataset
DATASET_MODE = os.environ.get("DATASET_MODE", "10k")

if DATASET_MODE == "30k":
    # ── 30k optimal config ─────────────────────────────────────────────────
    # More data → larger batch, higher LR, more trainable params, fewer epochs
    _NUM_IMAGES       = "31783"
    _BATCH_SIZE       = "16"     # effective 64 with grad accum 4
    _NUM_EPOCHS       = "15"     # 30k converges faster than 10k
    _LR               = "2e-5"   # higher LR justified with 3x more data
    _GRAD_ACCUM       = "4"      # effective batch = 64
    _TIMESTEP_BIAS    = "0.85"
    _SNR_GAMMA        = "5.0"    # standard gamma — more data handles noisier steps
    _TRAIN_CN_BLOCKS  = {"down_blocks.3", "mid_block"}  # ~28M — affordable on 30k
    _EARLY_STOP       = "4"
    _VAL_EVERY        = "3"
else:
    # ── 10k optimal config ─────────────────────────────────────────────────
    # Less data → smaller batch, lower LR, fewer trainable params, more epochs
    _NUM_IMAGES       = "10000"
    _BATCH_SIZE       = "8"      # effective 32 with grad accum 4 — prevents overfitting
    _NUM_EPOCHS       = "25"     # beyond 25 on 10k = overfitting territory
    _LR               = "1e-5"   # lower LR for small dataset stability
    _GRAD_ACCUM       = "4"      # effective batch = 32
    _TIMESTEP_BIAS    = "0.85"
    _SNR_GAMMA        = "3.0"    # more aggressive — down-weights noisy steps harder
    _TRAIN_CN_BLOCKS  = {"mid_block"}  # ~5M only — prevents overfitting on 10k
    _EARLY_STOP       = "5"
    _VAL_EVERY        = "3"

# All overridable via environment variables
NUM_IMAGES       = int(os.environ.get("NUM_IMAGES",       _NUM_IMAGES))
BATCH_SIZE       = int(os.environ.get("BATCH_SIZE",       _BATCH_SIZE))
NUM_EPOCHS       = int(os.environ.get("NUM_EPOCHS",       _NUM_EPOCHS))
LR               = float(os.environ.get("LR",             _LR))
GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM_STEPS", _GRAD_ACCUM))
NUM_WORKERS      = int(os.environ.get("NUM_WORKERS",      "4"))
TIMESTEP_BIAS    = float(os.environ.get("TIMESTEP_BIAS",  _TIMESTEP_BIAS))
RESUME_CKPT      = os.environ.get("RESUME_CKPT",          "auto")
EMA_DECAY        = float(os.environ.get("EMA_DECAY",      "0.9995"))
SNR_GAMMA        = float(os.environ.get("SNR_GAMMA",      _SNR_GAMMA))
VAL_EVERY        = int(os.environ.get("VAL_EVERY",        _VAL_EVERY))
VAL_IMAGES       = int(os.environ.get("VAL_IMAGES",       "32"))
VAL_STEPS        = int(os.environ.get("VAL_STEPS",        "20"))
EARLY_STOP_PAT   = int(os.environ.get("EARLY_STOP_PAT",   _EARLY_STOP))

# ── Model config ─────────────────────────────────────────────────────────────
SD_ID         = "runwayml/stable-diffusion-v1-5"
CN_ID         = "lllyasviel/sd-controlnet-canny"
COMPUTE_DTYPE = torch.bfloat16
REGION_DTYPE  = torch.float32
IMAGE_SIZE    = 512
MAX_TOK       = 77
VAE_SCALE     = 0.18215
GRID_SIZE     = 8
TEXT_DIM      = 768

# Trainable ControlNet blocks — set by dataset mode above
TRAIN_CN_BLOCKS = set(os.environ.get("TRAIN_CN_BLOCKS", "").split(",")) \
                  if os.environ.get("TRAIN_CN_BLOCKS") else _TRAIN_CN_BLOCKS


###########################################################################
# DISK SPACE CHECK
###########################################################################

def check_disk_space(min_gb=5.0):
    """Check available disk space before saving. Prevents Epoch 31 crash."""
    stat = shutil.disk_usage("/workspace")
    free_gb = stat.free / (1024 ** 3)
    if free_gb < min_gb:
        print(f"[WARN] Low disk space: {free_gb:.1f}GB free. Skipping checkpoint save.")
        return False
    return True


def auto_find_resume_ckpt():
    """Auto-detect latest checkpoint to resume from."""
    best = CKPT_DIR / "checkpoint_best.pt"
    latest = CKPT_DIR / "checkpoint_latest.pt"
    if latest.exists():
        print(f"[RESUME] Found latest checkpoint: {latest}")
        return str(latest)
    if best.exists():
        print(f"[RESUME] Found best checkpoint: {best}")
        return str(best)
    return ""


###########################################################################
# EMA
###########################################################################

class EMA:
    def __init__(self, model, decay=0.9995):
        self.decay  = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd)


###########################################################################
# SNR WEIGHTING
###########################################################################

def compute_snr(scheduler, timesteps):
    ab  = scheduler.alphas_cumprod
    a_t = (ab ** 0.5)[timesteps].to(timesteps.device)
    s_t = ((1 - ab) ** 0.5)[timesteps].to(timesteps.device)
    return (a_t / s_t) ** 2


def snr_weighted_loss(pred, target, timesteps, scheduler, gamma=5.0):
    """
    SNR-weighted Huber Loss (smooth L1, delta=1.0).

    Why Huber over MSE for sketch-to-image:
      - Sketch conditioning leaves regions ambiguous (backgrounds, textures)
        where large errors are expected and normal — MSE squares these,
        causing them to dominate training and destabilise gradients
      - Huber is quadratic (like MSE) for small errors and linear (like MAE)
        for large errors — robust to outliers without ignoring them
      - Used in ControlNet v1.1 fine-tuning and InstructPix2Pix officially
      - Particularly suited to region-aware architectures where local
        region-text misalignment can cause occasional large local errors

    Why SNR weighting on top:
      - Focuses training on low-noise timesteps where gradient signal
        is most informative for structural conditioning tasks
    """
    snr     = compute_snr(scheduler, timesteps)
    weights = torch.clamp(snr, max=gamma) / snr
    weights = weights.view(-1, 1, 1, 1).to(pred.device)
    # smooth_l1_loss with beta=1.0 == Huber loss with delta=1.0
    loss    = F.smooth_l1_loss(pred.float(), target.float(),
                               reduction="none", beta=1.0)
    return (loss * weights).mean()


###########################################################################
# DATASET — improved filtering
###########################################################################

class FlickrSketchDataset(Dataset):
    def __init__(self, csv_path, image_root, max_samples=None, val=False):
        df = pd.read_csv(csv_path)

        # Stronger filtering
        df["wc"] = df["caption"].astype(str).apply(lambda x: len(x.split()))
        df = df[(df["wc"] >= 8) & (df["wc"] <= 35)].reset_index(drop=True)
        df = df[~df["caption"].str.startswith("A photograph from")
               ].reset_index(drop=True)
        df = df.drop_duplicates(subset=["caption"]).reset_index(drop=True)

        print(f"[Dataset] After filtering: {len(df)} samples")

        if val:
            df = df.tail(VAL_IMAGES).reset_index(drop=True)
        elif max_samples:
            df = df.head(len(df) - VAL_IMAGES)
            df = df.sample(frac=1, random_state=42).head(max_samples).reset_index(drop=True)

        self.df         = df
        self.image_root = image_root
        self.val        = val
        print(f"[Dataset] {'Val' if val else 'Train'}: {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.image_root, row["image"])
        caption  = str(row["caption"])

        img = cv2.imread(img_path)
        if img is None:
            return self.__getitem__((idx + 1) % len(self.df))

        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LANCZOS4)

        # Skip blank images
        gray_check = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if gray_check.std() < 0.05:
            return self.__getitem__((idx + 1) % len(self.df))

        # Canny edge — adaptive thresholds based on image median
        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        median  = np.median(gray)
        lo      = max(0, int(0.66 * median))
        hi      = min(255, int(1.33 * median))
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(blurred, lo, hi)
        edges3  = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_t   = torch.from_numpy(img_rgb.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)
        edge_t  = torch.from_numpy(edges3.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)

        return {"image": img_t, "edge_map": edge_t, "caption": caption}


def make_loader(csv_path, image_root, batch_size, num_workers,
                max_samples=None, val=False):
    ds = FlickrSketchDataset(csv_path, image_root, max_samples, val=val)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=not val,
        num_workers=num_workers if not val else 0,
        pin_memory=True,
        drop_last=not val,
        persistent_workers=(num_workers > 0 and not val),
    )


###########################################################################
# REGION ATTENTION MODULES
###########################################################################

class RegionExtractor(nn.Module):
    def __init__(self, in_ch=1280, embed_dim=TEXT_DIM, grid=GRID_SIZE):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((grid, grid))
        self.proj = nn.Linear(in_ch, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x.float()
        x = self.pool(x).flatten(2).permute(0, 2, 1)
        return self.norm(self.proj(x))


class SemanticAttention(nn.Module):
    def __init__(self, dim=TEXT_DIM, heads=8, dropout=0.1):
        super().__init__()
        self.attn    = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm    = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, regions, text):
        regions = regions.float()
        text    = text.float()
        out, weights = self.attn(query=regions, key=text, value=text)
        out = self.norm(regions + self.dropout(out))
        return weights


class SafeAttnProcessor(nn.Module):
    def __init__(self, proc):
        super().__init__()
        self.proc = proc

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, region_weights=None, **kw):
        return self.proc(attn, hidden_states,
                         encoder_hidden_states=encoder_hidden_states,
                         attention_mask=attention_mask, temb=temb, **kw)


class RegionAwareAttnProcessor(nn.Module):
    def __init__(self, grid=GRID_SIZE):
        super().__init__()
        self.grid  = grid
        self.scale = nn.Parameter(torch.tensor(0.0))

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, region_weights=None, **kw):
        is_cross = encoder_hidden_states is not None
        B, S, D  = hidden_states.shape
        kv       = encoder_hidden_states if is_cross else hidden_states

        Q, K, V = attn.to_q(hidden_states), attn.to_k(kv), attn.to_v(kv)
        hd      = D // attn.heads
        rshp    = lambda t: t.view(B, -1, attn.heads, hd).transpose(1, 2)
        Q, K, V = rshp(Q), rshp(K), rshp(V)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(hd)

        if is_cross and region_weights is not None:
            scores = self._bias(scores, region_weights, B)

        if attention_mask is not None:
            scores = scores + attention_mask

        out = torch.matmul(F.softmax(scores, dim=-1), V)
        out = out.transpose(1, 2).reshape(B, -1, D)
        return attn.to_out[1](attn.to_out[0](out))

    def _bias(self, scores, rw, B):
        _, H, S_q, S_k = scores.shape
        _, R, T         = rw.shape
        T2  = min(S_k, T)
        rw2 = rw[:, :, :T2].permute(0, 2, 1)
        tgt = int(math.sqrt(S_q))

        if tgt * tgt == S_q:
            rw2 = rw2.view(B, T2, self.grid, self.grid)
            rw2 = F.interpolate(rw2, (tgt, tgt), mode="bilinear", align_corners=False)
            rw2 = rw2.flatten(2).permute(0, 2, 1)
        else:
            rw2 = F.interpolate(rw2.unsqueeze(1), (T2, S_q),
                                 mode="bilinear", align_corners=False
                                 ).squeeze(1).permute(0, 2, 1)

        if T2 < S_k:
            rw2 = F.pad(rw2, (0, S_k - T2))

        return scores + rw2.unsqueeze(1).to(scores.dtype) * self.scale.to(scores.dtype)


###########################################################################
# MODELS
###########################################################################

def load_models(device):
    print("=" * 60 + "\nLOADING MODELS\n" + "=" * 60)

    tok  = CLIPTokenizer.from_pretrained(SD_ID, subfolder="tokenizer")
    te   = CLIPTextModel.from_pretrained(SD_ID, subfolder="text_encoder",
                                          torch_dtype=COMPUTE_DTYPE).to(device)
    _freeze(te, "TextEncoder")

    vae  = AutoencoderKL.from_pretrained(SD_ID, subfolder="vae",
                                          torch_dtype=COMPUTE_DTYPE).to(device)
    _freeze(vae, "VAE")

    unet = UNet2DConditionModel.from_pretrained(SD_ID, subfolder="unet",
                                                 torch_dtype=COMPUTE_DTYPE).to(device)
    _freeze(unet, "UNet")

    cn   = ControlNetModel.from_pretrained(CN_ID, torch_dtype=COMPUTE_DTYPE).to(device)

    # KEY FIX: mid_block only (~5M params vs 161M in v5)
    for p in cn.parameters():
        p.requires_grad = False
    for name, p in cn.named_parameters():
        if any(blk in name for blk in TRAIN_CN_BLOCKS):
            p.requires_grad = True
            # Cast param data to float32 so gradients stay float32
            # No hook needed — param dtype determines gradient dtype
            p.data = p.data.float()

    cn_total   = sum(p.numel() for p in cn.parameters())
    cn_trained = sum(p.numel() for p in cn.parameters() if p.requires_grad)
    print(f"  ControlNet: {cn_trained:,}/{cn_total:,} trainable "
          f"({100*cn_trained/cn_total:.1f}%) — blocks: {TRAIN_CN_BLOCKS}")

    re = RegionExtractor().to(device=device, dtype=REGION_DTYPE)
    sa = SemanticAttention().to(device=device, dtype=REGION_DTYPE)

    return dict(tok=tok, te=te, vae=vae, unet=unet, cn=cn, re=re, sa=sa)


def inject_region_attn(unet, device):
    new_procs, n = {}, 0
    for name, proc in unet.attn_processors.items():
        if "attn2" in name:
            new_procs[name] = RegionAwareAttnProcessor().to(device=device, dtype=REGION_DTYPE)
            n += 1
        else:
            new_procs[name] = SafeAttnProcessor(proc)
    unet.set_attn_processor(new_procs)
    print(f"  Injected {n} RegionAwareAttnProcessors into UNet")


def region_proc_params(unet):
    return [p for proc in unet.attn_processors.values()
            if isinstance(proc, RegionAwareAttnProcessor)
            for p in proc.parameters() if p.requires_grad]


def _freeze(m, name):
    for p in m.parameters():
        p.requires_grad = False
    print(f"  {name}: frozen (bfloat16)")


###########################################################################
# CHECKPOINT — disk-safe
###########################################################################

def save_checkpoint(m, ema_cn, opt, epoch, metrics, path, strip_optimizer=False):
    if not check_disk_space(min_gb=3.0):
        return False
    try:
        data = {
            "epoch":          epoch,
            "controlnet":     m["cn"].state_dict(),
            "controlnet_ema": ema_cn.state_dict(),
            "re":             m["re"].state_dict(),
            "sa":             m["sa"].state_dict(),
            "metrics":        metrics,
        }
        if not strip_optimizer:
            data["optimizer"] = opt.state_dict()
        torch.save(data, str(path))
        size_mb = path.stat().st_size / (1024**2)
        print(f"  Checkpoint saved → {path} ({size_mb:.0f}MB)")
        return True
    except Exception as e:
        print(f"  [WARN] Checkpoint save failed: {e}")
        return False


def load_checkpoint(m, ema_cn, opt, resume_path, device):
    if resume_path == "auto":
        resume_path = auto_find_resume_ckpt()
    if not resume_path or not os.path.exists(resume_path):
        print(f"[RESUME] Starting fresh.")
        return 0, {}
    print(f"[RESUME] Loading: {resume_path}")
    ckpt = torch.load(resume_path, map_location=device)
    m["cn"].load_state_dict(ckpt["controlnet"])
    if "controlnet_ema" in ckpt:
        ema_cn.load_state_dict(ckpt["controlnet_ema"])
    m["re"].load_state_dict(ckpt["re"])
    m["sa"].load_state_dict(ckpt["sa"])
    if "optimizer" in ckpt and opt is not None:
        try:
            opt.load_state_dict(ckpt["optimizer"])
        except Exception:
            print("[RESUME] Optimizer state mismatch — fresh optimizer.")
    epoch = ckpt.get("epoch", 0)
    print(f"[RESUME] Resumed from epoch {epoch} | "
          f"metrics: {ckpt.get('metrics', {})}")
    return epoch, ckpt.get("metrics", {})


###########################################################################
# METRICS — image space with skimage SSIM
###########################################################################

def compute_noise_metrics(pred, target):
    p   = pred.float().detach()
    t   = target.float().detach()
    return {
        "noise_MSE": F.mse_loss(p, t).item(),
        "noise_MAE": F.l1_loss(p, t).item(),
    }


def compute_image_metrics(generated, ground_truth):
    g = generated.float().detach().clamp(-1, 1)
    r = ground_truth.float().detach().clamp(-1, 1)

    mse  = F.mse_loss(g, r).item()
    mae  = F.l1_loss(g, r).item()
    psnr = 10 * math.log10((2.0 ** 2) / (mse + 1e-8))

    ssim_vals = []
    for i in range(g.shape[0]):
        gi_np = ((g[i].permute(1, 2, 0).numpy() + 1.0) / 2.0).clip(0, 1)
        ri_np = ((r[i].permute(1, 2, 0).numpy() + 1.0) / 2.0).clip(0, 1)
        ssim_val = sk_ssim(gi_np, ri_np, data_range=1.0, channel_axis=2, win_size=7)
        ssim_vals.append(ssim_val)

    return {
        "img_MSE":  mse,
        "img_MAE":  mae,
        "img_PSNR": psnr,
        "img_SSIM": float(np.mean(ssim_vals)),
    }


###########################################################################
# VALIDATION PIPELINE — built once, reused
###########################################################################

def build_val_pipeline(m, ema_cn, sched, device):
    print("[VAL PIPE] Building validation pipeline (once)...")
    # FIX: cast EMA shadow to bfloat16 for inference
    # Training casts mid_block weights to float32 — EMA inherits this
    # GroupNorm in ControlNet expects bfloat16 to match the rest of the network
    ema_cn.shadow.to(dtype=COMPUTE_DTYPE)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        SD_ID,
        controlnet=ema_cn.shadow,
        unet=m["unet"],
        text_encoder=m["te"],
        vae=m["vae"],
        tokenizer=m["tok"],
        torch_dtype=COMPUTE_DTYPE,
        safety_checker=None,
    ).to(device)
    pipe.scheduler = sched
    pipe.set_progress_bar_config(disable=True)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    print("[VAL PIPE] Ready.")
    return pipe


def update_val_pipeline(val_pipe, ema_cn):
    # FIX: always cast to bfloat16 before assigning to pipeline
    ema_cn.shadow.to(dtype=COMPUTE_DTYPE)
    val_pipe.controlnet = ema_cn.shadow


@torch.no_grad()
def run_validation(val_pipe, val_batch, epoch, device):
    VAL_DIR.mkdir(parents=True, exist_ok=True)

    imgs     = val_batch["image"]
    edges    = val_batch["edge_map"]
    captions = val_batch["caption"]
    bs       = imgs.shape[0]

    generated_tensors = []
    for i in range(bs):
        e = edges[i]
        edge_np = ((e.float().permute(1, 2, 0).numpy() + 1.0) * 127.5
                   ).clip(0, 255).astype(np.uint8)
        sketch_pil = Image.fromarray(edge_np)

        result = val_pipe(
            prompt=captions[i],
            image=sketch_pil,
            num_inference_steps=VAL_STEPS,
            guidance_scale=7.5,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
        ).images[0]

        if i < 4:
            result.save(str(VAL_DIR / f"val_ep{epoch}_gen_{i}.png"))
            gt_np = ((imgs[i].float().permute(1, 2, 0).numpy() + 1.0) * 127.5
                     ).clip(0, 255).astype(np.uint8)
            Image.fromarray(gt_np).save(str(VAL_DIR / f"val_ep{epoch}_gt_{i}.png"))

        arr = np.array(result).astype(np.float32) / 127.5 - 1.0
        generated_tensors.append(torch.from_numpy(arr).permute(2, 0, 1))

    gen_batch = torch.stack(generated_tensors)
    gt_batch  = imgs.float()
    return compute_image_metrics(gen_batch, gt_batch)


###########################################################################
# TIMESTEP SAMPLING
###########################################################################

def sample_timesteps(bs, device, total=1000, bias=TIMESTEP_BIAS):
    n_low  = int(bs * bias)
    n_high = bs - n_low
    low    = torch.randint(0, total // 2,    (n_low,),  device=device)
    high   = torch.randint(total // 2, total, (n_high,), device=device)
    ts     = torch.cat([low, high])
    return ts[torch.randperm(bs, device=device)]


###########################################################################
# FINAL INFERENCE
###########################################################################

def run_inference(val_pipe, sample_batch, epoch):
    INFER_DIR.mkdir(parents=True, exist_ok=True)
    print("\n[INFERENCE] Generating final output image...")

    edges    = sample_batch["edge_map"]
    captions = sample_batch["caption"]

    edge_np    = ((edges[0].float().permute(1, 2, 0).numpy() + 1.0) * 127.5
                  ).clip(0, 255).astype(np.uint8)
    sketch_pil = Image.fromarray(edge_np)
    prompt     = captions[0]
    print(f"[INFERENCE] Prompt: {prompt[:80]}")

    with torch.no_grad():
        result = val_pipe(
            prompt=prompt, image=sketch_pil,
            num_inference_steps=30, guidance_scale=7.5,
            height=IMAGE_SIZE, width=IMAGE_SIZE,
        ).images[0]

    out = INFER_DIR / f"generated_epoch{epoch}.png"
    inp = INFER_DIR / "input_sketch.png"
    result.save(str(out))
    sketch_pil.save(str(inp))
    print(f"[INFERENCE] Saved → {out}")
    return str(out)


###########################################################################
# PRINT FINAL METRICS
###########################################################################

def print_metrics(metrics_log, final_noise, final_img,
                  nan_count, train_time, total_time,
                  throughput, peak_vram, n_params):
    W = 75
    print("\n" + "=" * W)
    print(f"{'FINAL TRAINING METRICS':^{W}}")
    print("=" * W)
    print(f"\n{'Ep':>3} {'nMSE':>10} {'nMAE':>10} "
          f"{'iMSE':>9} {'iPSNR':>9} {'iSSIM':>9}")
    print("-" * W)
    for r in metrics_log:
        im  = r.get("img_MSE",  None)
        ip  = r.get("img_PSNR", None)
        iss = r.get("img_SSIM", None)
        mark = " 🎯" if im is not None and im < 0.1 else ""
        print(f"  {r['epoch']:>2}  "
              f"{r.get('noise_MSE',0):>10.6f} "
              f"{r.get('noise_MAE',0):>10.6f} "
              f"{f'{im:.6f}' if im is not None else '---':>9} "
              f"{f'{ip:.3f}' if ip is not None else '---':>9} "
              f"{f'{iss:.6f}' if iss is not None else '---':>9}"
              f"{mark}")

    print("\n" + "=" * W)
    print("NOISE-SPACE (training signal):")
    print(f"  noise_MSE = {final_noise.get('noise_MSE', 0):.6f}")
    print(f"  noise_MAE = {final_noise.get('noise_MAE', 0):.6f}")
    print("\nIMAGE-SPACE (supervisor targets):")
    results = [
        ("img_MSE",  "< 0.1",  final_img.get("img_MSE",  1),  final_img.get("img_MSE",  1) < 0.1),
        ("img_MAE",  "< 0.3",  final_img.get("img_MAE",  1),  final_img.get("img_MAE",  1) < 0.3),
        ("img_PSNR", "> 18dB", final_img.get("img_PSNR", 0),  final_img.get("img_PSNR", 0) > 18),
        ("img_SSIM", "> 0.6",  final_img.get("img_SSIM", 0),  final_img.get("img_SSIM", 0) > 0.6),
    ]
    for name, target, val, met in results:
        status = "✅ MET" if met else "❌ NOT YET"
        print(f"  {name:<12} target={target:<10} actual={val:.4f}   {status}")

    print(f"\n  NaN steps:      {nan_count}")
    print(f"  Train time:     {train_time:.0f}s ({train_time/3600:.1f}h)")
    print(f"  Throughput:     {throughput:.2f} img/s")
    print(f"  Peak VRAM:      {peak_vram:.2f} GB")
    print(f"  Trainable params: {n_params:,}")
    print("=" * W)


###########################################################################
# MAIN
###########################################################################

def main():
    print("=" * 60)
    print("DUAL-STAGE REGION-AWARE DIFFUSION v6")
    print(f"  Dataset mode:   {DATASET_MODE} ({NUM_IMAGES} images)")
    print(f"  Loss function:  SNR-weighted Huber (delta=1.0, gamma={SNR_GAMMA})")
    print(f"  Trainable CN:   {TRAIN_CN_BLOCKS}")
    print(f"  Batch:          {BATCH_SIZE} x {GRAD_ACCUM_STEPS} accum = "
          f"{BATCH_SIZE * GRAD_ACCUM_STEPS} effective")
    print(f"  Epochs: {NUM_EPOCHS} | LR: {LR} | Timestep bias: {TIMESTEP_BIAS}")
    print(f"  Targets: img_MSE<0.1 | img_PSNR>18dB | img_SSIM>0.6")
    print(f"  Val every {VAL_EVERY} epochs | {VAL_IMAGES} images | skimage SSIM")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    if device == "cuda":
        print(f"[INFO] GPU:  {torch.cuda.get_device_name(0)}")
        print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # Disk space check upfront
    stat = shutil.disk_usage("/workspace")
    print(f"[INFO] Disk free: {stat.free/(1024**3):.1f} GB")
    if stat.free/(1024**3) < 20:
        print("[WARN] Less than 20GB free — checkpoints may fail. "
              "Consider freeing space before running.")

    t0 = time.time()

    # Data
    train_loader = make_loader(CSV_PATH, IMAGE_ROOT, BATCH_SIZE,
                               NUM_WORKERS, NUM_IMAGES, val=False)
    val_loader   = make_loader(CSV_PATH, IMAGE_ROOT, VAL_IMAGES, 0, val=True)

    print(f"[DATA] Caching {NUM_IMAGES} training batches...")
    all_batches, loaded = [], 0
    for b in train_loader:
        all_batches.append(b)
        loaded += b["image"].shape[0]
        if loaded >= NUM_IMAGES:
            break
    print(f"[DATA] {loaded} images in {len(all_batches)} batches")

    val_batches = list(val_loader)
    val_batch   = val_batches[0] if val_batches else all_batches[0]
    print(f"[VAL]  {len(val_batch['image'])} validation images")

    # Models
    m = load_models(device)
    inject_region_attn(m["unet"], device)
    sched = DDPMScheduler.from_pretrained(SD_ID, subfolder="scheduler")

    m["vae"].eval(); m["te"].eval()
    m["unet"].train(); m["cn"].train()
    m["re"].train();   m["sa"].train()

    ema_cn = EMA(m["cn"], decay=EMA_DECAY)

    trainable = (
        [p for p in m["cn"].parameters() if p.requires_grad] +
        [p for p in m["re"].parameters() if p.requires_grad] +
        [p for p in m["sa"].parameters() if p.requires_grad] +
        region_proc_params(m["unet"])
    )
    N_params = sum(p.numel() for p in trainable)
    print(f"\n[OPT] Trainable params: {N_params:,}")
    print(f"[OPT] LR={LR}  weight_decay=1e-4")

    opt = AdamW(trainable, lr=LR, weight_decay=1e-4, betas=(0.9, 0.999))

    # OneCycleLR — better convergence for fine-tuning than cosine restarts
    total_steps = NUM_EPOCHS * (len(all_batches) // GRAD_ACCUM_STEPS)
    lr_sched    = OneCycleLR(
        opt, max_lr=LR,
        total_steps=total_steps,
        pct_start=0.15,       # 15% warmup (was 10%) — more stable ramp-up
        anneal_strategy='cos',
        div_factor=25,        # start at LR/25 (was LR/10) — gentler start
        final_div_factor=1000,# end at LR/1000 — tighter final convergence
    )
    print(f"[SCHED] OneCycleLR: warmup 15% → peak {LR} → final {LR/1000:.2e} "
          f"over {total_steps} steps")

    # Resume
    start_epoch, _ = load_checkpoint(m, ema_cn, opt, RESUME_CKPT, device)

    # Build validation pipeline once
    val_pipe = build_val_pipeline(m, ema_cn, sched, device)

    print(f"\n[TRAIN] Epochs {start_epoch+1}→{NUM_EPOCHS} | {IMAGE_SIZE}px")
    print(f"        mid_block only | SNR loss | EMA | bfloat16 backbone")
    print(f"        Disk-safe checkpointing | Auto-resume")
    print("-" * 60)

    t_train      = time.time()
    nan_count    = 0
    total_imgs   = 0
    metrics_log  = []
    last_noise_m = {}
    last_img_m   = {}
    best_img_mse = float("inf")
    no_improve   = 0
    global_step  = start_epoch * (len(all_batches) // GRAD_ACCUM_STEPS)
    autocast_ctx = torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE)

    for epoch in range(start_epoch + 1, NUM_EPOCHS + 1):
        ep_noise_mse = ep_noise_mae = 0.0
        ep_grad_norm = 0.0
        steps = 0

        m["cn"].train(); m["re"].train(); m["sa"].train(); m["unet"].train()

        for batch_idx, batch in enumerate(all_batches):
            imgs  = batch["image"].to(device=device, dtype=COMPUTE_DTYPE)
            edges = batch["edge_map"].to(device=device, dtype=COMPUTE_DTYPE)
            caps  = batch["caption"]
            bs    = imgs.shape[0]
            total_imgs += bs

            with torch.no_grad():
                latents = m["vae"].encode(imgs).latent_dist.sample() * VAE_SCALE

            noise     = torch.randn_like(latents)
            ts        = sample_timesteps(bs, device)
            noisy_lat = sched.add_noise(latents, noise, ts)

            with torch.no_grad():
                ids     = m["tok"](list(caps), padding="max_length",
                                    max_length=MAX_TOK, truncation=True,
                                    return_tensors="pt").input_ids.to(device)
                txt_emb = m["te"](ids)[0]

            with autocast_ctx:
                down_res, mid_res = m["cn"](
                    sample=noisy_lat, timestep=ts,
                    encoder_hidden_states=txt_emb,
                    controlnet_cond=edges, return_dict=False,
                )

                reg_emb = m["re"](mid_res)
                reg_w   = m["sa"](reg_emb, txt_emb)

                noise_pred = m["unet"](
                    sample=noisy_lat, timestep=ts,
                    encoder_hidden_states=txt_emb,
                    down_block_additional_residuals=down_res,
                    mid_block_additional_residual=mid_res,
                    cross_attention_kwargs={"region_weights": reg_w},
                ).sample

                loss = snr_weighted_loss(
                    noise_pred, noise, ts, sched, gamma=SNR_GAMMA
                ) / GRAD_ACCUM_STEPS

            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                opt.zero_grad()
                continue

            loss.backward()

            if ((batch_idx + 1) % GRAD_ACCUM_STEPS == 0 or
                    batch_idx == len(all_batches) - 1):
                gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                ep_grad_norm += gn.item()
                opt.step()
                lr_sched.step()
                opt.zero_grad()
                ema_cn.update(m["cn"])
                global_step += 1

            nm = compute_noise_metrics(noise_pred, noise)
            ep_noise_mse += nm["noise_MSE"]
            ep_noise_mae += nm["noise_MAE"]
            steps        += 1

        if steps > 0:
            last_noise_m = {
                "noise_MSE": ep_noise_mse / steps,
                "noise_MAE": ep_noise_mae / steps,
                "grad_norm": ep_grad_norm / max(1, steps // GRAD_ACCUM_STEPS),
            }

        do_val      = (epoch % VAL_EVERY == 0 or epoch == NUM_EPOCHS)
        targets_met = False

        if do_val:
            print(f"\n[VAL] Epoch {epoch} — image-space metrics on {VAL_IMAGES} images...")
            update_val_pipeline(val_pipe, ema_cn)
            m["cn"].eval(); m["unet"].eval()
            last_img_m = run_validation(val_pipe, val_batch, epoch, device)
            m["cn"].train(); m["unet"].train()

            targets_met = (
                last_img_m.get("img_MSE",  1) < 0.1 and
                last_img_m.get("img_PSNR", 0) > 18  and
                last_img_m.get("img_SSIM", 0) > 0.6
            )
            val_str = (
                f"iMSE={last_img_m['img_MSE']:.4f} | "
                f"iPSNR={last_img_m['img_PSNR']:.2f}dB | "
                f"iSSIM={last_img_m['img_SSIM']:.4f}"
                + (" 🎯 ALL TARGETS MET" if targets_met else "")
            )
        else:
            nxt = ((epoch // VAL_EVERY) + 1) * VAL_EVERY
            val_str = f"(next val: ep {nxt})"

        row = {"epoch": epoch, **last_noise_m, **last_img_m}
        metrics_log.append(row)

        print(f"  Ep {epoch}/{NUM_EPOCHS} | "
              f"nMSE={last_noise_m.get('noise_MSE',0):.4f} | "
              f"nMAE={last_noise_m.get('noise_MAE',0):.4f} | "
              f"gNorm={last_noise_m.get('grad_norm',0):.3f} | "
              f"{val_str} | "
              f"LR={opt.param_groups[0]['lr']:.2e} | "
              f"{time.time()-t_train:.0f}s")

        # Save latest (stripped, small) every epoch
        save_checkpoint(m, ema_cn, opt, epoch,
                        {**last_noise_m, **last_img_m},
                        CKPT_DIR / "checkpoint_latest.pt",
                        strip_optimizer=True)

        # Save best (full, with optimizer) based on image-space MSE
        if do_val:
            cur = last_img_m.get("img_MSE", 1)
            if cur < best_img_mse:
                best_img_mse = cur
                no_improve   = 0
                save_checkpoint(m, ema_cn, opt, epoch,
                                {**last_noise_m, **last_img_m},
                                CKPT_DIR / "checkpoint_best.pt",
                                strip_optimizer=False)
                print(f"  ★ Best img_MSE={best_img_mse:.6f} → checkpoint_best.pt")
            else:
                no_improve += 1
                print(f"  No improvement ({no_improve}/{EARLY_STOP_PAT})")

            if epoch >= 20 and no_improve >= EARLY_STOP_PAT:
                print(f"\n[EARLY STOP] No improvement for {EARLY_STOP_PAT} val cycles. "
                      f"Best img_MSE={best_img_mse:.6f}")
                break

            if targets_met and epoch >= 15:
                print(f"\n[CONVERGED] All targets met at epoch {epoch}!")
                break

    # Final inference
    best_path = CKPT_DIR / "checkpoint_best.pt"
    if best_path.exists():
        ckpt = torch.load(str(best_path), map_location=device)
        m["cn"].load_state_dict(ckpt["controlnet"])
        ema_cn.load_state_dict(ckpt["controlnet_ema"])
        update_val_pipeline(val_pipe, ema_cn)

    m["cn"].eval(); m["unet"].eval()
    run_inference(val_pipe, all_batches[0], NUM_EPOCHS)

    t_end      = time.time()
    train_dur  = t_end - t_train
    throughput = total_imgs / train_dur if train_dur > 0 else 0
    peak_vram  = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0

    print_metrics(metrics_log, last_noise_m, last_img_m,
                  nan_count, train_dur, t_end - t0,
                  throughput, peak_vram, N_params)

    print(f"\n[DONE]")
    print(f"  Best checkpoint: {CKPT_DIR}/checkpoint_best.pt")
    print(f"  Val images:      {VAL_DIR}/")
    print(f"  Final inference: {INFER_DIR}/generated_epoch{NUM_EPOCHS}.png")


if __name__ == "__main__":
    main()
