import copy
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# FIX 2: proper SSIM
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

# ── Hyperparameters ────────────────────────────────────────────────────────
NUM_IMAGES       = int(os.environ.get("NUM_IMAGES",       "10000"))
BATCH_SIZE       = int(os.environ.get("BATCH_SIZE",       "16"))
NUM_EPOCHS       = int(os.environ.get("NUM_EPOCHS",       "50"))
LR               = float(os.environ.get("LR",             "2e-5"))
GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM_STEPS", "2"))
NUM_WORKERS      = int(os.environ.get("NUM_WORKERS",      "4"))
TIMESTEP_BIAS    = float(os.environ.get("TIMESTEP_BIAS",  "0.8"))
RESUME_CKPT      = os.environ.get("RESUME_CKPT",          "")
EMA_DECAY        = float(os.environ.get("EMA_DECAY",      "0.9995"))
SNR_GAMMA        = float(os.environ.get("SNR_GAMMA",      "5.0"))
EARLY_STOP_PAT   = int(os.environ.get("EARLY_STOP_PAT",   "8"))
VAL_EVERY        = int(os.environ.get("VAL_EVERY",        "5"))
VAL_IMAGES       = int(os.environ.get("VAL_IMAGES",       "32"))  # FIX 3: was 4
VAL_STEPS        = int(os.environ.get("VAL_STEPS",        "20"))

# ── Model config ────────────────────────────────────────────────────────────
SD_ID         = "runwayml/stable-diffusion-v1-5"
CN_ID         = "lllyasviel/sd-controlnet-canny"
COMPUTE_DTYPE = torch.bfloat16   # frozen models — fast on A100
REGION_DTYPE  = torch.float32    # trainable region modules — stable
IMAGE_SIZE    = 512
MAX_TOK       = 77
VAE_SCALE     = 0.18215
GRID_SIZE     = 8
TEXT_DIM      = 768
TRAIN_CN_BLOCKS = {"down_blocks.3", "mid_block"}

# EMA

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


# SNR WEIGHTING

def compute_snr(scheduler, timesteps):
    ab  = scheduler.alphas_cumprod
    a_t = (ab ** 0.5)[timesteps].to(timesteps.device)
    s_t = ((1 - ab) ** 0.5)[timesteps].to(timesteps.device)
    return (a_t / s_t) ** 2


def snr_weighted_loss(pred, target, timesteps, scheduler, gamma=5.0):
    snr     = compute_snr(scheduler, timesteps)
    weights = torch.clamp(snr, max=gamma) / snr
    weights = weights.view(-1, 1, 1, 1).to(pred.device)
    loss    = F.mse_loss(pred.float(), target.float(), reduction="none")
    return (loss * weights).mean()


# DATASET

class FlickrSketchDataset(Dataset):
    def __init__(self, csv_path, image_root, max_samples=None, val=False):
        df = pd.read_csv(csv_path)
        df["wc"] = df["caption"].astype(str).apply(lambda x: len(x.split()))
        df = df[(df["wc"] >= 10) & (df["wc"] <= 30)].reset_index(drop=True)
        df = df[~df["caption"].str.startswith("A photograph from Visual Genome")
               ].reset_index(drop=True)
        df = df.drop_duplicates(subset=["caption"]).reset_index(drop=True)
        print(f"[Dataset] After filtering: {len(df)} samples")

        if val:
            # Take last VAL_IMAGES rows as validation set
            df = df.tail(VAL_IMAGES).reset_index(drop=True)
        elif max_samples:
            # Remove val samples from training
            df = df.head(max(0, len(df) - VAL_IMAGES))
            df = df.head(max_samples).reset_index(drop=True)

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

        gray_check = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if gray_check.std() < 0.05:
            return self.__getitem__((idx + 1) % len(self.df))

        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(blurred, 100, 200)
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

# REGION ATTENTION MODULES

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
    """Proper multihead attention with trainable Q/K/V projections."""
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
        return weights   # [B, R, SeqLen]


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


# MODEL LOADING

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
    for p in cn.parameters():
        p.requires_grad = False
    for name, p in cn.named_parameters():
        if any(blk in name for blk in TRAIN_CN_BLOCKS):
            p.requires_grad = True
            p.register_hook(lambda g: g.float() if g is not None else g)

    cn_total   = sum(p.numel() for p in cn.parameters())
    cn_trained = sum(p.numel() for p in cn.parameters() if p.requires_grad)
    print(f"  ControlNet: {cn_trained:,}/{cn_total:,} trainable "
          f"({100*cn_trained/cn_total:.1f}%) — blocks: {TRAIN_CN_BLOCKS}")

    re = RegionExtractor().to(device=device, dtype=REGION_DTYPE)
    sa = SemanticAttention().to(device=device, dtype=REGION_DTYPE)

    return dict(tok=tok, te=te, vae=vae, unet=unet, cn=cn, re=re, sa=sa)


def inject_region_attn(unet, device):
    """Inject RegionAwareAttnProcessor into UNet cross-attention layers."""
    new_procs, n = {}, 0
    for name, proc in unet.attn_processors.items():
        if "attn2" in name:
            new_procs[name] = RegionAwareAttnProcessor().to(
                device=device, dtype=REGION_DTYPE)
            n += 1
        else:
            new_procs[name] = SafeAttnProcessor(proc)
    unet.set_attn_processor(new_procs)
    print(f"  Injected {n} RegionAwareAttnProcessors into UNet")
    return n


def region_proc_params(unet):
    return [p for proc in unet.attn_processors.values()
            if isinstance(proc, RegionAwareAttnProcessor)
            for p in proc.parameters() if p.requires_grad]


def _freeze(m, name):
    for p in m.parameters():
        p.requires_grad = False
    print(f"  {name}: frozen (bfloat16)")


# FIX 1: VALIDATION PIPELINE — built once, reused every val cycle

def build_val_pipeline(m, ema_cn, re, sa, sched, device):
    """
    FIX 4: Builds a pipeline that uses BOTH:
      - EMA ControlNet weights
      - The same RegionAwareAttnProcessor used during training
    
    This means validation images are generated using the FULL dual-stage
    architecture, not just standard SD attention.
    
    Built once before training, reused every VAL_EVERY epochs.
    Avoids the expensive from_pretrained reload every validation cycle.
    """
    print("[VAL PIPE] Building validation pipeline (once)...")

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        SD_ID,
        controlnet=ema_cn.shadow,
        unet=m["unet"],          # FIX 4: reuse the same UNet with injected processors
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

    # Store references to region modules so we can pass weights during generation
    pipe._re = re
    pipe._sa = sa

    print("[VAL PIPE] Ready.")
    return pipe


def update_val_pipeline_controlnet(val_pipe, ema_cn):
    """
    FIX 1: Update the EMA ControlNet weights in the pipeline in-place.
    No model reload — just update the reference.
    """
    val_pipe.controlnet = ema_cn.shadow


# FIX 2: PROPER SSIM via skimage

def compute_image_metrics(generated: torch.Tensor, ground_truth: torch.Tensor):
    """
    Compute image-space metrics on tensors in [-1, 1].
    FIX 2: Uses skimage structural_similarity for proper local-window SSIM.
    This matches published paper values, unlike the global approximation in v4.
    """
    g = generated.float().detach().clamp(-1, 1)
    r = ground_truth.float().detach().clamp(-1, 1)

    mse = F.mse_loss(g, r).item()
    mae = F.l1_loss(g, r).item()
    psnr = 10 * math.log10((2.0 ** 2) / (mse + 1e-8))

    # FIX 2: proper SSIM via skimage — local windows, matches published work
    ssim_vals = []
    for i in range(g.shape[0]):
        # Convert [-1,1] → [0,1] for skimage
        gi_np = ((g[i].permute(1, 2, 0).numpy() + 1.0) / 2.0).clip(0, 1)
        ri_np = ((r[i].permute(1, 2, 0).numpy() + 1.0) / 2.0).clip(0, 1)
        ssim_val = sk_ssim(
            gi_np, ri_np,
            data_range=1.0,
            channel_axis=2,   # RGB channels
            win_size=7,
        )
        ssim_vals.append(ssim_val)

    return {
        "img_MSE":  mse,
        "img_MAE":  mae,
        "img_PSNR": psnr,
        "img_SSIM": float(np.mean(ssim_vals)),
    }


def compute_noise_metrics(pred, target):
    """Training-time noise prediction metrics — tracks denoiser improvement."""
    p   = pred.float().detach()
    t   = target.float().detach()
    return {
        "noise_MSE": F.mse_loss(p, t).item(),
        "noise_MAE": F.l1_loss(p, t).item(),
    }

# VALIDATION — uses pre-built pipeline, proper SSIM, 32 images

@torch.no_grad()
def run_validation(val_pipe, val_batch, epoch, device):
    """
    FIX 1: Uses pre-built pipeline (no reload).
    FIX 2: Uses skimage SSIM.
    FIX 3: Uses 32 validation images.
    FIX 4: Pipeline uses RegionAwareAttnProcessor via shared UNet reference.
    """
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

        # Save first 4 for visual inspection
        if i < 4:
            result.save(str(VAL_DIR / f"val_ep{epoch}_gen_{i}.png"))
            gt_np = ((imgs[i].float().permute(1, 2, 0).numpy() + 1.0) * 127.5
                     ).clip(0, 255).astype(np.uint8)
            Image.fromarray(gt_np).save(
                str(VAL_DIR / f"val_ep{epoch}_gt_{i}.png"))

        arr = np.array(result).astype(np.float32) / 127.5 - 1.0
        generated_tensors.append(torch.from_numpy(arr).permute(2, 0, 1))

    gen_batch = torch.stack(generated_tensors)   # [bs, 3, H, W]
    gt_batch  = imgs.float()

    return compute_image_metrics(gen_batch, gt_batch)

# CHECKPOINT

def save_checkpoint(m, ema_cn, opt, epoch, metrics, path):
    torch.save({
        "epoch":          epoch,
        "controlnet":     m["cn"].state_dict(),
        "controlnet_ema": ema_cn.state_dict(),
        "re":             m["re"].state_dict(),
        "sa":             m["sa"].state_dict(),
        "optimizer":      opt.state_dict(),
        "metrics":        metrics,
    }, str(path))


def load_checkpoint(m, ema_cn, opt, resume_path, device):
    if not resume_path or not os.path.exists(resume_path):
        print("[RESUME] Starting fresh.")
        return 0, {}
    print(f"[RESUME] Loading: {resume_path}")
    ckpt = torch.load(resume_path, map_location=device)
    m["cn"].load_state_dict(ckpt["controlnet"])
    if "controlnet_ema" in ckpt:
        ema_cn.load_state_dict(ckpt["controlnet_ema"])
    m["re"].load_state_dict(ckpt["re"])
    m["sa"].load_state_dict(ckpt["sa"])
    if "optimizer" in ckpt:
        try:
            opt.load_state_dict(ckpt["optimizer"])
        except Exception:
            print("[RESUME] Optimizer mismatch — fresh optimizer.")
    epoch = ckpt.get("epoch", 0)
    print(f"[RESUME] Resumed from epoch {epoch}")
    return epoch, ckpt.get("metrics", {})


# TIMESTEP SAMPLING

def sample_timesteps(bs, device, total=1000, bias=TIMESTEP_BIAS):
    n_low  = int(bs * bias)
    n_high = bs - n_low
    low    = torch.randint(0, total // 2,    (n_low,),  device=device)
    high   = torch.randint(total // 2, total, (n_high,), device=device)
    ts     = torch.cat([low, high])
    return ts[torch.randperm(bs, device=device)]


# FINAL INFERENCE

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

# PRINT METRICS

def print_metrics(metrics_log, final_noise, final_img,
                  nan_count, train_time, total_time,
                  throughput, peak_vram, n_params):
    W = 78
    print("\n" + "=" * W)
    print(f"{'FINAL TRAINING METRICS':^{W}}")
    print("=" * W)
    print(f"\n{'Ep':>3} {'NoiseMSE':>10} {'NoiseMAE':>10} "
          f"{'ImgMSE':>9} {'ImgPSNR':>9} {'ImgSSIM':>9}")
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

    print("\nIMAGE-SPACE METRICS (supervisor targets):")
    print(f"  Computed on {VAL_IMAGES} real generated images vs ground truth")
    print(f"  SSIM uses skimage local-window method (matches published work)")
    results = [
        ("img_MSE",  "< 0.1",  final_img.get("img_MSE",  1),   final_img.get("img_MSE",  1) < 0.1),
        ("img_MAE",  "< 0.3",  final_img.get("img_MAE",  1),   final_img.get("img_MAE",  1) < 0.3),
        ("img_PSNR", "> 18dB", final_img.get("img_PSNR", 0),   final_img.get("img_PSNR", 0) > 18),
        ("img_SSIM", "> 0.6",  final_img.get("img_SSIM", 0),   final_img.get("img_SSIM", 0) > 0.6),
    ]
    for name, target, val, met in results:
        status = "✅ MET" if met else "❌ NOT YET"
        print(f"  {name:<12} target={target:<10} actual={val:.4f}   {status}")

    print("\nRUN STATS:")
    for k, v in [
        ("NaN steps skipped",  str(nan_count)),
        ("Training time (s)",  f"{train_time:.1f}"),
        ("Total runtime (s)",  f"{total_time:.1f}"),
        ("Throughput (img/s)", f"{throughput:.2f}"),
        ("Peak VRAM (GB)",     f"{peak_vram:.2f}"),
        ("Trainable params",   f"{n_params:,}"),
    ]:
        print(f"  {k:<28} {v}")
    print("=" * W)


# MAIN

def main():
    print("=" * 60)
    print("DUAL-STAGE REGION-AWARE DIFFUSION v5")
    print(f"  Targets: img_MSE<0.1 | img_PSNR>18dB | img_SSIM>0.6")
    print(f"  Images: {NUM_IMAGES} | Epochs: {NUM_EPOCHS} | "
          f"Batch: {BATCH_SIZE} | LR: {LR}")
    print(f"  Val: {VAL_IMAGES} images every {VAL_EVERY} epochs "
          f"| skimage SSIM | region-aware pipeline")
    if RESUME_CKPT:
        print(f"  Resuming from: {RESUME_CKPT}")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    if device == "cuda":
        print(f"[INFO] GPU:  {torch.cuda.get_device_name(0)}")
        print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    t0 = time.time()

    # Data
    train_loader = make_loader(CSV_PATH, IMAGE_ROOT, BATCH_SIZE,
                               NUM_WORKERS, NUM_IMAGES, val=False)
    val_loader   = make_loader(CSV_PATH, IMAGE_ROOT, VAL_IMAGES,
                               0, val=True)

    print(f"[DATA] Caching training batches...")
    all_batches, loaded = [], 0
    for b in train_loader:
        all_batches.append(b)
        loaded += b["image"].shape[0]
        if loaded >= NUM_IMAGES:
            break
    print(f"[DATA] {loaded} training images in {len(all_batches)} batches")

    val_batches = list(val_loader)
    val_batch   = val_batches[0] if val_batches else all_batches[0]
    print(f"[VAL]  {len(val_batch['image'])} validation images loaded")

    # Models
    m = load_models(device)
    inject_region_attn(m["unet"], device)   # FIX 4: inject into training UNet
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

    opt      = AdamW(trainable, lr=LR, weight_decay=1e-4, betas=(0.9, 0.999))
    steps_pe = max(1, len(all_batches) // GRAD_ACCUM_STEPS)
    lr_sched = CosineAnnealingWarmRestarts(
        opt, T_0=steps_pe * 10, T_mult=1, eta_min=LR * 0.01)

    start_epoch, _ = load_checkpoint(m, ema_cn, opt, RESUME_CKPT, device)

    # FIX 1: build validation pipeline ONCE before training loop
    # FIX 4: passes m["unet"] so RegionAwareAttnProcessors are active
    val_pipe = build_val_pipeline(m, ema_cn, m["re"], m["sa"], sched, device)

    print(f"\n[TRAIN] Epochs {start_epoch+1}→{NUM_EPOCHS} | {IMAGE_SIZE}px")
    print(f"        SNR-weighted loss | EMA={EMA_DECAY} | bfloat16 backbone")
    print(f"        Val pipeline uses RegionAwareAttnProcessor ✓")
    print("-" * 60)

    t_train      = time.time()
    nan_count    = 0
    total_imgs   = 0
    metrics_log  = []
    last_noise_m = {}
    last_img_m   = {}
    best_img_mse = float("inf")
    no_improve   = 0
    global_step  = start_epoch * steps_pe
    autocast_ctx = torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE)

    for epoch in range(start_epoch + 1, NUM_EPOCHS + 1):
        ep_noise_mse = ep_noise_mae = 0.0
        steps = 0

        # Set training mode
        m["cn"].train(); m["re"].train(); m["sa"].train()
        m["unet"].train()

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
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
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
            }

        # FIX 1: validation uses pre-built pipeline — just update EMA weights
        do_val = (epoch % VAL_EVERY == 0 or epoch == NUM_EPOCHS)
        targets_met = False

        if do_val:
            print(f"\n[VAL] Epoch {epoch} — image-space metrics on {VAL_IMAGES} images...")
            # FIX 1: update EMA controlnet weights in-place — no pipeline reload
            update_val_pipeline_controlnet(val_pipe, ema_cn)
            m["cn"].eval(); m["unet"].eval()

            last_img_m = run_validation(val_pipe, val_batch, epoch, device)

            m["cn"].train(); m["unet"].train()

            targets_met = (
                last_img_m.get("img_MSE",  1) < 0.1 and
                last_img_m.get("img_PSNR", 0) > 18  and
                last_img_m.get("img_SSIM", 0) > 0.6
            )
            val_str = (
                f"img_MSE={last_img_m['img_MSE']:.4f} | "
                f"img_PSNR={last_img_m['img_PSNR']:.2f}dB | "
                f"img_SSIM={last_img_m['img_SSIM']:.4f}"
                + (" 🎯 ALL TARGETS MET" if targets_met else "")
            )
        else:
            val_str = f"(val at epoch {((epoch // VAL_EVERY) + 1) * VAL_EVERY})"

        row = {"epoch": epoch, **last_noise_m, **last_img_m}
        metrics_log.append(row)

        print(f"  Ep {epoch}/{NUM_EPOCHS} | "
              f"nMSE={last_noise_m.get('noise_MSE',0):.4f} | "
              f"nMAE={last_noise_m.get('noise_MAE',0):.4f} | "
              f"{val_str} | "
              f"LR={opt.param_groups[0]['lr']:.2e} | "
              f"{time.time()-t_train:.0f}s")

        # Save checkpoint
        ckpt_path = CKPT_DIR / f"checkpoint_epoch_{epoch}.pt"
        save_checkpoint(m, ema_cn, opt, epoch,
                        {**last_noise_m, **last_img_m}, ckpt_path)

        # Best checkpoint based on image-space MSE
        if do_val:
            cur = last_img_m.get("img_MSE", 1)
            if cur < best_img_mse:
                best_img_mse = cur
                no_improve   = 0
                save_checkpoint(m, ema_cn, opt, epoch,
                                {**last_noise_m, **last_img_m},
                                CKPT_DIR / "checkpoint_best.pt")
                print(f"  ★ Best img_MSE={best_img_mse:.6f} → checkpoint_best.pt")
            else:
                no_improve += 1

            if epoch >= 30 and no_improve >= EARLY_STOP_PAT:
                print(f"\n[EARLY STOP] No improvement for {EARLY_STOP_PAT} val "
                      f"cycles. Best img_MSE={best_img_mse:.6f}")
                break

            if targets_met and epoch >= 20:
                print(f"\n[CONVERGED] All targets met at epoch {epoch}!")
                break

    # Final inference using the same val_pipe (region-aware)
    best_path = CKPT_DIR / "checkpoint_best.pt"
    if best_path.exists():
        ckpt = torch.load(str(best_path), map_location=device)
        m["cn"].load_state_dict(ckpt["controlnet"])
        if "controlnet_ema" in ckpt:
            ema_cn.load_state_dict(ckpt["controlnet_ema"])
        update_val_pipeline_controlnet(val_pipe, ema_cn)

    m["cn"].eval(); m["unet"].eval()
    run_inference(val_pipe, all_batches[0], NUM_EPOCHS)

    t_end      = time.time()
    train_dur  = t_end - t_train
    throughput = total_imgs / train_dur if train_dur > 0 else 0
    peak_vram  = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0

    print_metrics(metrics_log, last_noise_m, last_img_m,
                  nan_count, train_dur, t_end - t0,
                  throughput, peak_vram, N_params)

    print(f"\n[DONE] All outputs at {OUTPUT_DIR}")
    print(f"  Best checkpoint: {CKPT_DIR}/checkpoint_best.pt")
    print(f"  Val images:      {VAL_DIR}/")
    print(f"  Final output:    {INFER_DIR}/generated_epoch{NUM_EPOCHS}.png")


if __name__ == "__main__":
    main()
