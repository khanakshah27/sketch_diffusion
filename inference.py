import argparse
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from transformers import CLIPTokenizer

SD_ID    = "runwayml/stable-diffusion-v1-5"
CN_ID    = "lllyasviel/sd-controlnet-canny"
DTYPE    = torch.float32
IMG_SIZE = 512
GRID     = 8
TEXT_DIM = 768


class RegionExtractor(nn.Module):
    def __init__(self, in_ch=1280, embed_dim=TEXT_DIM, grid=GRID):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((grid, grid))
        self.proj = nn.Linear(in_ch, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x.float()
        x = self.pool(x).flatten(2).permute(0, 2, 1)
        return self.norm(self.proj(x))


class SemanticAttention(nn.Module):
    def __init__(self, dim=TEXT_DIM):
        super().__init__()
        self.scale = math.sqrt(dim)

    def forward(self, regions, text):
        q = regions.float()
        k = text.float()
        return F.softmax(torch.matmul(q, k.transpose(-1, -2)) / self.scale, dim=-1)


def make_sketch(image_path):
    img     = cv2.imread(image_path)
    img     = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 100, 200)
    return Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--prompt",     required=True)
    parser.add_argument("--output",     default="generated.png")
    parser.add_argument("--steps",      type=int, default=30)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    # Load ControlNet with checkpoint
    cn = ControlNetModel.from_pretrained(CN_ID, torch_dtype=DTYPE).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    cn.load_state_dict(ckpt["controlnet"])
    cn.eval()
    print(f"[INFO] Loaded checkpoint: {args.checkpoint}")

    # Load pipeline
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        SD_ID, controlnet=cn, torch_dtype=DTYPE, safety_checker=None,
    ).to(device)
    pipe.set_progress_bar_config(disable=False)

    # Generate sketch
    sketch = make_sketch(args.image_path)
    print(f"[INFO] Prompt: {args.prompt}")

    with torch.no_grad():
        result = pipe(
            prompt=args.prompt,
            image=sketch,
            num_inference_steps=args.steps,
            guidance_scale=7.5,
            height=IMG_SIZE,
            width=IMG_SIZE,
        ).images[0]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    result.save(args.output)
    sketch.save(args.output.replace(".png", "_sketch.png"))
    print(f"[DONE] Saved → {args.output}")


if __name__ == "__main__":
    main()
