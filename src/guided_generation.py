# src/guided_generation.py

import os
import glob
import math
import random

import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.utils import save_image
from diffusers import StableDiffusionXLPipeline
from PIL import Image

# ================================================================
# CONFIG
# ================================================================

OUTPUT_DIR = "data/guided_samples"
PATCH_DATASET_ROOT = "data/patch_dataset"  # where FACE/CAT/... folders live
os.makedirs(OUTPUT_DIR, exist_ok=True)



# We generate a 1024×1024 macro and use a 128×128 grid over it.
# Each macro-tile: 8×8, each micro-tile: 32×32 → final 4096×4096
MACRO_SIZE = 1024                 # SDXL image size
GRID_SIZE = 128                   # 128x128 tiles = 16384 micro-objects
MACRO_TILE_PX = MACRO_SIZE // GRID_SIZE  # 1024 / 128 = 8 pixels per macro tile

# ---------------- MICRO (PATCH) ----------------
MICRO_TILE_PX = 32                # each micro-object is 32x32
FINAL_RES = GRID_SIZE * MICRO_TILE_PX  # 128 * 32 = 4096

# ---------------- RECOLORING ----------------
# Strong recolor using AdaIN-like matching.
# 0.0 → original tile colors, 1.0 → fully recolored to macro tile stats
RECOLOR_STRENGTH = 1.0  # "strong recolor"

# ---------------- SDXL SAMPLING ----------------
NUM_INFERENCE_STEPS = 35
GUIDANCE_SCALE = 6.0

# ---------------- HOW MANY POSTERS ----------------
NUM_SAMPLES = 4


# choose which micro class to use: "FACE", "CAT", "FLOWER", "SYMBOL"
MICRO_CLASS = "SYMBOL"        # <<< change to CAT/FLOWER/SYMBOL as you like


# ================================================================
# UTILS
# ================================================================

to_tensor = T.ToTensor()     # PIL -> [0,1] tensor


def log(msg: str):
    print(msg, flush=True)


# ================================================================
# LOAD SDXL (simple, safe)
# ================================================================

def load_sdxl():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        dtype=dtype,
        use_safetensors=True,
    ).to(device)

    pipe.safety_checker = None

    # xFormers is optional; if not installed, just ignore
    try:
        pipe.enable_xformers_memory_efficient_attention()
        log(">> xFormers enabled.")
    except Exception as e:
        log(f">> xFormers NOT available or failed: {e}")

    log(f">> SDXL on {device}, dtype={dtype}")
    return pipe, device


# ================================================================
# 1) GENERATE MACRO SCENE
# ================================================================

@torch.no_grad()
def generate_macro_scene(pipe, prompt: str):
    """
    Generate a 1×3×MACRO_SIZE×MACRO_SIZE tensor in [0,1] using SDXL.
    """
    log(f"\n>> Prompt: {prompt}")
    out = pipe(
        prompt,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        height=MACRO_SIZE,
        width=MACRO_SIZE,
        output_type="pt",   # returns tensor in [0,1]
    )
    img = out.images[0].unsqueeze(0)  # (1,3,H,W)
    return img


# ================================================================
# 2) LOAD MICRO PATCH DATASET
# ================================================================

def load_micro_patches(micro_class: str):
    """
    Loads all PNG/JPG images from data/patch_dataset/<micro_class>,
    returns:
        patches: list[Tensor 3×MICRO_TILE_PX×MICRO_TILE_PX in 0..1]
        feats:   Tensor N×4 (mean R, G, B, brightness)
    """
    folder = os.path.join(PATCH_DATASET_ROOT, micro_class)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Micro class folder not found: {folder}")

    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, ext)))

    if not paths:
        raise RuntimeError(f"No images found in {folder}")

    log(f">> Loading {len(paths)} micro patches from {folder}")

    patches = []
    feats_list = []

    for p in paths:
        img = to_tensor(Image.open(p).convert("RGB"))

        # ensure consistent size for tiles
        if img.shape[1] != MICRO_TILE_PX or img.shape[2] != MICRO_TILE_PX:
            img = F.interpolate(
                img.unsqueeze(0),
                size=(MICRO_TILE_PX, MICRO_TILE_PX),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        patches.append(img)

        # simple 4D feature: mean R,G,B and brightness
        mean_rgb = img.view(3, -1).mean(dim=1)  # 3
        brightness = mean_rgb.mean().unsqueeze(0)
        feat = torch.cat([mean_rgb, brightness], dim=0)  # 4
        feats_list.append(feat)

    feats = torch.stack(feats_list, dim=0)  # (N,4)
    return patches, feats  # list of 3×H×W, Tensor N×4


# ================================================================
# 3) STRONG RECOLOR (AdaIN-style)
# ================================================================

def adain_recolor(tile: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    AdaIN-style recolor:
    tile:   3×H×W  micro-object
    target: 3×H×W  macro tile (rescaled to same size beforehand)
    """
    tile_flat = tile.view(3, -1)
    ref_flat = target.view(3, -1)

    mean_tile = tile_flat.mean(dim=1, keepdim=True)
    mean_ref = ref_flat.mean(dim=1, keepdim=True)

    std_tile = tile_flat.std(dim=1, keepdim=True) + 1e-6
    std_ref = ref_flat.std(dim=1, keepdim=True) + 1e-6

    out = (tile_flat - mean_tile) * (std_ref / std_tile) + mean_ref
    out = out.view_as(tile)
    return out.clamp(0.0, 1.0)


# ================================================================
# 4) BUILD MOSAIC
# ================================================================

def build_mosaic(macro_img: torch.Tensor,
                 micro_patches,
                 micro_feats: torch.Tensor,
                 top_k: int = 1):
    """
    macro_img: 1×3×H×W in [0,1]
    micro_patches: list of 3×H×W tensors (CPU)
    micro_feats: N×4 (meanRGB+brightness) on CPU
    top_k: choose a random patch among the best K matches (prevents patterns)
    """

    # ---- prepare macro image ----
    macro = macro_img.squeeze(0)  # 3×H×W
    macro = F.interpolate(
        macro.unsqueeze(0),
        size=(MACRO_SIZE, MACRO_SIZE),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    N = micro_feats.shape[0]
    out_H = GRID_SIZE * MICRO_TILE_PX
    out_W = GRID_SIZE * MICRO_TILE_PX
    mosaic = torch.zeros(3, out_H, out_W)

    log(f">> Building mosaic: {GRID_SIZE}x{GRID_SIZE} → {out_H}x{out_W}")

    for gy in range(GRID_SIZE):
        for gx in range(GRID_SIZE):

            # ---- extract macro tile ----
            y0 = gy * MACRO_TILE_PX
            x0 = gx * MACRO_TILE_PX
            tile = macro[:, y0:y0 + MACRO_TILE_PX, x0:x0 + MACRO_TILE_PX]

            tile_flat = tile.reshape(3, -1)
            mean_rgb = tile_flat.mean(dim=1)      # (3,)
            brightness = mean_rgb.mean()          # scalar
            tile_feat = torch.tensor([mean_rgb[0], mean_rgb[1], mean_rgb[2], brightness])

            # ---- compute distance to all micro patch features ----
            diff = micro_feats - tile_feat.unsqueeze(0)
            dist = (diff ** 2).sum(dim=1)

            # ---- pick best patch (or top-k random) ----
            if top_k <= 1:
                idx = int(torch.argmin(dist).item())
            else:
                topk = torch.topk(dist, k=top_k, largest=False).indices
                idx = int(random.choice(topk).item())

            micro = micro_patches[idx]  # 3×H×W

            # ======================================================================
            # ⭐ NEW BRIGHTNESS-PRESERVING RECOLOR (does NOT destroy face structure)
            # ======================================================================

            # brightness of micro patch
            micro_gray = micro.mean().item() + 1e-6
            tile_gray = brightness.item() + 1e-6

            # scale factor to match brightness
            scale = tile_gray / micro_gray

            # keep faces visible — clamp scaling
            scale = max(0.7, min(scale, 1.4))

            # apply brightness scaling only (no heavy recolor)
            recolored = (micro * scale).clamp(0, 1)

            # VERY LIGHT color tint (optional)
            tint_strength = 0.10  # 10% color tint – tiny and safe
            recolored = recolored * (1 - tint_strength) + mean_rgb.view(3,1,1) * tint_strength
            recolored = recolored.clamp(0, 1)

            # ---- paste into mosaic ----
            oy = gy * MICRO_TILE_PX
            ox = gx * MICRO_TILE_PX
            mosaic[:, oy:oy+MICRO_TILE_PX, ox:ox+MICRO_TILE_PX] = recolored

        if (gy + 1) % 8 == 0:
            log(f"   row {gy+1}/{GRID_SIZE} complete")

    return mosaic


# ================================================================
# MAIN
# ================================================================

def main():
    pipe, device = load_sdxl()

    micro_patches, micro_feats = load_micro_patches(MICRO_CLASS)
    micro_feats = micro_feats.cpu()
    micro_patches = [p.cpu() for p in micro_patches]

    MACRO_PROMPTS = [
        "vast alpine mountains with icy peaks, vivid blue sky, crisp sunlight",
        "bright green rolling hills, high contrast, bold shadows",
        "majestic rocky mountains at sunrise, golden hour glow, dramatic",
        "deep blue fjord mountains with reflections, extremely vivid",
        "towering desert canyon cliffs, bold warm colors, hard sunlight",
    ]

    log(f"\n### GENERATING MOSAICS ({len(MACRO_PROMPTS)} prompts) ###\n")

    for i, prompt in enumerate(MACRO_PROMPTS):
        log(f"\n>> Generating mosaic for prompt {i+1}/{len(MACRO_PROMPTS)}")
        log(f">> Prompt: {prompt}")

        # 1. generate macro scene (single prompt)
        macro = generate_macro_scene(pipe, prompt).cpu()

        # 2. build mosaic
        mosaic = build_mosaic(macro, micro_patches, micro_feats)

        # 3. save
        save_path = os.path.join(
            OUTPUT_DIR, f"mosaic_{MICRO_CLASS}_{i}.png"
        )
        save_image(mosaic, save_path)
        log(f">> Saved: {save_path}\n")




if __name__ == "__main__":
    main()
