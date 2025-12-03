# ================================================================
# Micro–Macro Mosaic Generator
# - Micro-tiles are generated ON-THE-FLY using SDXL (legal)
# - Macro-scene is generated with SDXL
# - No stored datasets, no external images
# ================================================================

import os
import math
import random
import faiss
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import transforms as T
from diffusers import StableDiffusionXLPipeline


# ================================================================
# CONFIG
# ================================================================

OUTPUT_DIR = "data/guided_samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Macro scene (big picture) ----
MACRO_SIZE = 1024              # SDXL generation
GRID_SIZE = 64                
MACRO_TILE_PX = MACRO_SIZE // GRID_SIZE   # = 16 px per macro tile

# ---- Micro-patches ----
MICRO_TILE_PX = 64             # final output tile resolution
MICRO_SAMPLES = 300            # how many SDXL micro-tiles to generate

# ---- Final mosaic resolution ----
FINAL_RES = GRID_SIZE * MICRO_TILE_PX     # 4096 x 4096

# ---- Recoloring ----
RECOLOR_STRENGTH = 0.05       # 15% color tint only
BRIGHTNESS_CLAMP = (0.7, 1.4)  # brightness scaling range

# ---- SDXL Sampling ----
MACRO_STEPS = 35
MICRO_STEPS = 15
GUIDANCE = 5.5

# ---- Prompts ----
MACRO_PROMPTS = [
    "vast alpine mountains with icy peaks, vivid blue sky, crisp sunlight",
    "bright rolling hills under dramatic lighting, vivid greens, bold shadows",
    "majestic rocky mountains at sunrise, golden glow, highly detailed",
]

MICRO_CLASSES = {
    "FACE": [
        "a flat 2D round face icon, thick outline, solid fill, minimal eyes, simple mouth, no shading",
        "a minimal vector face emoji, centered, flat colors, clean edges, plain background",
    ],
    "CAT": [
        "cute emoji-style cat face, glossy, centered",
        "kawaii cat emoji, clean lines",
    ],
    "FLOWER": [
        "simple flower emoji, centered, glossy",
        "kawaii daisy icon, white petals, cute face",
    ],
    "SYMBOL": [
        "bold black icon, centered, high contrast",
        "bold geometric shape icon, crisp edges",
    ]
}

MICRO_CLASS = "FACE"     # change here


# ================================================================
# UTILITIES
# ================================================================

to_tensor = T.ToTensor()

def log(x):
    print(x, flush=True)


# -----------------------------
# Convert RGB → LAB
# -----------------------------
def rgb_to_lab_tensor(x):
    """
    x: tensor 3xHxW in [0,1]
    returns: 3-dim mean LAB feature
    """
    x = x.cpu()                
    img = (x * 255).permute(1, 2, 0).numpy().astype("uint8")
    pil = Image.fromarray(img, mode="RGB")
    lab = np.array(pil.convert("LAB"))
    L = lab[:, :, 0].mean()
    A = lab[:, :, 1].mean()
    B = lab[:, :, 2].mean()
    return torch.tensor([L, A, B], dtype=torch.float32)


# ================================================================
# LOAD SDXL
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

    try:
        pipe.enable_xformers_memory_efficient_attention()
        log(">> xFormers enabled")
    except:
        log(">> xFormers NOT available")

    log(f">> SDXL on {device}, dtype={dtype}")
    return pipe, device


# ================================================================
# 1) Generate micro-tiles on the fly (legal)
# ================================================================

@torch.no_grad()
def generate_micro_tiles(pipe, device, micro_class):
    """
    LEGAL SDXL-ONLY MICRO TILE GENERATOR
    -----------------------------------
    - SDXL cannot generate clean 32×32 images directly.
    - So we generate at 512×512 (clean object)
    - Then we downsample to 32×32 using area interpolation
      (perfect for preserving flat shapes and removing noise).

    Returns:
        tiles: list of 3×MICRO_TILE_PX×MICRO_TILE_PX tensors (CPU)
    """

    prompts = MICRO_CLASSES[micro_class]
    tiles = []

    log(f">> Generating {MICRO_SAMPLES} micro tiles of class {micro_class}")

    for i in range(MICRO_SAMPLES):

        # pick random prompt variant
        prompt = random.choice(prompts)

        # -------------------------------------------------------------
        # 1) High-resolution generation (necessary for clean shapes)
        # -------------------------------------------------------------
        out = pipe(
            prompt,
            height=512,
            width=512,
            num_inference_steps=MICRO_STEPS,
            guidance_scale=3.0,
            output_type="pt",   # returns torch tensor in [0,1]
        )

        big = out.images[0].unsqueeze(0)   # (1,3,512,512)

        # -------------------------------------------------------------
        # 2) Downsample cleanly → 32×32 micro tile
        #    "area" mode prevents aliasing + keeps edges sharp
        # -------------------------------------------------------------
        small = F.interpolate(
            big,
            size=(MICRO_TILE_PX, MICRO_TILE_PX),
            mode="area"
        ).squeeze(0)  # (3, MICRO_TILE_PX, MICRO_TILE_PX)

        # -------------------------------------------------------------
        # 3) Store on CPU
        # -------------------------------------------------------------
        tiles.append(small.cpu())

        if (i + 1) % 50 == 0:
            log(f"   generated {i+1}/{MICRO_SAMPLES} micro tiles")

    return tiles




# ================================================================
# 2) Compute LAB features + index in FAISS
# ================================================================

def build_faiss_index(tiles):
    feats = []
    for t in tiles:
        feats.append(rgb_to_lab_tensor(t))

    feats = torch.stack(feats, dim=0).numpy().astype("float32")

    index = faiss.IndexFlatL2(3)
    index.add(feats)

    return feats, index


# ================================================================
# 3) Generate macro scene
# ================================================================

@torch.no_grad()
def generate_macro(pipe, prompt, device):
    log(f"\n>> Generating macro for prompt:\n{prompt}\n")

    out = pipe(
        prompt,
        height=MACRO_SIZE,
        width=MACRO_SIZE,
        num_inference_steps=MACRO_STEPS,
        guidance_scale=GUIDANCE,
        output_type="pt",
    )
    return out.images[0]   # 3xH x W


# ================================================================
# 4) Mosaic builder (CPU)
# ================================================================

def build_mosaic(macro, tiles, feats, index, micro_class):
    macro = macro.unsqueeze(0)      # 1×3×H×W
    macro = F.interpolate(macro, size=(MACRO_SIZE, MACRO_SIZE), mode="bilinear")
    macro = macro.squeeze(0)

    mosaic = torch.zeros(3, FINAL_RES, FINAL_RES)
    macro_np = macro.cpu().numpy()

    log(f">> Building mosaic {GRID_SIZE}x{GRID_SIZE} → {FINAL_RES}x{FINAL_RES}")

    # dithering offset to avoid grid artifacts
    dither_amp = 0.05

    for gy in range(GRID_SIZE):
        for gx in range(GRID_SIZE):
            y0 = gy * MACRO_TILE_PX
            x0 = gx * MACRO_TILE_PX

            block = macro[:, y0:y0 + MACRO_TILE_PX, x0:x0 + MACRO_TILE_PX]
            lab_feat = rgb_to_lab_tensor(block)

            # dithering
            jitter = torch.randn(3) * dither_amp
            query = (lab_feat + jitter).numpy().astype("float32")

            # nearest neighbor
            D, I = index.search(query.reshape(1, -1), 1)
            idx = int(I[0][0])

            chosen = tiles[idx]

            # brightness-match
            micro_gray = chosen.mean().item() + 1e-6
            macro_gray = block.mean().item() + 1e-6
            scale = macro_gray / micro_gray
            scale = max(BRIGHTNESS_CLAMP[0], min(scale, BRIGHTNESS_CLAMP[1]))

            recolored = (chosen * scale).clamp(0, 1)

            # slight tint
            tint = block.mean(dim=(1,2)).view(3,1,1)
            recolored = recolored*(1-RECOLOR_STRENGTH) + tint*RECOLOR_STRENGTH

            oy = gy * MICRO_TILE_PX
            ox = gx * MICRO_TILE_PX
            mosaic[:, oy:oy+MICRO_TILE_PX, ox:ox+MICRO_TILE_PX] = recolored

        if (gy+1) % 8 == 0:
            log(f"   row {gy+1}/{GRID_SIZE} complete")

    return mosaic


# ================================================================
# MAIN
# ================================================================

def main():

    pipe, device = load_sdxl()

    # === 1) Generate micro patch dataset ===
    tiles = generate_micro_tiles(pipe, device, MICRO_CLASS)
    tiles = [t.cpu() for t in tiles]     

    # === 2) Build FAISS index ===
    feats, index = build_faiss_index(tiles)

    # === 3) Generate mosaics ===
    for i, prompt in enumerate(MACRO_PROMPTS):

        macro = generate_macro(pipe, prompt, device).cpu()

        mosaic = build_mosaic(macro, tiles, feats, index, MICRO_CLASS)

        save_path = os.path.join(OUTPUT_DIR, f"mosaic_{MICRO_CLASS}_{i}.png")
        save_image(mosaic, save_path)
        log(f">> Saved: {save_path}\n")


if __name__ == "__main__":
    main()
