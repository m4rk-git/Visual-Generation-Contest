# ================================================================
# Micro–Macro Mosaic Generator
# - Micro-tiles are generated ON-THE-FLY using SDXL (legal)
# - Macro-scene is generated with SDXL
# - No stored datasets, no external images
# ================================================================

import os
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
MACRO_SIZE = 1024              # SDXL generation size (H=W=1024)
GRID_SIZE = 64                 # 64 x 64 = 4096 tiles
MACRO_TILE_PX = MACRO_SIZE // GRID_SIZE   # = 16 px per macro tile

# ---- Micro-patches (tiny images that form the mosaic) ----
MICRO_TILE_PX = 64             # each tile in final mosaic is 64x64
MICRO_SAMPLES = 300            # how many SDXL micro-tiles to generate

# ---- Final mosaic resolution ----
FINAL_RES = GRID_SIZE * MICRO_TILE_PX     # 64 * 64 = 4096 x 4096

# ---- Recoloring ----
RECOLOR_STRENGTH = 0.05        # 5% color tint only
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
    ],
}

MICRO_CLASS = "FACE"     # <<< change here for CAT / FLOWER / SYMBOL


# ================================================================
# UTILITIES
# ================================================================

to_tensor = T.ToTensor()


def log(x: str):
    print(x, flush=True)


# -----------------------------
# Convert RGB → LAB (mean color)
# -----------------------------
def rgb_to_lab_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    x: tensor 3xHxW in [0,1] (CPU or CUDA)
    returns: 3-dim mean LAB feature (float32 on CPU)
    """
    x = x.detach().cpu()
    img = (x * 255).permute(1, 2, 0).numpy().astype("uint8")
    pil = Image.fromarray(img, mode="RGB")
    lab = np.array(pil.convert("LAB"))  # H x W x 3

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
    except Exception:
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
    - SDXL produces higher quality at larger resolutions.
    - We generate at 512×512, then downsample to MICRO_TILE_PX×MICRO_TILE_PX
      using area interpolation (good for icon-like shapes).

    Returns:
        tiles: list of 3×MICRO_TILE_PX×MICRO_TILE_PX tensors on CPU in [0,1]
    """

    prompts = MICRO_CLASSES[micro_class]
    tiles = []

    log(f">> Generating {MICRO_SAMPLES} micro tiles of class {micro_class}")

    for i in range(MICRO_SAMPLES):
        prompt = random.choice(prompts)

        out = pipe(
            prompt,
            height=512,
            width=512,
            num_inference_steps=MICRO_STEPS,
            guidance_scale=3.0,
            output_type="pt",   # torch tensor in [0,1]
        )

        big = out.images[0].unsqueeze(0)   # (1,3,512,512)

        small = F.interpolate(
            big,
            size=(MICRO_TILE_PX, MICRO_TILE_PX),
            mode="area"                    # good for downsampling
        ).squeeze(0)                        # (3, MICRO_TILE_PX, MICRO_TILE_PX)

        tiles.append(small.cpu())

        if (i + 1) % 50 == 0:
            log(f"   generated {i+1}/{MICRO_SAMPLES} micro tiles")

    return tiles


# ================================================================
# 2) Compute LAB features + index in FAISS
# ================================================================

def build_faiss_index(tiles):
    """
    tiles: list of 3xHxW tensors
    returns:
        feats:  N x 3 float32 array
        index:  FAISS IndexFlatL2(3)
    """
    feats_list = [rgb_to_lab_tensor(t) for t in tiles]
    feats = torch.stack(feats_list, dim=0).numpy().astype("float32")

    d = feats.shape[1]  # should be 3
    index = faiss.IndexFlatL2(d)
    index.add(feats)

    log(f">> FAISS index built: N={feats.shape[0]}, dim={d}")
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
    return out.images[0]   # 3 x H x W in [0,1]


# ================================================================
# 4) Mosaic builder (CPU)
# ================================================================

def build_mosaic(macro, tiles, feats, index, micro_class):
    """
    macro: 3×H×W in [0,1] (CPU)
    tiles: list of 3×MICRO_TILE_PX×MICRO_TILE_PX in [0,1] (CPU)
    feats: N×3 LAB features (not used directly here, but kept for clarity)
    index: FAISS IndexFlatL2(3)
    """

    # ensure macro is exactly MACRO_SIZE×MACRO_SIZE
    macro = macro.unsqueeze(0)      # 1×3×H×W
    macro = F.interpolate(macro, size=(MACRO_SIZE, MACRO_SIZE), mode="bilinear", align_corners=False)
    macro = macro.squeeze(0)        # 3×MACRO_SIZE×MACRO_SIZE

    mosaic = torch.zeros(3, FINAL_RES, FINAL_RES)
    log(f">> Building mosaic {GRID_SIZE}x{GRID_SIZE} → {FINAL_RES}x{FINAL_RES}")

    dither_amp = 0.05  # small LAB jitter to break regular patterns

    for gy in range(GRID_SIZE):
        for gx in range(GRID_SIZE):
            y0 = gy * MACRO_TILE_PX
            x0 = gx * MACRO_TILE_PX

            block = macro[:, y0:y0 + MACRO_TILE_PX, x0:x0 + MACRO_TILE_PX]

            lab_feat = rgb_to_lab_tensor(block)
            jitter = torch.randn(3) * dither_amp
            query = (lab_feat + jitter).numpy().astype("float32")

            # safety check to avoid FAISS dimension mismatch
            assert query.shape[0] == index.d, f"Query dim {query.shape[0]} != index dim {index.d}"

            D, I = index.search(query.reshape(1, -1), 1)
            idx = int(I[0][0])
            chosen = tiles[idx]

            # brightness-match
            micro_gray = chosen.mean().item() + 1e-6
            macro_gray = block.mean().item() + 1e-6
            scale = macro_gray / micro_gray
            scale = max(BRIGHTNESS_CLAMP[0], min(scale, BRIGHTNESS_CLAMP[1]))

            recolored = (chosen * scale).clamp(0, 1)

            # very slight color tint towards macro tile
            tint = block.mean(dim=(1, 2), keepdim=True)  # 3×1×1
            recolored = recolored * (1 - RECOLOR_STRENGTH) + tint * RECOLOR_STRENGTH
            recolored = recolored.clamp(0, 1)

            oy = gy * MICRO_TILE_PX
            ox = gx * MICRO_TILE_PX
            mosaic[:, oy:oy + MICRO_TILE_PX, ox:ox + MICRO_TILE_PX] = recolored

        if (gy + 1) % 8 == 0:
            log(f"   row {gy+1}/{GRID_SIZE} complete")

    return mosaic


# ================================================================
# MAIN
# ================================================================

def main():
    pipe, device = load_sdxl()

    # 1) Generate micro patch dataset
    tiles = generate_micro_tiles(pipe, device, MICRO_CLASS)
    tiles = [t.cpu() for t in tiles]

    # 2) Build FAISS index
    feats, index = build_faiss_index(tiles)

    # 3) Generate mosaics for each macro prompt
    for i, prompt in enumerate(MACRO_PROMPTS):
        macro = generate_macro(pipe, prompt, device).cpu()
        mosaic = build_mosaic(macro, tiles, feats, index, MICRO_CLASS)

        save_path = os.path.join(OUTPUT_DIR, f"mosaic_{MICRO_CLASS}_{i}.png")
        save_image(mosaic, save_path)
        log(f">> Saved: {save_path}\n")


if __name__ == "__main__":
    main()
