# src/generate_patches.py

import os
import torch
from tqdm import tqdm
from diffusers import StableDiffusionXLPipeline
from utils.image_utils import extract_center_patch


# ============================
# CONFIG
# ============================

OUTPUT_DIR = "data/patch_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PATCH_SIZE = 64
SAMPLES_PER_CLASS = 300

CLASS_PROMPTS = {
    "FACE": [
        "a cute emoji-style human face, centered on white background, simple, glossy, smooth edges",
        "a friendly emoji face, centered, soft shading, white background, high quality",
        "a stylized cartoon emoji face, centered, clean design, glossy finish, white background"
    ],
    "CAT": [
        "a cute emoji-style cat face, centered on white background, simple, glossy, smooth edges",
        "a kawaii cat emoji face, centered, soft shading, high quality",
        "a stylized cartoon cat face, centered, glossy, white background"
    ],
    "FLOWER": [
        "a cute daisy emoji, centered on white background, simple, glossy, high quality",
        "a stylized flower emoji, centered, soft shading, white background",
        "a simple floral icon, centered, glossy, white background"
    ],
    "SYMBOL": [
        "a bold black letter A symbol, centered on white background, glossy, clean edges",
        "a geometric triangle icon, centered on white background, sharp edges, high contrast",
        "a minimalist black circle icon, centered on white background, clean and sharp"
    ],
}


# ============================
# LOAD SDXL (same as working test)
# ============================

def load_sdxl():
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32  # fallback

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
        use_safetensors=True
    ).to(device)

    print(f">> Using device: {device}, dtype: {dtype}")
    return pipe, device


# ============================
# GENERATE PATCH DATASET
# ============================

def generate_patch_dataset():
    pipe, device = load_sdxl()
    print("\n>> SDXL loaded. Starting dataset generation...\n")

    for cls, prompts in CLASS_PROMPTS.items():
        cls_dir = os.path.join(OUTPUT_DIR, cls)
        os.makedirs(cls_dir, exist_ok=True)

        print(f">>> Generating class: {cls} → {SAMPLES_PER_CLASS} samples")

        for i in tqdm(range(SAMPLES_PER_CLASS)):
            prompt = prompts[i % len(prompts)]

            # NO manual autocast, just like your working test
            img = pipe(
                prompt,
                num_inference_steps=40,
                guidance_scale=5.0,
            ).images[0]
            
        

            patch = extract_center_patch(img, PATCH_SIZE)
            patch.save(os.path.join(cls_dir, f"{cls}_{i:04d}.png"))

    print("\n>> DONE! Synthetic patch dataset generated at:", OUTPUT_DIR)


if __name__ == "__main__":
    generate_patch_dataset()
