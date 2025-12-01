# src/generate_patches.py

import os
import torch
from tqdm import tqdm
from diffusers import StableDiffusionXLPipeline
from utils.image_utils import extract_center_patch


OUTPUT_DIR = "data/patch_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PATCH_SIZE = 64
SAMPLES_PER_CLASS = 300


CLASS_PROMPTS = {
    "FACE": [
        "a single cartoon face, white background, centered, simple, icon style",
        "a single human emoji face, plain background, centered",
        "a flat minimal face symbol, centered, bright background"
    ],
    "CAT": [
        "a single cartoon cat, white background, centered, simple",
        "a single cat silhouette icon, centered, minimal",
        "a kawaii cat face emoji, centered, plain white background"
    ],
    "FLOWER": [
        "a single red flower, white background, centered, high contrast",
        "a single daisy flower icon, flat style, centered",
        "a simple floral symbol, centered, plain background"
    ],
    "SYMBOL": [
        "a single black letter A, bold, white background, centered",
        "a single geometric triangle symbol, centered, simple",
        "a minimalist black circle icon, centered on white background"
    ],
}


def load_sdxl():
    # Detect device ONCE and keep it fixed
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Use fp16 ONLY on MPS
    dtype = torch.float16 if device == "mps" else torch.float32

    # Load SDXL with correct dtype
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
        use_safetensors=True
    )
    pipe = pipe.to(device)

    print(f">> Using device: {device}, dtype: {dtype}")
    return pipe, device, dtype



def generate_patch_dataset():
    pipe, device, dtype = load_sdxl()
    print(">> SDXL loaded. Starting dataset generation...\n")

    # Autocast should use SAME dtype as model, never override float16/float32
    autocast_dtype = dtype

    for cls, prompts in CLASS_PROMPTS.items():
        cls_dir = os.path.join(OUTPUT_DIR, cls)
        os.makedirs(cls_dir, exist_ok=True)

        print(f"Generating class: {cls} → {SAMPLES_PER_CLASS} samples")

        for i in tqdm(range(SAMPLES_PER_CLASS)):
            prompt = prompts[i % len(prompts)]

            # Always use the SAME autocast dtype, never branch on device inside loop
            with torch.autocast(device_type=device, dtype=autocast_dtype):
                img = pipe(
                    prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5
                ).images[0]

            patch = extract_center_patch(img, PATCH_SIZE)
            patch.save(os.path.join(cls_dir, f"{cls}_{i:04d}.png"))

    print("\n>> DONE! Synthetic patch dataset generated at:", OUTPUT_DIR)



if __name__ == "__main__":
    generate_patch_dataset()
