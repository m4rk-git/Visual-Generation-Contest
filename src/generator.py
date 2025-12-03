import torch
import random
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from torch.nn import functional as F
from .utils import log
import config


def load_sdxl():
    log(">> Loading SDXL...")

    # Optimization: Use FP16 and a specific VAE if needed
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )

    if config.DEVICE == "cuda":
        pipe.to("cuda")
        # pipe.enable_xformers_memory_efficient_attention() # Enable if xformers is installed

    return pipe


@torch.no_grad()
def generate_macro(pipe, prompt):
    log(f">> Generating Macro Scene: '{prompt}'")
    out = pipe(
        prompt,
        height=config.MACRO_SIZE,
        width=config.MACRO_SIZE,
        num_inference_steps=config.MACRO_STEPS,
        guidance_scale=config.GUIDANCE,
    ).images[0]

    # Convert PIL to Tensor [3, H, W]
    from torchvision.transforms.functional import to_tensor

    return to_tensor(out)


@torch.no_grad()
def generate_micro_pool(pipe, demand_plan):
    """
    Generates tiles based on the analyzed demand plan.
    """
    pool_images = []
    pool_features = []  # To store RGB means for matching

    log(f">> Starting Micro-Tile Generation (Budget: {config.MICRO_SAMPLES})...")

    total_generated = 0

    for item in demand_plan:
        color = item["color_name"]
        count = item["count"]

        # Create a prompt that enforces the color
        # We vary the base prompt slightly to ensure diversity

        log(f"   ... Generating {count} tiles for '{color}'")

        # Batching could optimize this further, but loop is safer for memory
        for _ in range(count):
            base_prompt = random.choice(config.MICRO_BASE_PROMPTS)

            # Construct specific prompt
            full_prompt = f"{base_prompt}, dominated by {color} color, {color} background, {color} lighting"

            # Generate small (512 is min for good SDXL quality, we downscale later)
            # Using fewer steps (15) for speed
            out = pipe(
                full_prompt,
                height=512,
                width=512,
                num_inference_steps=config.MICRO_STEPS,
                output_type="pt",
            ).images[
                0
            ]  # [3, 512, 512]

            # Downsample immediately to save RAM
            small_tile = (
                F.interpolate(
                    out.unsqueeze(0),
                    size=(config.MICRO_TILE_PX, config.MICRO_TILE_PX),
                    mode="area",
                )
                .squeeze(0)
                .cpu()
            )

            pool_images.append(small_tile)

            # Store feature (Mean RGB)
            # You could add LAB conversion here if you implemented it in utils
            feat = small_tile.mean(dim=(1, 2))
            pool_features.append(feat)

            total_generated += 1
            if total_generated % 50 == 0:
                print(".", end="", flush=True)

    print("")  # Newline
    log(f">> Generation Complete. Pool size: {len(pool_images)}")

    # Stack features for fast search
    pool_features = torch.stack(pool_features)
    return pool_images, pool_features
