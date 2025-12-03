import torch
import random
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
from torch.nn import functional as F
from .utils import log
import config


def load_sdxl():
    log(">> Loading SDXL...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    if config.DEVICE == "cuda":
        pipe.to("cuda")
    return pipe


@torch.no_grad()
def generate_macro(pipe, prompt):
    log(f">> Generating Macro: {prompt}")
    out = pipe(
        prompt,
        height=config.MACRO_SIZE,
        width=config.MACRO_SIZE,
        num_inference_steps=config.MACRO_STEPS,
        guidance_scale=config.GUIDANCE,
    ).images[0]

    from torchvision.transforms.functional import to_tensor

    return to_tensor(out)


@torch.no_grad()
def generate_palette_tiles(pipe, palette_info):
    palette_tiles = {}
    total_tiles = len(palette_info)

    # 1. LOAD THE ACTIVE STYLE CONFIG
    active_style = config.TILE_STYLE
    style_settings = config.STYLE_CONFIG[active_style]

    log(f">> Generating {total_tiles} Palette Tiles in style: {active_style}...")

    candidate = None

    for i, item in enumerate(palette_info, 1):
        cid = item["id"]
        color = item["color_name"]

        log(
            f"   [{i}/{total_tiles}] Generating {active_style} tile for: {color.upper()}..."
        )

        # Retry loop for quality control
        valid_tile = None
        for attempt in range(3):

            # 2. CONSTRUCT PROMPT BASED ON STYLE
            base = random.choice(style_settings["base_prompts"])

            if active_style == "FACES":
                # 3D Prompting (Lighting focus)
                prompt = (
                    f"{base}, {color} color scheme, {color} skin, {color} background, "
                    f"{color} lighting, isolated, solitary, 3d render"
                )
            else:
                # Anime Prompting (Flat Color focus)
                prompt = (
                    f"{base}, {color} hair, {color} eyes, {color} clothes, {color} theme, "
                    f"flat color, 2d art, simple background, centered, distinct facial features"
                )

            # 3. GENERATE
            out = pipe(
                prompt,
                negative_prompt=style_settings["negative_prompt"],
                height=1024,
                width=1024,
                num_inference_steps=config.MICRO_STEPS,
                guidance_scale=style_settings["guidance"],  # Use specific guidance
                output_type="pt",
            ).images[0]

            # 4. CENTER CROP (Crucial for both styles)
            c_h, c_w = 1024, 1024
            crop_size = 600
            start_y = (c_h - crop_size) // 2
            start_x = (c_w - crop_size) // 2
            center_patch = out[
                :, start_y : start_y + crop_size, start_x : start_x + crop_size
            ]

            # 5. RESIZE
            candidate = (
                F.interpolate(
                    center_patch.unsqueeze(0),
                    size=(config.MICRO_TILE_PX, config.MICRO_TILE_PX),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .cpu()
            )

            # 6. FILTER
            std_dev = candidate.std()
            if std_dev < 0.05:
                log(f"     -> Attempt {attempt+1} failed: Too flat (std={std_dev:.3f})")
                continue
            elif std_dev > 0.45:  # Anime allows slightly more complexity than 3D icons
                log(
                    f"     -> Attempt {attempt+1} failed: Too noisy (std={std_dev:.3f})"
                )
                continue

            valid_tile = candidate
            break

        if valid_tile is None:
            log(f"     -> Warning: Using fallback for {color}")
            valid_tile = candidate

        palette_tiles[cid] = valid_tile

    return palette_tiles
