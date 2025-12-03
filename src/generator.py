import torch
import random
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from torch.nn import functional as F
from .utils import log
import config

def load_sdxl():
    log(">> Loading SDXL...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
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
        guidance_scale=config.GUIDANCE
    ).images[0]
    
    from torchvision.transforms.functional import to_tensor
    return to_tensor(out)

@torch.no_grad()
def generate_palette_tiles(pipe, palette_info):
    palette_tiles = {}
    
    log(f">> Generating {len(palette_info)} Palette Tiles with Quality Control...")
    
    for item in palette_info:
        cid = item['id']
        color = item['color_name']
        
        # RETRY LOOP: Try up to 3 times to get a good tile for this color
        valid_tile = None
        for attempt in range(3):
            
            # 1. Prompt with Negative Prompt to stop tiling
            base = random.choice(config.MICRO_BASE_PROMPTS)
            prompt = f"{base}, {color} skin, {color} theme, {color} background, close-up"
            
            out = pipe(
                prompt,
                negative_prompt=config.NEGATIVE_PROMPT, # Ensure this is in config.py!
                height=1024,
                width=1024,
                num_inference_steps=config.MICRO_STEPS,
                output_type="pt"
            ).images[0] # [3, 1024, 1024]
            
            # 2. CENTER CROP (The "Solo Face" Fix)
            # Cut out the middle 600x600 to ignore edges where grids form
            c_h, c_w = 1024, 1024
            crop_size = 600
            start_y = (c_h - crop_size) // 2
            start_x = (c_w - crop_size) // 2
            center_patch = out[:, start_y:start_y+crop_size, start_x:start_x+crop_size]
            
            # 3. Resize to final micro size
            candidate = F.interpolate(
                center_patch.unsqueeze(0),
                size=(config.MICRO_TILE_PX, config.MICRO_TILE_PX),
                mode="bilinear",
                align_corners=False
            ).squeeze(0).cpu()
            
            # 4. FILTERING (The Quality Check)
            # We check the candidate *before* accepting it.
            std_dev = candidate.std()
            
            # Criteria: 
            # < 0.05 implies a gray/solid blob (Model Failure)
            # > 0.35 implies high-frequency noise (chaotic grid or artifacts)
            if std_dev < 0.05:
                log(f"   [Retry {attempt+1}/3] {color} tile too flat (std={std_dev:.3f})")
                continue # Try again
            elif std_dev > 0.35:
                log(f"   [Retry {attempt+1}/3] {color} tile too noisy (std={std_dev:.3f})")
                continue # Try again
            
            # If we pass checks, accept it
            valid_tile = candidate
            break
        
        # Fallback: If all 3 attempts failed, use the last one anyway to avoid crashing
        if valid_tile is None:
            log(f"   [Warning] Could not generate perfect tile for {color}. Using last attempt.")
            valid_tile = candidate
            
        palette_tiles[cid] = valid_tile
        
    return palette_tiles