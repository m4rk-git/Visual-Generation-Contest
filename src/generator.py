import torch
import random
from diffusers import StableDiffusionXLPipeline
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
    """
    Generates 1 tile for each color in the palette.
    Returns a dict mapping {cluster_id: tile_tensor}
    """
    palette_tiles = {}
    
    log(f">> Generating {len(palette_info)} Palette Tiles...")
    
    for item in palette_info:
        cid = item['id']
        color = item['color_name']
        
        base = random.choice(config.MICRO_BASE_PROMPTS)
        prompt = f"{base}, dominant color is {color}, {color} background, {color} lighting, strong hue"
        
        # Generate
        out = pipe(
            prompt,
            height=512, # Generate 512 for quality
            width=512,
            num_inference_steps=config.MICRO_STEPS, # 20 steps for good quality
            output_type="pt"
        ).images[0]
        
        # Resize to final micro size
        small = F.interpolate(
            out.unsqueeze(0),
            size=(config.MICRO_TILE_PX, config.MICRO_TILE_PX),
            mode="area"
        ).squeeze(0).cpu()
        
        # Store in dict
        palette_tiles[cid] = small
        
    return palette_tiles