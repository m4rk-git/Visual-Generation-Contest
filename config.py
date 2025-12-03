import os

# Paths
OUTPUT_DIR = "output_samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Resolution ----
MACRO_SIZE = 1024
GRID_SIZE = 64  # 64x64 = 4096 total tiles
MACRO_TILE_PX = MACRO_SIZE // GRID_SIZE

MICRO_TILE_PX = 64  # Final tile size

# ---- The Optimization ----
PALETTE_SIZE = 64  # Only generate 16 distinct tiles!
# Increase to 32 for better color fidelity if time permits.

# ---- SDXL ----
DEVICE = "cuda"
MACRO_STEPS = 30
MICRO_STEPS = 15  # We can afford higher steps/quality since we only gen 16 images
GUIDANCE = 5.0

TILE_STYLE = "ANIME"

# ---- Prompts ----
MACRO_PROMPTS = [
    "majestic landscape, northern lights over snowy mountains, vivid colors, 8k",
    "lush green forest clearing with sunlight streaming through trees, studio ghibli style",
    "cyberpunk city street at night, neon lights, highly detailed",
]

# ---- PROMPT DATABASE ----
STYLE_CONFIG = {
    "FACES": {
        "base_prompts": [
            "3d render of a single robot head, straight on, centered",
            "glossy plastic mannequin head, minimal, centered",
            "sculpture of a rounded face, smooth 3d style, solo object",
        ],
        "negative_prompt": (
            "grid, tiling, sprite sheet, collection, group, many faces, "
            "multiple views, split screen, text, watermark, detailed background, "
            "body, neck, shoulders, anime, 2d, drawing, sketch"
        ),
        "guidance": 8.0,  # High guidance for 3D object coherence
    },
    "ANIME": {
        "base_prompts": [
            "flat 2d vector art of a cute anime chibi character, simple, clean lines, white background",
            "cel shaded anime character portrait, studio ghibli style, vibrant, 2d art",
            "hand drawn manga character headshot, minimal, flat color, centered",
        ],
        "negative_prompt": (
            "3d, realistic, cgi, render, photograph, shiny, gradient, volumetric lighting, "
            "sketch, rough, messy, grayscale, monochrome, low quality, "
            "grid, tiling, multiple characters, text, signature"
        ),
        "guidance": 7.0,  # Slightly lower for "artistic" 2D style
    },
}
