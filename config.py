# Paths
OUTPUT_DIR = "output_samples"

# ---- Resolution ----
MACRO_SIZE = 1024
GRID_SIZE = 64  # 64x64 = 4096 total tiles
MACRO_TILE_PX = MACRO_SIZE // GRID_SIZE

MICRO_TILE_PX = 64  # Final tile size

# ---- The Optimization ----
PALETTE_SIZE = 64  # Number of colors in macro palette

# ---- SDXL ----
DEVICE = "cuda"
MACRO_STEPS = 30
MICRO_STEPS = 15  # We can afford higher steps/quality since we only gen 16 images
GUIDANCE = 5.0

TILE_STYLE = "FLOWERS"

# ---- Prompts ----
MACRO_PROMPTS = [
    "majestic landscape, northern lights over snowy mountains, vivid colors, 8k",
    "lush green forest clearing with sunlight streaming through trees, studio ghibli style",
    "steampunk industrial city, brass gears and steam clouds, victorian architecture, golden hour lighting",
]

# ---- PROMPTS ----
STYLE_CONFIG = {
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
        "guidance": 7.0,
    },
    "FISH": {
        "base_prompts": [
            "macro photography of a single tropical fish, side profile, exotic, centered",
            # REMOVED: "3d render of a cute round pufferfish, floating, minimal, centered",
            "close-up of a beta fish, flowing fins, elegant, solo object",
            "vibrant clownfish swimming, coral reef background, detailed scales, centered",
            "majestic koi fish, top-down view, rippled water, artistic style",
        ],
        "negative_prompt": (
            "pufferfish, blowfish, spiked fish, balloon fish, "  # <--- Explicitly ban them here too
            "fishing, dead, market, food, sushi, cooking, hook, net, "
            "grid, tiling, flock, school of fish, multiple fish, text, watermark, "
            "aquarium glass, reflection, messy background"
        ),
        "guidance": 7.5,
    },
    "FLOWERS": {
        "base_prompts": [
            "macro photography of a single flower head, top-down view, centered, highly detailed",
            "close-up of a blooming blossom, soft natural lighting, depth of field, solo object",
            "botanical illustration of a flower, clean background, vivid colors, centered",
        ],
        "negative_prompt": (
            "bouquet, vase, stem, leaves, garden, field, many flowers, grid, tiling, "
            "wilted, dry, dead, text, watermark, messy background, insects"
        ),
        "guidance": 7.0,
    },
}
