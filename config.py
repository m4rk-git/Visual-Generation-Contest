import os

# Paths
OUTPUT_DIR = "output_samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Resolution Settings ----
MACRO_SIZE = 1024  # Resolution of the main scene
GRID_SIZE = 64  # 64x64 grid = 4096 tiles total
MACRO_TILE_PX = MACRO_SIZE // GRID_SIZE  # 16px source area

# ---- Micro-Tile Generation ----
MICRO_TILE_PX = 64  # Final resolution of small tiles
MICRO_SAMPLES = 1200  # Total budget: How many micro-images to generate
# Note: 1200 unique images is enough to cover a 4096 grid
# without obvious repetition if distribution is matched.

# ---- Clustering (The Optimization) ----
N_COLOR_CLUSTERS = 12  # How many dominant colors to extract from the scene

# ---- SDXL Settings ----
DEVICE = "cuda"
DTYPE = "fp16"  # Use float16 for speed
MACRO_STEPS = 30  # Higher quality for the big picture
MICRO_STEPS = 15  # Lower steps for icons (faster)
GUIDANCE = 5.0

# ---- Prompts ----
# The base prompts for the micro-tiles. The code will append color info automatically.
MICRO_BASE_PROMPTS = [
    "simple 3d render of a cute emoji face, glossy, centered, white background",
    "minimalist smooth 3d object, emoji style, centered, high contrast",
    "cute glossy icon, 3d render, isometric view, clean background",
]

MACRO_PROMPTS = [
    "majestic landscape, northern lights over snowy mountains, vivid colors, 8k",
    "lush green forest clearing with sunlight streaming through trees, studio ghibli style",
    "cyberpunk city street at night, neon lights, rain reflections, highly detailed",
]
