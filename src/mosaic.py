import torch
import faiss
import numpy as np
from .utils import log
import config
from torchvision.transforms import functional as TF


def build_mosaic(macro_tensor, pool_images, pool_features):
    """
    Matches generated tiles to the macro scene.
    """
    log(">> Assembling Mosaic...")

    # 1. Prepare Search Index
    # pool_features is [N, 3] (RGB means)
    # Using Faiss for speed (though simple cdist is fine for N=1200)
    db_feats = pool_features.numpy().astype("float32")
    index = faiss.IndexFlatL2(3)
    index.add(db_feats)

    # 2. Iterate over Grid
    c, h, w = macro_tensor.shape
    mosaic = torch.zeros(
        (3, config.FINAL_RES, config.FINAL_RES)
    )  # Note: FINAL_RES needs def in config

    # Calculate final resolution
    FINAL_RES = config.GRID_SIZE * config.MICRO_TILE_PX
    mosaic = torch.zeros((3, FINAL_RES, FINAL_RES))

    # We iterate through the macro image patches
    for y in range(config.GRID_SIZE):
        for x in range(config.GRID_SIZE):

            # Extract target color from macro
            y_start = y * config.MACRO_TILE_PX
            x_start = x * config.MACRO_TILE_PX

            patch = macro_tensor[
                :,
                y_start : y_start + config.MACRO_TILE_PX,
                x_start : x_start + config.MACRO_TILE_PX,
            ]
            target_rgb = patch.mean(dim=(1, 2)).numpy().astype("float32")

            # Dithering (Optional): Add slight noise to target to vary selection
            jitter = np.random.normal(0, 0.02, size=3).astype("float32")
            query = target_rgb + jitter

            # Find best match in our Generated Pool
            _, idxs = index.search(query.reshape(1, 3), 1)
            best_idx = idxs[0][0]

            best_tile = pool_images[best_idx]

            # --- Optional: Color Correction ---
            # Even with demand-generation, a slight tint ensures smooth transitions
            # Calculate match color
            match_mean = pool_features[best_idx].view(3, 1, 1)
            target_mean = torch.tensor(target_rgb).view(3, 1, 1)

            # Shift match color towards target (Simple Color Transfer)
            # corrected = tile - tile_mean + target_mean
            corrected = best_tile - match_mean + target_mean
            corrected = torch.clamp(corrected, 0, 1)

            # Place in mosaic
            out_y = y * config.MICRO_TILE_PX
            out_x = x * config.MICRO_TILE_PX

            mosaic[
                :,
                out_y : out_y + config.MICRO_TILE_PX,
                out_x : out_x + config.MICRO_TILE_PX,
            ] = corrected

    return mosaic
