import torch
from .utils import log
import config


def assemble_mosaic(palette_tiles, index_map, macro_tensor):
    """
    palette_tiles: dict {id: tensor}
    index_map: np array [Grid, Grid] with cluster IDs
    macro_tensor: The original full-res macro image [3, H, W]
    """
    log(">> Assembling Mosaic with Color Transfer...")

    # Calculate dimensions
    final_res = config.GRID_SIZE * config.MICRO_TILE_PX
    mosaic = torch.zeros((3, final_res, final_res))

    # Pre-calculate mean color of every palette tile for fast adjustment
    palette_means = {}
    for cid, img in palette_tiles.items():
        palette_means[cid] = img.mean(dim=(1, 2))

    # Iterate through the grid
    for y in range(config.GRID_SIZE):
        for x in range(config.GRID_SIZE):

            # 1. Get location info
            cluster_id = index_map[y, x]

            # Tile we want to use (Source)
            source_tile = palette_tiles[cluster_id]
            source_mean = palette_means[cluster_id].view(3, 1, 1)

            # Spot in the Macro Image (Target)
            macro_y = y * config.MACRO_TILE_PX
            macro_x = x * config.MACRO_TILE_PX

            target_patch = macro_tensor[
                :,
                macro_y : macro_y + config.MACRO_TILE_PX,
                macro_x : macro_x + config.MACRO_TILE_PX,
            ]
            target_mean = target_patch.mean(dim=(1, 2)).view(3, 1, 1)

            # 2. COLOR TRANSFER (The Fix)

            # We add a 'strength' factor (0.0 to 1.0).
            # 1.0 = Perfect match to macro (Very recognizable, slightly tinted tiles)
            BLEND_STRENGTH = 0.8

            shift = target_mean - source_mean

            adjusted_tile = source_tile + (shift * BLEND_STRENGTH)

            # 3. Clamp to valid image range [0, 1]
            adjusted_tile = torch.clamp(adjusted_tile, 0, 1)

            # 4. Place in mosaic
            out_y = y * config.MICRO_TILE_PX
            out_x = x * config.MICRO_TILE_PX

            mosaic[
                :,
                out_y : out_y + config.MICRO_TILE_PX,
                out_x : out_x + config.MICRO_TILE_PX,
            ] = adjusted_tile

    return mosaic
