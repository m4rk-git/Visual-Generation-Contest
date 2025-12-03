from sklearn.cluster import KMeans
from torch.nn import functional as F
from .utils import log, get_color_name
import config

def quantize_macro(macro_tensor):
    """
    1. Resizes macro to grid size (64x64).
    2. Clusters pixels into PALETTE_SIZE colors.
    3. Returns:
       - palette_info: List of dicts (id, rgb, prompt_color)
       - index_map: 64x64 array where each pixel is a cluster ID.
    """
    log(">> Quantizing Macro Scene...")
    
    # 1. Downsample
    # [3, H, W] -> [1, 3, Grid, Grid]
    small_map = F.interpolate(
        macro_tensor.unsqueeze(0), 
        size=(config.GRID_SIZE, config.GRID_SIZE), 
        mode="area"
    ).squeeze(0)
    
    # 2. Reshape for K-Means
    # [Grid*Grid, 3]
    pixels = small_map.permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
    
    # 3. Fit K-Means
    kmeans = KMeans(n_clusters=config.PALETTE_SIZE, n_init=10)
    kmeans.fit(pixels)
    
    # 4. Create Palette Data
    palette_info = []
    for i in range(config.PALETTE_SIZE):
        rgb = kmeans.cluster_centers_[i]
        name = get_color_name(rgb)
        palette_info.append({
            "id": i,
            "rgb": rgb,
            "color_name": name
        })
        log(f"   Palette Color {i:02d}: {name.upper()}")
        
    # 5. Create the Map
    # Reshape labels back to [Grid, Grid]
    index_map = kmeans.labels_.reshape(config.GRID_SIZE, config.GRID_SIZE)
    
    return palette_info, index_map