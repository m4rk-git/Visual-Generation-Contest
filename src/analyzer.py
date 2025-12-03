import torch
import numpy as np
from sklearn.cluster import KMeans
from torch.nn import functional as F
from .utils import log, get_color_name
import config


def analyze_demand(macro_image_tensor):
    """
    1. Downscales macro image to grid size.
    2. Clusters pixels to find dominant color palettes.
    3. Calculates how many micro-tiles are needed for each color.
    """
    log(">> Analyzing Macro Scene Color Distribution...")

    # 1. Downsample to Grid Size (e.g., 64x64)
    # macro input is [3, H, W] -> unsqueeze to [1, 3, H, W]
    small_map = F.interpolate(
        macro_image_tensor.unsqueeze(0),
        size=(config.GRID_SIZE, config.GRID_SIZE),
        mode="area",
    ).squeeze(
        0
    )  # [3, 64, 64]

    # 2. Flatten to list of pixels for clustering
    # [3, 64, 64] -> [64*64, 3]
    pixels = small_map.permute(1, 2, 0).reshape(-1, 3).cpu().numpy()

    # 3. K-Means Clustering
    # We use fewer clusters (e.g., 12) to group similar needs together
    kmeans = KMeans(n_clusters=config.N_COLOR_CLUSTERS, n_init=10)
    kmeans.fit(pixels)

    # 4. Calculate Demand
    # labels_ is an array of 4096 integers (0 to 11)
    unique, counts = np.unique(kmeans.labels_, return_counts=True)

    demand_plan = []

    total_pixels = pixels.shape[0]

    log(f">> computed {len(unique)} color clusters. Planning generation budget:")

    for cluster_id, count in zip(unique, counts):
        ratio = count / total_pixels

        # Allocate budget (how many unique tiles to generate for this color)
        # We ensure at least 5 tiles per cluster if it exists
        n_samples = int(ratio * config.MICRO_SAMPLES)
        if n_samples < 5:
            n_samples = 5

        # Get the representative color of this cluster
        center_rgb = kmeans.cluster_centers_[cluster_id]
        color_name = get_color_name(center_rgb)

        demand_plan.append(
            {
                "cluster_id": cluster_id,
                "rgb_center": center_rgb,
                "color_name": color_name,
                "count": n_samples,
                "ratio": ratio,
            }
        )

        log(
            f"   Cluster {cluster_id}: {color_name.upper():<10} | Area: {ratio*100:.1f}% | Budget: {n_samples} tiles"
        )

    return demand_plan
