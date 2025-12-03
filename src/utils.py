import torch
import numpy as np
from PIL import Image


def log(msg):
    print(f"[Mosaic] {msg}", flush=True)


def rgb_to_lab(img_tensor):
    """
    Convert a batch of RGB tensors (C, H, W) to LAB color space features.
    Returns mean L, A, B values.
    """
    # Simple approximation or use a library if precise color matching is needed.
    # For speed, we usually just stick to RGB mean matching, but here is a placeholder
    # if you want to implement full RGB->LAB.
    # For this contest, RGB Mean Matching is often sufficient and faster.
    return img_tensor.mean(dim=(1, 2))  # Returns (R, G, B) means


def get_color_name(rgb_array):
    """
    Maps an RGB (0-1 range) numpy array to a color name for prompting.
    """
    r, g, b = rgb_array

    # Define a simple palette
    colors = {
        "red": (1.0, 0.0, 0.0),
        "green": (0.0, 1.0, 0.0),
        "blue": (0.0, 0.0, 1.0),
        "cyan": (0.0, 1.0, 1.0),
        "magenta": (1.0, 0.0, 1.0),
        "yellow": (1.0, 1.0, 0.0),
        "black": (0.1, 0.1, 0.1),
        "white": (0.9, 0.9, 0.9),
        "gray": (0.5, 0.5, 0.5),
        "orange": (1.0, 0.5, 0.0),
        "purple": (0.5, 0.0, 0.5),
        "brown": (0.6, 0.4, 0.2),
        "dark blue": (0.0, 0.0, 0.5),
        "beige": (0.9, 0.9, 0.7),
    }

    # Find nearest neighbor
    best_name = "colorful"
    min_dist = float("inf")

    for name, val in colors.items():
        val = np.array(val)
        dist = np.sum((rgb_array - val) ** 2)
        if dist < min_dist:
            min_dist = dist
            best_name = name

    return best_name
