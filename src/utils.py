import numpy as np

def log(msg):
    print(f"[Mosaic] {msg}", flush=True)

def get_color_name(rgb_array):
    """
    Maps RGB (0-1 floats) to a descriptive color name for prompting.
    """
    # Expanded dictionary for better SDXL guidance
    colors = {
        # Grayscale
        "pure white": (1.0, 1.0, 1.0),
        "light gray": (0.8, 0.8, 0.8),
        "dark gray": (0.3, 0.3, 0.3),
        "pitch black": (0.05, 0.05, 0.05),
        
        # Reds/Pinks
        "bright red": (1.0, 0.0, 0.0),
        "dark red": (0.5, 0.0, 0.0),
        "pink": (1.0, 0.7, 0.8),
        "magenta": (1.0, 0.0, 1.0),
        
        # Greens
        "bright green": (0.0, 1.0, 0.0),
        "dark forest green": (0.0, 0.3, 0.0),
        "olive": (0.5, 0.5, 0.0),
        "lime": (0.7, 1.0, 0.2),
        
        # Blues
        "bright blue": (0.0, 0.0, 1.0),
        "navy blue": (0.0, 0.0, 0.3),
        "sky blue": (0.5, 0.8, 1.0),
        "cyan": (0.0, 1.0, 1.0),
        "teal": (0.0, 0.5, 0.5),
        
        # Warm
        "bright yellow": (1.0, 1.0, 0.0),
        "orange": (1.0, 0.5, 0.0),
        "brown": (0.4, 0.2, 0.1),
        "beige": (0.9, 0.9, 0.7),
        "gold": (1.0, 0.8, 0.0),
        "purple": (0.5, 0.0, 0.5)
    }
    
    best_name = "colorful"
    min_dist = float("inf")
    
    for name, val in colors.items():
        val = np.array(val)
        # Weighted euclidean distance (human eye is more sensitive to green)
        # simple euclidean is fine too
        dist = np.sum((rgb_array - val)**2)
        if dist < min_dist:
            min_dist = dist
            best_name = name
            
    return best_name