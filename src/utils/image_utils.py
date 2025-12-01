# src/utils/image_utils.py

from PIL import Image

def extract_center_patch(img: Image.Image, patch_size=64):
    """
    Extract a center patch of fixed size from an image.
    """
    w, h = img.size
    left = (w - patch_size) // 2
    top = (h - patch_size) // 2
    return img.crop((left, top, left + patch_size, top + patch_size))

