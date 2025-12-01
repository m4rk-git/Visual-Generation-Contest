# utils/image_utils.py

from PIL import Image

def extract_center_patch(img: Image.Image, patch_size: int = 64) -> Image.Image:
    """
    Take a PIL image, crop a centered square, then resize to patch_size x patch_size.
    This is deliberately simple and robust.
    """
    w, h = img.size
    side = min(w, h)

    left   = (w - side) // 2
    top    = (h - side) // 2
    right  = left + side
    bottom = top + side

    cropped = img.crop((left, top, right, bottom))
    patch = cropped.resize((patch_size, patch_size), Image.LANCZOS)
    patch = patch.convert("RGB")
    return patch
