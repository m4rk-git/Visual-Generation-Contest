import sys
import os
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from pipeline_panorama import SDXLPanoramaGenerator

def debug_run():
    print("Initializing Custom Pipeline for Debugging...")
    generator = SDXLPanoramaGenerator()
    
    # 1. TEST: Generate a standard square image using the CUSTOM loop
    # If this produces garbage, the loop logic (get_views/merge_views) is broken.
    print("Test 1: Generating Standard 1024x1024 (Custom Loop)...")
    image_square = generator.generate(
        "A photo of a cute corgi", 
        height=1024, 
        width=1024,  # Standard size
        steps=10      # Low steps for speed
    )
    image_square.save("../output/debug_square.png")
    print("Saved debug_square.png")

    # 2. TEST: Generate a small wide image
    # If Test 1 works but this fails, the wrapping/stride logic is broken.
    print("Test 2: Generating Wide 2048x1024...")
    image_wide = generator.generate(
        "A photo of a cute corgi", 
        height=1024, 
        width=2048, 
        steps=10
    )
    image_wide.save("../output/debug_wide.png")
    print("Saved debug_wide.png")

if __name__ == "__main__":
    debug_run()