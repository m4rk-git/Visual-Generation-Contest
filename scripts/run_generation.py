import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from pipeline_panorama import SDXLPanoramaGenerator

def main():
    generator = SDXLPanoramaGenerator()
    
    # 1. FRONT PROMPT (The Subject)
    # Only active in the middle of the image
    prompt_front = (
        "equirectangular projection, "
        "a gargantuan supermassive black hole with a glowing golden accretion disk, "
        "gravitational lensing, event horizon, cinematic lighting, 8k resolution"
    )
    
    # 2. BACK PROMPT (The Void)
    # Active everywhere else. Explicitly ask for empty space.
    prompt_back = (
        "equirectangular projection, "
        "empty deep space background, pitch black void, distant tiny stars, "
        "minimalist, darkness, high contrast, 8k resolution"
    )
    
    # Increase width to 8192 for crisp stars
    width = 10240
    
    print(f"Generating Solitary Black Hole ({width}x1024)...")
    
    image = generator.generate(
        prompt_front=prompt_front,
        prompt_back=prompt_back,
        height=1024, 
        width=width, 
        steps=50,
        guidance_scale=8.0
    )
    
    output_path = "../output/final_panorama.png"
    image.save(output_path)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()