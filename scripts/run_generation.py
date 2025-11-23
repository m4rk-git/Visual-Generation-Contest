import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from pipeline_panorama import SDXLPanoramaGenerator

def main():
    generator = SDXLPanoramaGenerator()
    
    # Prompt Engineering for Wide Aspect Ratio
    # Keywords like "wide angle", "panorama", "seamless" help.
    prompt = "A breathtaking seamless panorama of a futuristic cyberpunk city at night, neon lights reflecting in rain puddles, towering skyscrapers, flying cars, 8k resolution, highly detailed, cinematic lighting, wide angle lens"
    
    # Generate 4096px wide image (Ratio 4:1)
    # This might take ~1-2 minutes on KCLOUD A100
    image = generator.generate(prompt, height=1024, width=4096, steps=50)
    
    output_path = "../output/final_panorama.png"
    image.save(output_path)
    print(f"Saved panorama to {output_path}")

if __name__ == "__main__":
    main()