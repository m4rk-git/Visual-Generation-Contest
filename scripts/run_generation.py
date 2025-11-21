import sys
import os

# Add 'src' to python path so we can import our module
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline_panorama import SDXLPanoramaGenerator


def main():
    # Initialize
    generator = SDXLPanoramaGenerator()

    # Prompt: Choose something that loops well
    prompt = "A seamless 360 degree panorama of a cyberpunk city skyline at night, neon lights, futuristic buildings, reflection in water, 8k, highly detailed"

    # Generate
    # Note: Start with smaller width (e.g., 2048) to test speed first!
    image = generator.generate(prompt, height=1024, width=2048, steps=30)

    # Save
    output_path = "../output/panorama_test.png"
    image.save(output_path)
    print(f"Saved panorama to {output_path}")


if __name__ == "__main__":
    main()
