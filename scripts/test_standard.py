import torch
from diffusers import StableDiffusionXLPipeline

def test_standard_sdxl():
    print("Loading Standard SDXL Pipeline...")
    
    # 1. Load Model (Exactly as you do in your custom pipeline)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        torch_dtype=torch.float16, 
        variant="fp16", 
        use_safetensors=True
    ).to("cuda")

    # 2. Define a simple prompt
    prompt = "A photo of a cute corgi running in a park, high resolution, 8k"
    
    print("Generating standard 1024x1024 image...")
    
    # 3. Generate (Standard call, no custom loops)
    # We intentionally do NOT cast VAE to float32 here yet, to see if that's the root cause.
    image = pipe(prompt=prompt).images[0]
    
    # 4. Save
    output_path = "../output/standard_test.png"
    image.save(output_path)
    print(f"Saved test image to {output_path}")

if __name__ == "__main__":
    test_standard_sdxl()