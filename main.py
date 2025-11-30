import torch
import torchvision.transforms.functional as TF
from diffusers import StableDiffusionXLPipeline, DDIMScheduler

# 1. Setup Model
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
device = "cuda"

# Load components
# Note: use_safetensors=True helps with loading speed if available
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    use_safetensors=True,
    variant="fp16"
)
pipe = pipe.to(device)

# 2. Define the Low Pass Filter
def get_low_pass(tensor, kernel_size=33, sigma=2.0):
    return TF.gaussian_blur(tensor, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])

# 3. Configuration
prompt_low = "A portrait of Marilyn Monroe, smooth features, blurry aesthetic" 
prompt_high = "A portrait of Albert Einstein, detailed texture, wrinkles, sharp lines" 
negative_prompt = "low quality, bad anatomy, distorted, ugly"

kernel_size = 33 
sigma = 1.5
num_inference_steps = 50
guidance_scale = 7.5

# 4. Encoding Prompts & Conditioning
def encode_prompts(prompt, neg_prompt):
    """
    Helper to encode prompts and handle the complex SDXL return values.
    Returns: (prompt_embeds, pooled_embeds) concatenated for CFG
    """
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=neg_prompt,
    )
    
    # Concatenate for Classifier-Free Guidance (Negative first, then Positive)
    # Batch size becomes 2: [uncond, cond]
    embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    pooled = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
    
    return embeds, pooled

def get_add_time_ids(height, width):
    """
    SDXL requires 'time_ids' which encode the original image size and crop coordinates.
    Since we aren't using the pipe.__call__, we must manually create them.
    """
    original_size = (height, width)
    target_size = (height, width)
    crops_coords_top_left = (0, 0)
    
    # Create the ID tensor: [original_height, original_width, crop_top, crop_left, target_height, target_width]
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids], dtype=torch.float16, device=device)
    
    # Duplicate for CFG (batch size 2)
    return torch.cat([add_time_ids] * 2, dim=0)

with torch.no_grad():
    # A. Encode Prompt A (Low Freq Target)
    embeds_A, pooled_A = encode_prompts(prompt_low, negative_prompt)
    
    # B. Encode Prompt B (High Freq Target)
    embeds_B, pooled_B = encode_prompts(prompt_high, negative_prompt)
    
    # C. Create Time IDs
    # Default SDXL resolution is 1024x1024
    add_time_ids = get_add_time_ids(1024, 1024)

    # D. Prepare "added_cond_kwargs" dictionaries for the UNet
    # This was the cause of your error. It must be a dict!
    added_cond_kwargs_A = {"text_embeds": pooled_A, "time_ids": add_time_ids}
    added_cond_kwargs_B = {"text_embeds": pooled_B, "time_ids": add_time_ids}

# 5. The Custom Sampling Loop
scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
scheduler.set_timesteps(num_inference_steps)

# Random Initial Latents (128x128 latent size = 1024x1024 image size)
latents = torch.randn((1, 4, 128, 128), device=device, dtype=torch.float16) * scheduler.init_noise_sigma

print("Generating Hybrid Image...")
with torch.no_grad():
    for i, t in enumerate(scheduler.timesteps):
        # Expand latents for CFG (batch size 2: [uncond, cond])
        latent_model_input = torch.cat([latents] * 2) 
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # --- A. Predict Noise for Prompt A (Low Freq) ---
        noise_pred_A_full = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=embeds_A,
            added_cond_kwargs=added_cond_kwargs_A # CORRECTED: Passing Dict
        ).sample
        
        # Perform CFG manually
        noise_pred_uncond, noise_pred_text = noise_pred_A_full.chunk(2)
        noise_pred_A = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # --- B. Predict Noise for Prompt B (High Freq) ---
        noise_pred_B_full = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=embeds_B,
            added_cond_kwargs=added_cond_kwargs_B # CORRECTED: Passing Dict
        ).sample
        
        noise_pred_uncond, noise_pred_text = noise_pred_B_full.chunk(2)
        noise_pred_B = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # --- C. Mix in Frequency Domain ---
        low_freq_A = get_low_pass(noise_pred_A, kernel_size, sigma)
        low_freq_B = get_low_pass(noise_pred_B, kernel_size, sigma)
        high_freq_B = noise_pred_B - low_freq_B
        
        # Combine: Low Freq of A + High Freq of B
        noise_pred_hybrid = low_freq_A + high_freq_B

        # Step the scheduler
        latents = scheduler.step(noise_pred_hybrid, t, latents).prev_sample
        
        if i % 10 == 0:
            print(f"Step {i}/{num_inference_steps}")

# 6. Decode and Save
print("Decoding...")
pipe.vae.to(dtype=torch.float32)
latents = latents.to(dtype=torch.float32)

with torch.no_grad():
    # Decode
    image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    
    # FIX 2: Denormalize the image
    # VAE outputs values between -1 and 1. We need to shift them to 0 and 1.
    image = (image / 2 + 0.5).clamp(0, 1)

    # Convert to PIL
    image = image.cpu()
    image = TF.to_pil_image(image[0].float())
    image.save("hybrid_result.png")
    print("Done! Saved to hybrid_result.png")