import torch
import torchvision.transforms.functional as TF
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from utils import get_low_pass

# 1. Setup Model
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
device = "cuda"

# Load components
pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)


# 3. Configuration
prompt_low = (
    "A portrait of Marilyn Monroe, smooth features, blurry aesthetic"  # Far view
)
prompt_high = "A portrait of Albert Einstein, detailed texture, wrinkles, sharp lines"  # Near view
negative_prompt = "low quality, bad anatomy, distorted"

# Hyperparameters for Hybrid Effect
kernel_size = 33  # Controls how "blurry" the low pass is
sigma = 2.0  # Higher = more high-freq content bleeds into low-pass

# 4. Encoding Prompts
with torch.no_grad():
    # Helper to get embeddings (standard SDXL embedding logic)
    # Note: For simplicity, using pipe's internal helper if available or manual encoding
    # This part assumes you get conditioning tensors: embed_low, embed_high, embed_neg
    (embed_low, _, pooled_low, _) = pipe.encode_prompt(
        prompt_low, device=device, do_classifier_free_guidance=True
    )
    (embed_high, _, pooled_high, _) = pipe.encode_prompt(
        prompt_high, device=device, do_classifier_free_guidance=True
    )

# 5. The Custom Sampling Loop
scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
scheduler.set_timesteps(num_inference_steps=50)

# Random Initial Latents
latents = torch.randn((1, 4, 128, 128), device=device, dtype=torch.float16)

print("Generating Hybrid Image...")
with torch.no_grad():
    for t in scheduler.timesteps:
        # Expand latents for CFG (uncond, cond)
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # --- A. Predict Noise for Low Freq (Prompt A) ---
        noise_pred_A_full = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=embed_low,
            added_cond_kwargs=pooled_low,
        ).sample
        # Perform CFG manually
        noise_pred_uncond, noise_pred_text = noise_pred_A_full.chunk(2)
        noise_pred_A = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

        # --- B. Predict Noise for High Freq (Prompt B) ---
        noise_pred_B_full = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=embed_high,
            added_cond_kwargs=pooled_high,
        ).sample
        noise_pred_uncond, noise_pred_text = noise_pred_B_full.chunk(2)
        noise_pred_B = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

        # --- C. Mix in Frequency Domain ---
        low_freq_A = get_low_pass(noise_pred_A, kernel_size, sigma)
        low_freq_B = get_low_pass(noise_pred_B, kernel_size, sigma)
        high_freq_B = noise_pred_B - low_freq_B

        # Combine: Low Freq of A + High Freq of B
        noise_pred_hybrid = low_freq_A + high_freq_B

        # Step the scheduler
        latents = scheduler.step(noise_pred_hybrid, t, latents).prev_sample

# 6. Decode and Save
with torch.no_grad():
    image = pipe.vae.decode(
        latents / pipe.vae.config.scaling_factor, return_dict=False
    )[0]
    image = TF.to_pil_image(image[0].float())
    image.save("hybrid_result.png")
