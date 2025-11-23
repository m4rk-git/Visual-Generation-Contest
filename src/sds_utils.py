import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline

class SDSLoss:
    def __init__(self, device="cuda"):
        self.device = device
        print("Loading SDXL for SDS...")
        # 1. Load Pipeline
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to(device)
        self.pipe.enable_vae_tiling()
        
        # 2. CRITICAL FIX: Force VAE to Float32
        # SDXL VAE produces NaNs in Float16. We must use Float32.
        self.pipe.vae.to(dtype=torch.float32)
        
        # We don't need the standard scheduler logic, just the alphas
        self.alphas = self.pipe.scheduler.alphas_cumprod.to(device)

    def encode_text(self, prompt):
        # Pre-compute text embeddings
        (prompt_embeds, negative_prompt_embeds, 
         pooled_prompt_embeds, negative_pooled_prompt_embeds) = self.pipe.encode_prompt(prompt)
        return {
            "prompt_embeds": prompt_embeds,
            "neg_prompt_embeds": negative_prompt_embeds,
            "pooled": pooled_prompt_embeds,
            "neg_pooled": negative_pooled_prompt_embeds
        }

    def compute_loss(self, rendered_image, text_embeddings, guidance_scale=100):
        """
        Computes the SDS gradient.
        rendered_image: [1, 3, 512, 512] Float32 tensor, differentiable
        """
        target_size = 512
        
        # 3. Encode Image to Latents using FLOAT32 VAE
        # Input to VAE must be Float32 to avoid NaNs
        img_input = rendered_image.to(dtype=torch.float32) * 2.0 - 1.0 # Scale to [-1, 1]
        
        # Encode (in FP32)
        dist = self.pipe.vae.encode(img_input).latent_dist
        latents = dist.sample() * self.pipe.vae.config.scaling_factor
        
        # 4. Cast Latents to Float16 for the UNet
        # The UNet is still in FP16 for memory efficiency
        latents = latents.to(dtype=torch.float16)
        
        # Sample a random timestep t
        t = torch.randint(20, 980, (1,), device=self.device)
        noise = torch.randn_like(latents)
        
        # Add noise to latents
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, t)
        
        # 5. Predict Noise (UNet forward)
        with torch.no_grad():
            # Expand for CFG
            latent_model_input = torch.cat([noisy_latents] * 2)
            
            # Predict
            added_cond_kwargs = {
                "text_embeds": torch.cat([text_embeddings["neg_pooled"], text_embeddings["pooled"]]),
                "time_ids": torch.tensor([[target_size, target_size, 0, 0, target_size, target_size]] * 2, device=self.device)
            }
            
            noise_pred = self.pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=torch.cat([text_embeddings["neg_prompt_embeds"], text_embeddings["prompt_embeds"]]),
                added_cond_kwargs=added_cond_kwargs
            ).sample
            
            # CFG
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # 6. Calculate Gradient (SDS Update)
        grad_direction = (noise_pred - noise)
        
        # BACKPROP
        # We manually hook the gradient into the latents to flow back to the rendered image
        target = (latents - grad_direction).detach()
        
        # Loss must be computed in the precision of the latents (FP16)
        # But we cast to float32 for stability in the MSE calculation just in case
        loss = 0.5 * F.mse_loss(latents.float(), target.float())
        
        return loss