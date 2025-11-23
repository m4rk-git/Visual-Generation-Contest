import torch
from diffusers import StableDiffusionXLPipeline


class SDSLoss:
    def __init__(self, device="cuda"):
        self.device = device
        print("Loading SDXL for SDS...")
        # Load in half precision for speed/memory
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to(device)
        self.pipe.enable_vae_tiling()

        # We don't need the standard scheduler logic, just the alphas
        self.alphas = self.pipe.scheduler.alphas_cumprod.to(device)

    def encode_text(self, prompt):
        # Pre-compute text embeddings to save time
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(prompt)
        return {
            "prompt_embeds": prompt_embeds,
            "neg_prompt_embeds": negative_prompt_embeds,
            "pooled": pooled_prompt_embeds,
            "neg_pooled": negative_pooled_prompt_embeds,
        }

    def compute_loss(self, rendered_image, text_embeddings, guidance_scale=100):
        """
        Computes the SDS gradient.
        rendered_image: [1, 3, 512, 512] Float32 tensor, differentiable
        """
        # 1. Resize/Preprocess image for SDXL
        # SDXL works best at 1024, but 512 is faster for optimization loops
        # We assume input is already 512x512
        target_size = 512

        # 2. Encode Image to Latents (Expensive step!)
        # We must cast to fp16 for the VAE
        img_input = (
            rendered_image.to(dtype=torch.float16) * 2.0 - 1.0
        )  # Scale to [-1, 1]
        latents = self.pipe.vae.encode(img_input).latent_dist.sample()
        latents = latents * self.pipe.vae.config.scaling_factor

        # 3. Add Noise (Timestep sampling)
        # Sample a random timestep t ~ U[0.02, 0.98]
        t = torch.randint(20, 980, (1,), device=self.device)
        noise = torch.randn_like(latents)

        # q_sample: Add noise to latents
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, t)

        # 4. Predict Noise (UNet forward)
        # We need gradients for the input (latents), but we detach the UNet
        with torch.no_grad():
            # Expand for CFG
            latent_model_input = torch.cat([noisy_latents] * 2)

            # Predict
            added_cond_kwargs = {
                "text_embeds": torch.cat(
                    [text_embeddings["neg_pooled"], text_embeddings["pooled"]]
                ),
                "time_ids": torch.tensor(
                    [[target_size, target_size, 0, 0, target_size, target_size]] * 2,
                    device=self.device,
                ),
            }

            noise_pred = self.pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=torch.cat(
                    [
                        text_embeddings["neg_prompt_embeds"],
                        text_embeddings["prompt_embeds"],
                    ]
                ),
                added_cond_kwargs=added_cond_kwargs,
            ).sample

            # CFG
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        # 5. Calculate Gradient (SDS Update)
        # The gradient direction is (predicted_noise - actual_noise)
        # We weight it by w(t) (omitted for simplicity, usually 1)
        grad_direction = noise_pred - noise

        # BACKPROP: We want to maximize the probability of the image => Minimize the difference
        # We manually hook the gradient into the latents to flow back to the rendered image
        # This is the "Score Distillation" trick
        target = (latents - grad_direction).detach()
        loss = 0.5 * F.mse_loss(latents, target)

        return loss
