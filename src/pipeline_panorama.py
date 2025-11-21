import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from tqdm import tqdm

class SDXLPanoramaGenerator:
    def __init__(self, model_id="stabilityai/stable-diffusion-xl-base-1.0", device="cuda"):
        # 1. Load SDXL
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            variant="fp16", 
            use_safetensors=True
        ).to(device)
        self.device = device
        
        # 2. Use EulerDiscreteScheduler (Best for Panorama/Tiling)
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        
        # 3. Force UNet to Float16
        self.pipe.unet.to(dtype=torch.float16)
        
        # 4. Force VAE to Float32 (Critical for preventing artifacts/black pixels)
        self.pipe.vae.to(dtype=torch.float32)
        self.pipe.enable_vae_tiling()

    @torch.no_grad()
    def get_views(self, panorama_latents, window_size=128, stride=64):
        views = []
        h, w = panorama_latents.shape[2], panorama_latents.shape[3]
        for x in range(0, w, stride):
            if x + window_size <= w:
                view = panorama_latents[:, :, :, x:x+window_size]
            else:
                remaining = w - x
                overflow = window_size - remaining
                view = torch.cat([
                    panorama_latents[:, :, :, x:], 
                    panorama_latents[:, :, :, :overflow]
                ], dim=3)
            views.append(view)
        return views

    @torch.no_grad()
    def merge_noise(self, noise_views, latent_shape, window_size=128, stride=64):
        # We merge NOISE predictions, not latents
        count_map = torch.zeros(latent_shape, device=self.device, dtype=torch.float16)
        value_map = torch.zeros(latent_shape, device=self.device, dtype=torch.float16)
        h, w = latent_shape[2], latent_shape[3]

        for i, x in enumerate(range(0, w, stride)):
            noise_view = noise_views[i]
            if x + window_size <= w:
                value_map[:, :, :, x:x+window_size] += noise_view
                count_map[:, :, :, x:x+window_size] += 1.0
            else:
                remaining = w - x
                overflow = window_size - remaining
                value_map[:, :, :, x:] += noise_view[:, :, :, :remaining]
                count_map[:, :, :, x:] += 1.0
                value_map[:, :, :, :overflow] += noise_view[:, :, :, remaining:]
                count_map[:, :, :, :overflow] += 1.0
        
        # Avoid division by zero
        count_map = count_map.clamp(min=1.0)
        return value_map / count_map

    @torch.no_grad()
    def generate(self, prompt, height=1024, width=4096, steps=30, guidance_scale=7.5):
        latent_H, latent_W = height // 8, width // 8
        
        # 1. Init Noise
        latents = torch.randn((1, 4, latent_H, latent_W), device=self.device, dtype=torch.float16)
        latents = latents * self.pipe.scheduler.init_noise_sigma
        
        # 2. Encode Prompts
        (prompt_embeds, negative_prompt_embeds, 
         pooled_prompt_embeds, negative_pooled_prompt_embeds) = self.pipe.encode_prompt(
            prompt=prompt, 
            do_classifier_free_guidance=True
        )

        # 3. Set Time IDs (Fixed to 1024x1024 to prevent stretching/garbage)
        target_size = (1024, 1024)
        add_time_ids = torch.tensor(
            [[target_size[0], target_size[1], 0, 0, target_size[0], target_size[1]]], 
            device=self.device, dtype=torch.float16
        )
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0) # Batch of 2 for CFG

        self.pipe.scheduler.set_timesteps(steps)
        
        print(f"Generating Panorama ({width}x{height})...")
        for t in tqdm(self.pipe.scheduler.timesteps):
            # A. Get Views
            views = self.get_views(latents)
            noise_preds = []
            
            # B. Calculate Noise for each view
            for view in views:
                # Expand for CFG
                latent_model_input = torch.cat([view] * 2)
                latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = latent_model_input.to(dtype=torch.float16)

                # Predict Noise
                noise_pred = self.pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds]),
                    added_cond_kwargs={
                        "text_embeds": torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds]), 
                        "time_ids": add_time_ids
                    }
                ).sample

                # Perform CFG
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                noise_preds.append(noise_pred)

            # C. Merge NOISE (Gradient Fusion)
            # We merge the noise maps FIRST
            merged_noise = self.merge_noise(noise_preds, latents.shape)

            # D. Step ONCE on the full panorama
            # This ensures the scheduler index only increments by 1 per timestep
            latents = self.pipe.scheduler.step(merged_noise, t, latents).prev_sample

        # 4. Decode
        print("Decoding...")
        latents = latents.to(dtype=torch.float32)
        image = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor, return_dict=False)[0]
        image = self.pipe.image_processor.postprocess(image, output_type="pil")[0]
        
        return image