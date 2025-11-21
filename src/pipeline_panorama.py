import torch
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
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
        
        # 2. SWITCH TO DDIM (Critical Fix)
        # DDIM is robust for custom loops because it doesn't rely on an internal 
        # step counter that breaks when we process multiple views per timestep.
        self.pipe.scheduler = DDIMScheduler.from_config(
            self.pipe.scheduler.config, 
            steps_offset=1
        )
        
        # 3. Force UNet to Float16
        self.pipe.unet.to(dtype=torch.float16)
        
        # 4. Force VAE to Float32 (Prevents Black/Static Images)
        self.pipe.vae.to(dtype=torch.float32)
        self.pipe.enable_vae_tiling()

    @torch.no_grad()
    def get_views(self, panorama_latents, window_size=128, stride=64):
        """ Cropping Logic with Wrap-Around """
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
    def merge_latents(self, views, latent_shape, window_size=128, stride=64):
        """ Merges the PREDICTED x_{t-1} latents (MultiDiffusion) """
        count_map = torch.zeros(latent_shape, device=self.device, dtype=torch.float16)
        value_map = torch.zeros(latent_shape, device=self.device, dtype=torch.float16)
        h, w = latent_shape[2], latent_shape[3]

        for i, x in enumerate(range(0, w, stride)):
            view = views[i]
            if x + window_size <= w:
                value_map[:, :, :, x:x+window_size] += view
                count_map[:, :, :, x:x+window_size] += 1.0
            else:
                remaining = w - x
                overflow = window_size - remaining
                value_map[:, :, :, x:] += view[:, :, :, :remaining]
                count_map[:, :, :, x:] += 1.0
                value_map[:, :, :, :overflow] += view[:, :, :, remaining:]
                count_map[:, :, :, :overflow] += 1.0
        
        count_map = count_map.clamp(min=1.0)
        return value_map / count_map

    @torch.no_grad()
    def generate(self, prompt, height=1024, width=4096, steps=30, guidance_scale=7.5):
        # Calculate Latent Dimensions
        latent_H, latent_W = height // 8, width // 8
        
        # 1. Init Random Noise
        latents = torch.randn((1, 4, latent_H, latent_W), device=self.device, dtype=torch.float16)
        
        # DDIM usually uses init_noise_sigma = 1.0, but we check the scheduler just in case
        latents = latents * self.pipe.scheduler.init_noise_sigma
        
        # 2. Encode Prompts
        (prompt_embeds, negative_prompt_embeds, 
         pooled_prompt_embeds, negative_pooled_prompt_embeds) = self.pipe.encode_prompt(
            prompt=prompt, 
            do_classifier_free_guidance=True
        )

        # 3. Time IDs (Coordinate conditioning)
        # We tell the model each crop is a valid 1024x1024 image to avoid artifacts
        target_size = (1024, 1024)
        add_time_ids = torch.tensor(
            [[target_size[0], target_size[1], 0, 0, target_size[0], target_size[1]]], 
            device=self.device, dtype=torch.float16
        )
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0) # Batch for CFG

        # 4. Set Timesteps
        self.pipe.scheduler.set_timesteps(steps)
        timesteps = self.pipe.scheduler.timesteps

        print(f"Generating Panorama ({width}x{height}) using DDIM MultiDiffusion...")
        
        for t in tqdm(timesteps):
            # A. Get Views of CURRENT latents
            current_views = self.get_views(latents)
            next_step_views = []
            
            # B. Process each view independently
            for view in current_views:
                # 1. Expand for CFG
                latent_input = torch.cat([view] * 2)
                latent_input = self.pipe.scheduler.scale_model_input(latent_input, t)
                latent_input = latent_input.to(dtype=torch.float16)

                # 2. Predict Noise
                noise_pred = self.pipe.unet(
                    latent_input,
                    t,
                    encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds]),
                    added_cond_kwargs={
                        "text_embeds": torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds]), 
                        "time_ids": add_time_ids
                    }
                ).sample

                # 3. Guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # 4. Step individually (Compute x_{t-1} for this crop)
                # DDIM handles this correctly without messing up global state
                view_next = self.pipe.scheduler.step(noise_pred, t, view).prev_sample
                next_step_views.append(view_next)

            # C. Merge the PREDICTED x_{t-1} views
            # This averages the geometry, ensuring the panorama stays consistent
            latents = self.merge_latents(next_step_views, latents.shape)

        # 5. Decode
        print("Decoding...")
        latents = latents.to(dtype=torch.float32)
        image = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor, return_dict=False)[0]
        image = self.pipe.image_processor.postprocess(image, output_type="pil")[0]
        
        return image