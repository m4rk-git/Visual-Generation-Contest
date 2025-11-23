import torch
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from tqdm import tqdm

class SDXLPanoramaGenerator:
    def __init__(self, model_id="stabilityai/stable-diffusion-xl-base-1.0", device="cuda"):
        self.device = device
        
        # 1. Load SDXL
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            variant="fp16", 
            use_safetensors=True
        ).to(device)
        
        # 2. Use DDIM (Stateless - Safe for Noise Fusion)
        self.pipe.scheduler = DDIMScheduler.from_config(
            self.pipe.scheduler.config, 
            steps_offset=1
        )
        
        # 3. Optimizations
        self.pipe.unet.to(dtype=torch.float16)
        self.pipe.vae.to(dtype=torch.float32) # Fixes Black/NaN pixels
        self.pipe.enable_vae_tiling()

    @torch.no_grad()
    def get_views(self, panorama_latents, window_size=128, stride=64):
        """ Crops views with seamless wrap-around """
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
    def merge_maps(self, tiles, latent_shape, window_size=128, stride=64):
        """ 
        Generic merger for Noise or Latents.
        Averages overlapping tiles to create a seamless map.
        """
        count_map = torch.zeros(latent_shape, device=self.device, dtype=torch.float16)
        value_map = torch.zeros(latent_shape, device=self.device, dtype=torch.float16)
        h, w = latent_shape[2], latent_shape[3]

        for i, x in enumerate(range(0, w, stride)):
            tile = tiles[i]
            if x + window_size <= w:
                value_map[:, :, :, x:x+window_size] += tile
                count_map[:, :, :, x:x+window_size] += 1.0
            else:
                remaining = w - x
                overflow = window_size - remaining
                value_map[:, :, :, x:] += tile[:, :, :, :remaining]
                count_map[:, :, :, x:] += 1.0
                value_map[:, :, :, :overflow] += tile[:, :, :, remaining:]
                count_map[:, :, :, :overflow] += 1.0
        
        # Avoid division by zero
        count_map = count_map.clamp(min=1.0)
        return value_map / count_map

    @torch.no_grad()
    def generate(self, prompt, height=1024, width=8192, steps=50, guidance_scale=7.5):
        latent_H, latent_W = height // 8, width // 8
        
        # 1. Init Random Noise
        latents = torch.randn((1, 4, latent_H, latent_W), device=self.device, dtype=torch.float16)
        
        # 2. Encode Prompts
        (prompt_embeds, negative_prompt_embeds, 
         pooled_prompt_embeds, negative_pooled_prompt_embeds) = self.pipe.encode_prompt(
            prompt=prompt, 
            do_classifier_free_guidance=True
        )

        # 3. Time IDs (Standard 1024x1024 crops)
        target_size = (1024, 1024)
        add_time_ids = torch.tensor(
            [[target_size[0], target_size[1], 0, 0, target_size[0], target_size[1]]], 
            device=self.device, dtype=torch.float16
        )
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0) 

        self.pipe.scheduler.set_timesteps(steps)
        
        print(f"Generating Sharp Panorama ({width}x{height}) using Noise Fusion...")
        
        for t in tqdm(self.pipe.scheduler.timesteps):
            # A. Get Views of current latents
            views = self.get_views(latents)
            noise_preds = []
            
            # B. Predict Noise for each view (Independent)
            for view in views:
                latent_input = torch.cat([view] * 2)
                latent_input = self.pipe.scheduler.scale_model_input(latent_input, t)
                latent_input = latent_input.to(dtype=torch.float16)

                noise_pred = self.pipe.unet(
                    latent_input,
                    t,
                    encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds]),
                    added_cond_kwargs={
                        "text_embeds": torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds]), 
                        "time_ids": add_time_ids
                    }
                ).sample

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                noise_preds.append(noise_pred)

            # C. FUSE NOISE (The Sharpness Fix)
            # Instead of stepping individually and blurring the result,
            # we average the noise directions. This keeps textures sharp.
            merged_noise = self.merge_maps(noise_preds, latents.shape)

            # D. Global Step
            # We take ONE step on the full panorama using the consensus noise.
            latents = self.pipe.scheduler.step(merged_noise, t, latents).prev_sample

        print("Decoding High-Res...")
        latents = latents.to(dtype=torch.float32)
        
        # VAE Tiling is active, so this handles 8k widths fine
        image = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor, return_dict=False)[0]
        image = self.pipe.image_processor.postprocess(image, output_type="pil")[0]
        
        return image