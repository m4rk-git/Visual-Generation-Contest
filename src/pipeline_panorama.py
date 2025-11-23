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
        
        # 2. Use DDIM
        self.pipe.scheduler = DDIMScheduler.from_config(
            self.pipe.scheduler.config, 
            steps_offset=1
        )
        
        self.pipe.unet.to(dtype=torch.float16)
        self.pipe.vae.to(dtype=torch.float32)
        self.pipe.enable_vae_tiling()

    @torch.no_grad()
    def get_views(self, panorama_latents, window_size=128, stride=64):
        """ Returns list of views AND their starting x-coordinates """
        views = []
        coords = [] # Track where each view starts
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
            coords.append(x)
        return views, coords

    @torch.no_grad()
    def merge_maps(self, tiles, latent_shape, window_size=128, stride=64):
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
        
        count_map = count_map.clamp(min=1.0)
        return value_map / count_map

    @torch.no_grad()
    def generate(self, prompt_front, prompt_back, height=1024, width=8192, steps=50, guidance_scale=7.5):
        latent_H, latent_W = height // 8, width // 8
        
        # 1. Init Random Noise
        latents = torch.randn((1, 4, latent_H, latent_W), device=self.device, dtype=torch.float16)
        
        # 2. Encode BOTH Prompts
        # Front = Black Hole
        (front_embeds, neg_embeds, front_pooled, neg_pooled) = self.pipe.encode_prompt(
            prompt=prompt_front, do_classifier_free_guidance=True
        )
        # Back = Empty Space
        (back_embeds, _, back_pooled, _) = self.pipe.encode_prompt(
            prompt=prompt_back, do_classifier_free_guidance=True
        )

        # 3. Time IDs
        target_size = (1024, 1024)
        add_time_ids = torch.tensor(
            [[target_size[0], target_size[1], 0, 0, target_size[0], target_size[1]]], 
            device=self.device, dtype=torch.float16
        )
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0) 

        self.pipe.scheduler.set_timesteps(steps)
        
        print(f"Generating Region-Aware Panorama ({width}x{height})...")
        
        center_x = latent_W // 2
        # How wide is the black hole? (Let's say middle 25% of the image)
        focus_radius = latent_W // 8  
        
        for t in tqdm(self.pipe.scheduler.timesteps):
            # We now get coords along with views
            views, coords = self.get_views(latents)
            noise_preds = []
            
            for i, view in enumerate(views):
                start_x = coords[i]
                tile_center = start_x + 64 # 64 is half of window_size 128
                
                # --- SPATIAL LOGIC ---
                # Calculate distance from the center of the image
                dist_from_center = abs(tile_center - center_x)
                
                # If we are in the middle, use "Front Prompt". Else use "Back Prompt"
                if dist_from_center < focus_radius:
                    prompt_e = front_embeds
                    pooled_e = front_pooled
                else:
                    prompt_e = back_embeds
                    pooled_e = back_pooled
                
                # --- PREDICT ---
                latent_input = torch.cat([view] * 2)
                latent_input = self.pipe.scheduler.scale_model_input(latent_input, t)
                latent_input = latent_input.to(dtype=torch.float16)

                noise_pred = self.pipe.unet(
                    latent_input,
                    t,
                    encoder_hidden_states=torch.cat([neg_embeds, prompt_e]),
                    added_cond_kwargs={
                        "text_embeds": torch.cat([neg_pooled, pooled_e]), 
                        "time_ids": add_time_ids
                    }
                ).sample

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                noise_preds.append(noise_pred)

            merged_noise = self.merge_maps(noise_preds, latents.shape)
            latents = self.pipe.scheduler.step(merged_noise, t, latents).prev_sample

        print("Decoding...")
        latents = latents.to(dtype=torch.float32)
        image = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor, return_dict=False)[0]
        image = self.pipe.image_processor.postprocess(image, output_type="pil")[0]
        
        return image