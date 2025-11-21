import torch
from diffusers import StableDiffusionXLPipeline
from tqdm import tqdm


class SDXLPanoramaGenerator:
    def __init__(
        self, model_id="stabilityai/stable-diffusion-xl-base-1.0", device="cuda"
    ):
        # Load SDXL in fp16 to save memory (crucial for KCLOUD)
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to(device)
        self.device = device

    @torch.no_grad()
    def get_views(self, panorama_latents, window_size=128, stride=64):
        """
        Crops the large panorama latent into smaller standard SDXL windows.
        Handles the 'seamless' wrap-around logic.
        """
        views = []
        h, w = panorama_latents.shape[2], panorama_latents.shape[3]

        # Simple sliding window with wrap-around
        for x in range(0, w, stride):
            if x + window_size <= w:
                # Standard crop
                view = panorama_latents[:, :, :, x : x + window_size]
            else:
                # Wrap-around crop (Circle behavior)
                remaining = w - x
                overflow = window_size - remaining
                view = torch.cat(
                    [
                        panorama_latents[:, :, :, x:],  # Right edge
                        panorama_latents[:, :, :, :overflow],  # Left edge (Wrap)
                    ],
                    dim=3,
                )
            views.append(view)
        return views

    @torch.no_grad()
    def merge_views(self, views, latent_shape, window_size=128, stride=64):
        """
        Averages the denoised views back into the large panorama canvas.
        """
        count_map = torch.zeros(latent_shape, device=self.device)
        value_map = torch.zeros(latent_shape, device=self.device)
        h, w = latent_shape[2], latent_shape[3]

        for i, x in enumerate(range(0, w, stride)):
            view = views[i]

            if x + window_size <= w:
                value_map[:, :, :, x : x + window_size] += view
                count_map[:, :, :, x : x + window_size] += 1.0
            else:
                # Handle wrap-around merge
                remaining = w - x
                overflow = window_size - remaining

                value_map[:, :, :, x:] += view[:, :, :, :remaining]
                count_map[:, :, :, x:] += 1.0

                value_map[:, :, :, :overflow] += view[:, :, :, remaining:]
                count_map[:, :, :, :overflow] += 1.0

        return value_map / count_map

    @torch.no_grad()
    def generate(self, prompt, height=1024, width=4096, steps=50):
        # 1. Define Latent Shapes (Scale factor 8 for SDXL)
        latent_H, latent_W = height // 8, width // 8

        # 2. Initial Random Noise
        latents = torch.randn(
            (1, 4, latent_H, latent_W), device=self.device, dtype=torch.float16
        )
        latents = latents * self.pipe.scheduler.init_noise_sigma

        # 3. Encode Prompt
        (
            prompt_embeds,
            negative_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(prompt=prompt)

        # 4. The Denoising Loop
        self.pipe.scheduler.set_timesteps(steps)

        print("Starting Panorama Generation...")
        for t in tqdm(self.pipe.scheduler.timesteps):
            # A. Crop views from the large canvas
            views = self.get_views(latents)

            # B. Denoise each view individually (Batch processing)
            # Note: For speed, you can batch these views if GPU RAM allows.
            # Here we do one by one to be safe.
            denoised_views_pred = []

            for view in views:
                # Expand embeddings to match view batch size (1)
                # Predict noise using the UNet
                latent_input = self.pipe.scheduler.scale_model_input(view, t)

                noise_pred = self.pipe.unet(
                    latent_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={
                        "text_embeds": pooled_prompt_embeds,
                        "time_ids": torch.tensor(
                            [[height, 1024, 0, 0, height, 1024]], device=self.device
                        ),  # Simplified ID logic
                    },
                ).sample
                denoised_views_pred.append(noise_pred)

            # C. Fuse views back together
            noise_pred_panorama = self.merge_views(denoised_views_pred, latents.shape)

            # D. Step the scheduler
            latents = self.pipe.scheduler.step(
                noise_pred_panorama, t, latents
            ).prev_sample

        # 5. Decode (Tiled VAE helps here, but standard decode for now)
        print("Decoding...")
        image = self.pipe.vae.decode(
            latents / self.pipe.vae.config.scaling_factor, return_dict=False
        )[0]
        image = self.pipe.image_processor.postprocess(image, output_type="pil")[0]

        return image
