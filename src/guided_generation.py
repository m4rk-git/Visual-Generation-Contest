import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from diffusers import StableDiffusionXLPipeline
from models.simple_cnn import SimplePatchCNN


# ================================================================
# CONFIG
# ================================================================

OUTPUT_DIR = "data/guided_samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = 384
EDIT_STEPS = 12
EDIT_LR = 0.05
PATCH_SIZE = 64
NUM_SAMPLES = 4

CLASS_TO_IDX = {"CAT": 0, "FACE": 1, "FLOWER": 2, "SYMBOL": 3}
TARGET_CLASS = "FACE"
TARGET_IDX = CLASS_TO_IDX[TARGET_CLASS]

LANDSCAPE_PROMPTS = [
    "a beautiful scenic landscape, cinematic, high detail",
    "a futuristic neon-lit street at night, wide shot",
    "a forest trail with morning light and fog",
    "ancient ruins covered by jungle vegetation",
    "a cozy indoor room with warm ambient light",
]


# ================================================================
# UTIL — CENTER PATCH EXTRACTION
# ================================================================

def extract_center_patch(img, patch_size):
    B, C, H, W = img.shape
    ph = pw = patch_size
    y0 = (H - ph) // 2
    x0 = (W - pw) // 2
    patch = img[:, :, y0:y0+ph, x0:x0+pw]
    return patch


# ================================================================
# LOAD SDXL (UNET on GPU, VAE on CPU)
# ================================================================

def load_sdxl():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )

    pipe.safety_checker = None
    pipe.enable_vae_tiling()

    # UNET + encoders to GPU
    pipe = pipe.to(device)

    # VAE to CPU
    pipe.vae.to("cpu")

    print(">> Loaded SDXL: UNET on GPU, VAE on CPU")
    return pipe, device


# ================================================================
# LOAD CLASSIFIER
# ================================================================

def load_classifier(device):
    model = SimplePatchCNN(num_classes=4).to(device)
    state = torch.load("models/patch_classifier.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


# ================================================================
# MANUAL SDXL SAMPLING (NO PIPE(...))
# ================================================================

@torch.no_grad()
def generate_scene(pipe, device):
    """
    Full SDXL sampling without using pipe(...)
    Works with UNet on GPU and VAE on CPU.
    """

    unet = pipe.unet
    vae = pipe.vae
    scheduler = pipe.scheduler

    # choose prompt
    prompt = LANDSCAPE_PROMPTS[torch.randint(0, len(LANDSCAPE_PROMPTS), (1,)).item()]

    # ---- 1. Encode text ----
    cond, uncond, pooled, uncond_pooled = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )

    text_embeds = torch.cat([uncond, cond], dim=0).to(device, torch.float16)
    pooled_embeds = torch.cat([uncond_pooled, pooled], dim=0).to(device, torch.float16)

    # time ids
    proj_dim = pipe.text_encoder_2.config.projection_dim
    add_time_ids = pipe._get_add_time_ids(
        original_size=(IMAGE_SIZE, IMAGE_SIZE),
        crops_coords_top_left=(0,0),
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        dtype=torch.float16,
        text_encoder_projection_dim=proj_dim,
    ).to(device)
    add_time_ids = add_time_ids.repeat(2,1)

    added_cond = {
        "text_embeds": pooled_embeds,
        "time_ids": add_time_ids,
    }

    # ---- 2. Sampling loop ----
    scheduler.set_timesteps(35, device=device)
    timesteps = scheduler.timesteps

    latents = torch.randn(
        (1, unet.config.in_channels, IMAGE_SIZE // 8, IMAGE_SIZE // 8),
        device=device,
        dtype=torch.float16
    )

    for t in timesteps:
        noisy_latent = torch.cat([latents]*2)
        noisy_latent = scheduler.scale_model_input(noisy_latent, t)

        noise_pred = unet(
            noisy_latent,
            t,
            encoder_hidden_states=text_embeds,
            added_cond_kwargs=added_cond,
        ).sample

        uncond_np, cond_np = noise_pred.chunk(2)
        guided = uncond_np + 5.5 * (cond_np - uncond_np)

        latents = scheduler.step(guided, t, latents).prev_sample

    # ---- 3. CPU VAE decode ----
    z = (latents / vae.config.scaling_factor).to("cpu").float()
    decoded = vae.decode(z).sample          # [-1,1]
    img = (decoded/2 + 0.5).clamp(0,1)      # [0,1]
    return img


# ================================================================
# MICRO-PATCH GRADIENT EDITOR
# ================================================================

def micro_patch_edit(pipe, classifier, base_img, device, target_idx):
    vae = pipe.vae
    scaling = vae.config.scaling_factor

    # 1) ENCODE ON CPU
    with torch.no_grad():
        latent = vae.encode(base_img.cpu()).latent_dist.sample()

    latent = (latent * scaling).to(device).half()
    latent.requires_grad_(True)

    optim = torch.optim.Adam([latent], lr=EDIT_LR)
    ce_loss = nn.CrossEntropyLoss()
    target = torch.tensor([target_idx], device=device)

    for _ in range(EDIT_STEPS):
        optim.zero_grad()

        # CPU decode
        z_cpu = (latent / scaling).detach().cpu().float()
        with torch.no_grad():
            decoded = vae.decode(z_cpu).sample
        img = (decoded/2 + 0.5).clamp(0,1).to(device)

        # extract center patch
        patch = extract_center_patch(img, PATCH_SIZE)

        logits = classifier(patch.float())
        loss = ce_loss(logits, target)

        loss.backward()
        optim.step()

        with torch.no_grad():
            latent[:] = torch.clamp(latent, -3, 3)

    # final decode
    z_cpu = (latent / scaling).detach().cpu().float()
    with torch.no_grad():
        out = vae.decode(z_cpu).sample
    final = (out/2+0.5).clamp(0,1)
    return final


# ================================================================
# MAIN
# ================================================================

def main():
    torch.cuda.empty_cache()

    pipe, device = load_sdxl()
    classifier = load_classifier(device)

    print(f"\n>> Generating {NUM_SAMPLES} guided images")
    print(f">> Micro-class target = {TARGET_CLASS} ({TARGET_IDX})\n")

    for i in range(NUM_SAMPLES):
        base_img = generate_scene(pipe, device)
        final = micro_patch_edit(pipe, classifier, base_img, device, TARGET_IDX)

        save_path = f"{OUTPUT_DIR}/{TARGET_CLASS}_micro_{i}.png"
        save_image(final, save_path)
        print("Saved:", save_path)


if __name__ == "__main__":
    main()
