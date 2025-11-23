import sys
import os
import torch
import torch.optim as optim
from tqdm import tqdm
import imageio
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from renderer import SimpleVolumeRenderer
from voxel_grid import LearnableVoxelGrid
from sds_utils import SDSLoss


def save_gif(renderer, voxel_grid, path, frames=60):
    print(f"Saving GIF to {path}...")
    images = []
    for angle in np.linspace(0, 360, frames):
        origins, dirs = renderer.get_camera_rays(
            H=256, W=256, azimuth=angle, radius=3.0
        )
        with torch.no_grad():
            img = renderer.render(voxel_grid(), origins, dirs)
        img_np = (img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        images.append(img_np)
    imageio.mimsave(path, images, fps=15)


def main():
    device = "cuda"

    # 1. Configuration
    prompt_A = "A highly detailed vibrant colorful parrot, 3d render, 8k"
    prompt_B = "A shiny golden musical note symbol, 3d render, 8k"

    # 2. Initialize Components
    renderer = SimpleVolumeRenderer(device)
    grid_model = LearnableVoxelGrid(resolution=64, device=device).to(device)
    sds_loss = SDSLoss(device)

    # 3. Pre-compute Embeddings
    print("Encoding prompts...")
    embeds_A = sds_loss.encode_text(prompt_A)
    embeds_B = sds_loss.encode_text(prompt_B)

    # 4. Optimizer
    # Voxels need high learning rate
    optimizer = optim.Adam(grid_model.parameters(), lr=0.05)

    print("Starting Optimization (DreamFusion)...")
    pbar = tqdm(range(500))  # 500 Steps should be enough for a draft

    for step in pbar:
        optimizer.zero_grad()

        # --- VIEW A: Front (0 degrees) ---
        # Render
        origins_A, dirs_A = renderer.get_camera_rays(
            H=512, W=512, azimuth=0, radius=2.5
        )
        img_A = renderer.render(grid_model(), origins_A, dirs_A)
        # Permute for SDXL [H, W, C] -> [1, C, H, W]
        img_A_batch = img_A.permute(2, 0, 1).unsqueeze(0)
        # SDS Loss
        loss_A = sds_loss.compute_loss(img_A_batch, embeds_A)

        # --- VIEW B: Side (90 degrees) ---
        # Render
        origins_B, dirs_B = renderer.get_camera_rays(
            H=512, W=512, azimuth=90, radius=2.5
        )
        img_B = renderer.render(grid_model(), origins_B, dirs_B)
        img_B_batch = img_B.permute(2, 0, 1).unsqueeze(0)
        loss_B = sds_loss.compute_loss(img_B_batch, embeds_B)

        # Total Loss
        total_loss = loss_A + loss_B

        # Backprop
        total_loss.backward()
        optimizer.step()

        pbar.set_description(f"Loss: {total_loss.item():.4f}")

        # Cleanup VRAM
        del img_A, img_B, total_loss

        if step % 100 == 0:
            save_gif(renderer, grid_model, f"../output/step_{step}.gif")

    # Final Save
    save_gif(renderer, grid_model, "../output/final_illusion.gif")
    print("Done!")


if __name__ == "__main__":
    main()
