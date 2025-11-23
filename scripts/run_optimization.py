import sys
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import imageio
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from renderer import SimpleVolumeRenderer
from voxel_grid import LearnableVoxelGrid
from sds_utils import SDSLoss

def save_gif(renderer, voxel_grid, path, frames=60):
    print(f"Saving GIF to {path}...")
    images = []
    # Create angles
    angles = np.linspace(0, 360, frames)
    
    for angle in angles:
        # Get Rays
        origins, dirs = renderer.get_camera_rays(
            H=256, W=256, 
            azimuth=float(angle), 
            radius=3.0
        )
        
        with torch.no_grad():
            img = renderer.render(voxel_grid(), origins, dirs)
            
            # --- FIX: Handle NaN/Inf values ---
            # Replace NaNs with 0.0 (Black) to prevent casting errors
            img = torch.nan_to_num(img, 0.0)
            img = img.clamp(0, 1)
            
        img_np = (img.cpu().numpy() * 255).astype(np.uint8)
        images.append(img_np)
        
    imageio.mimsave(path, images, fps=15)

def main():
    device = "cuda"
    
    # 1. Configuration
    # We use a Multi-View Prompt trick:
    # Front = Parrot, Side = Musical Note
    prompt_A = "A highly detailed vibrant colorful parrot, 3d render, 8k"
    prompt_B = "A shiny golden musical note symbol, 3d render, 8k"
    
    # 2. Initialize Components
    renderer = SimpleVolumeRenderer(device)
    # Higher resolution (64->96) helps shape definition
    grid_model = LearnableVoxelGrid(resolution=96, device=device).to(device)
    sds_loss = SDSLoss(device)
    
    # 3. Pre-compute Embeddings
    print("Encoding prompts...")
    embeds_A = sds_loss.encode_text(prompt_A)
    embeds_B = sds_loss.encode_text(prompt_B)
    
    # 4. Optimizer
    # We optimize the Voxel Grid parameters
    optimizer = optim.Adam(grid_model.parameters(), lr=0.05)
    
    print("Starting Optimization (Visual Anagram)...")
    total_steps = 500
    pbar = tqdm(range(total_steps)) 
    
    for step in pbar:
        optimizer.zero_grad()
        
        # --- VIEW A: Front (0 degrees) => Parrot ---
        # Randomize camera slightly around 0 deg to make it robust 3D
        azimuth_A = float(0 + np.random.randn() * 5)
        origins_A, dirs_A = renderer.get_camera_rays(H=512, W=512, azimuth=azimuth_A, radius=2.5)
        
        img_A = renderer.render(grid_model(), origins_A, dirs_A)
        
        # SDS expects [1, 3, H, W]
        img_A_batch = img_A.permute(2, 0, 1).unsqueeze(0)
        loss_A = sds_loss.compute_loss(img_A_batch, embeds_A)
        
        # --- VIEW B: Side (90 degrees) => Musical Note ---
        # Randomize camera slightly around 90 deg
        azimuth_B = float(90 + np.random.randn() * 5)
        origins_B, dirs_B = renderer.get_camera_rays(H=512, W=512, azimuth=azimuth_B, radius=2.5)
        
        img_B = renderer.render(grid_model(), origins_B, dirs_B)
        img_B_batch = img_B.permute(2, 0, 1).unsqueeze(0)
        loss_B = sds_loss.compute_loss(img_B_batch, embeds_B)
        
        # Total Loss
        total_loss = loss_A + loss_B
        
        # Backprop
        total_loss.backward()
        optimizer.step()
        
        pbar.set_description(f"Loss: {total_loss.item():.4f}")
        
        # Cleanup VRAM to avoid OOM on A100 20GB
        del img_A, img_B, img_A_batch, img_B_batch, loss_A, loss_B, total_loss
        
        # Save GIF periodically
        if step % 100 == 0:
            save_gif(renderer, grid_model, f"../output/step_{step}.gif")
            # Clear cache after saving to be safe
            torch.cuda.empty_cache()

    # Final Save
    save_gif(renderer, grid_model, "../output/final_illusion.gif")
    print("Done! Check ../output/final_illusion.gif")

if __name__ == "__main__":
    main()