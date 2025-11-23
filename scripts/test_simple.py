import sys
import os
import torch
import torch.optim as optim
import numpy as np
import imageio
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from renderer import SimpleVolumeRenderer
from voxel_grid import LearnableVoxelGrid
from sds_utils import SDSLoss

# --- HELPER: Total Variation Loss (Fixes Shimmering) ---
def compute_tv_loss(grid):
    # grid: [4, H, W, D]
    # Calculate difference between neighbors in X, Y, Z directions
    diff_x = (grid[:, 1:, :, :] - grid[:, :-1, :, :]).abs().mean()
    diff_y = (grid[:, :, 1:, :] - grid[:, :, :-1, :]).abs().mean()
    diff_z = (grid[:, :, :, 1:] - grid[:, :, :, :-1]).abs().mean()
    return diff_x + diff_y + diff_z

def save_debug_gif(renderer, voxel_grid, path):
    images = []
    for angle in np.linspace(0, 360, 20): 
        origins, dirs = renderer.get_camera_rays(H=256, W=256, azimuth=angle, radius=2.5)
        with torch.no_grad():
            img = renderer.render(voxel_grid(), origins, dirs, background="white")
            img = torch.nan_to_num(img, 1.0).clamp(0, 1)
        img_np = (img.cpu().numpy() * 255).astype(np.uint8)
        images.append(img_np)
    imageio.mimsave(path, images, fps=10)

def main():
    device = "cuda"
    prompt = "A glowing red sphere, 3d render, 4k"
    
    print(f"Test Target: '{prompt}'")

    renderer = SimpleVolumeRenderer(device)
    grid_model = LearnableVoxelGrid(resolution=64, device=device).to(device)
    sds_loss = SDSLoss(device)
    
    # Lower LR slightly to let TV loss work
    optimizer = optim.Adam([
        {'params': grid_model.colors, 'lr': 0.1}, # Lowered from 1.0
        {'params': grid_model.density, 'lr': 0.05}
    ])
    
    target_embeds = sds_loss.encode_text(prompt)
    
    print("Starting Test (Red Init + TV Loss)...")
    
    for step in range(201):
        optimizer.zero_grad()
        
        azimuth = float(np.random.rand() * 360)
        elevation = float(np.random.rand() * 30 - 15)
        origins, dirs = renderer.get_camera_rays(H=512, W=512, azimuth=azimuth, elevation=elevation, radius=2.5)
        
        img = renderer.render(grid_model(), origins, dirs, background="white")
        
        img_batch = img.permute(2, 0, 1).unsqueeze(0)
        
        # Main SDS Loss
        loss_sds = sds_loss.compute_loss(img_batch, target_embeds, guidance_scale=100.0)
        
        # TV Loss (Smoothness)
        # We assume density (channel 3) needs the most smoothing to stop "fog"
        # We smooth colors (0-3) less to allow texture
        full_grid = grid_model()
        loss_tv_density = compute_tv_loss(full_grid[3:4, ...]) * 1.0
        loss_tv_color = compute_tv_loss(full_grid[0:3, ...]) * 0.1
        
        total_loss = loss_sds + loss_tv_density + loss_tv_color
        
        total_loss.backward()
        
        # Clip grads
        torch.nn.utils.clip_grad_norm_(grid_model.parameters(), 1.0)
        
        optimizer.step()
        
        if step % 20 == 0:
            mean_r = torch.sigmoid(grid_model.colors.data[0].mean()).item()
            mean_g = torch.sigmoid(grid_model.colors.data[1].mean()).item()
            mean_b = torch.sigmoid(grid_model.colors.data[2].mean()).item()
            print(f"Step {step:03d} | SDS: {loss_sds.item():.2f} | TV: {loss_tv_density.item():.4f}")
            print(f"   > Avg Colors -> R:{mean_r:.3f} G:{mean_g:.3f} B:{mean_b:.3f}")

        if step % 50 == 0:
            save_debug_gif(renderer, grid_model, f"../output/test_apple_{step}.gif")

    print("Test Complete. Check output/test_apple_0.gif immediately!")

if __name__ == "__main__":
    main()