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

def save_debug_gif(renderer, voxel_grid, path):
    images = []
    # Save a quick spin
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
    prompt = "A bright red apple, 3d render, white background, 4k"
    
    print(f"Test Target: '{prompt}'")

    renderer = SimpleVolumeRenderer(device)
    grid_model = LearnableVoxelGrid(resolution=64, device=device).to(device)
    sds_loss = SDSLoss(device)
    
    # Use 0.05 LR. If colors still don't move, we might increase this later.
    optimizer = optim.Adam(grid_model.parameters(), lr=0.05)
    
    target_embeds = sds_loss.encode_text(prompt)
    
    print("Starting Sanity Check (Random Init + Color Grad Debug)...")
    
    # 200 Steps
    for step in range(201):
        optimizer.zero_grad()
        
        azimuth = float(np.random.rand() * 360)
        elevation = float(np.random.rand() * 30 - 15)
        origins, dirs = renderer.get_camera_rays(H=512, W=512, azimuth=azimuth, elevation=elevation, radius=2.5)
        
        img = renderer.render(grid_model(), origins, dirs, background="white")
        
        img_batch = img.permute(2, 0, 1).unsqueeze(0)
        
        # Increase scale slightly to 50 for stronger color signal
        loss = sds_loss.compute_loss(img_batch, target_embeds, guidance_scale=50.0)
        
        loss.backward()
        
        if step % 20 == 0:
            # --- DETAILED GRADIENT CHECK ---
            # Color Gradients (First 3 channels)
            color_grad = grid_model.grid.grad[:3].norm().item()
            # Density Gradients (4th channel)
            density_grad = grid_model.grid.grad[3].norm().item()
            
            # Values
            mean_r = torch.sigmoid(grid_model.grid.data[0].mean()).item()
            mean_g = torch.sigmoid(grid_model.grid.data[1].mean()).item()
            mean_b = torch.sigmoid(grid_model.grid.data[2].mean()).item()
            max_density = torch.nn.functional.softplus(grid_model.grid.data[3].max()).item()

            print(f"Step {step:03d} | Loss: {loss.item():.2f}")
            print(f"   > Gradients -> Color: {color_grad:.4f} | Density: {density_grad:.4f}")
            print(f"   > Avg Colors -> R:{mean_r:.2f} G:{mean_g:.2f} B:{mean_b:.2f}")
            print(f"   > Max Density: {max_density:.2f}")

        optimizer.step()
        
        if step % 50 == 0:
            save_debug_gif(renderer, grid_model, f"../output/test_apple_{step}.gif")

    print("Test Complete.")

if __name__ == "__main__":
    main()