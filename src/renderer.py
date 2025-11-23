import torch
import torch.nn.functional as F

class SimpleVolumeRenderer:
    def __init__(self, device="cuda"):
        self.device = device

    def get_camera_rays(self, H, W, fov=60.0, elevation=0.0, azimuth=0.0, radius=2.5):
        # Explicitly cast inputs to float32
        el = torch.deg2rad(torch.tensor(elevation, device=self.device, dtype=torch.float32))
        az = torch.deg2rad(torch.tensor(azimuth, device=self.device, dtype=torch.float32))
        
        x = radius * torch.cos(el) * torch.sin(az)
        y = radius * torch.sin(el)
        z = radius * torch.cos(el) * torch.cos(az)
        camera_pos = torch.stack([x, y, z])

        i, j = torch.meshgrid(torch.linspace(-1, 1, W), torch.linspace(-1, 1, H), indexing='xy')
        i, j = i.to(self.device), j.to(self.device)
        
        tan_fov = torch.tan(torch.deg2rad(torch.tensor(fov / 2, device=self.device, dtype=torch.float32)))
        dirs = torch.stack([i * tan_fov, -j * tan_fov, -torch.ones_like(i)], dim=-1)
        
        z_axis = F.normalize(camera_pos, dim=0)
        world_up = torch.tensor([0.0, 1.0, 0.0], device=self.device, dtype=torch.float32)
        x_axis = F.normalize(torch.cross(world_up, z_axis, dim=0), dim=0)
        y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=0), dim=0)
        
        R = torch.stack([x_axis, y_axis, z_axis], dim=1)
        ray_dirs = (dirs @ R.T)
        ray_origins = camera_pos.expand_as(ray_dirs)
        
        return ray_origins, ray_dirs

    # --- THIS LINE WAS MISSING THE 'background' ARGUMENT ---
    def render(self, voxel_grid, ray_origins, ray_dirs, num_samples=64, background="white"):
        # 1. Sample Points
        z_vals = torch.linspace(0.5, 4.5, num_samples, device=self.device) 
        z_vals_expanded = z_vals.view(1, 1, -1, 1)
        points = ray_origins.unsqueeze(2) + ray_dirs.unsqueeze(2) * z_vals_expanded
        
        # 2. Query Grid
        grid_coords = points.view(1, -1, 1, 1, 3)
        input_grid = voxel_grid.unsqueeze(0)
        sampled_features = F.grid_sample(input_grid, grid_coords, align_corners=True)
        sampled_features = sampled_features.view(4, -1, num_samples).permute(1, 2, 0)
        
        rgb = torch.sigmoid(sampled_features[..., :3])
        density = F.softplus(sampled_features[..., 3])
        
        # 3. Accumulate
        dists = z_vals[1:] - z_vals[:-1]
        dists = torch.cat([dists, torch.tensor([1e10], device=self.device)])
        
        alpha = 1.0 - torch.exp(-density * dists)
        weights = alpha * torch.cumprod(1.0 - alpha + 1e-10, dim=1)
        
        accumulated_color = (weights.unsqueeze(-1) * rgb).sum(dim=1)
        accumulated_alpha = weights.sum(dim=1).unsqueeze(-1) # Total opacity
        
        # 4. Background Blending
        if background == "white":
            bg_color = torch.ones_like(accumulated_color)
        else:
            bg_color = torch.zeros_like(accumulated_color) # Black
            
        final_color = accumulated_color + bg_color * (1.0 - accumulated_alpha)
        
        H = int(torch.sqrt(torch.tensor(final_color.shape[0])))
        return final_color.view(H, H, 3)