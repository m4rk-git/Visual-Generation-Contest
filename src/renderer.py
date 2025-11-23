import torch
import torch.nn.functional as F


class SimpleVolumeRenderer:
    def __init__(self, device="cuda"):
        self.device = device

    def get_camera_rays(self, H, W, fov=60.0, elevation=0.0, azimuth=0.0, radius=2.5):
        """Generates rays for a camera at a specific angle"""
        # 1. Camera Position (Spherical Coordinates)
        el, az = torch.deg2rad(torch.tensor(elevation)), torch.deg2rad(
            torch.tensor(azimuth)
        )
        x = radius * torch.cos(el) * torch.sin(az)
        y = radius * torch.sin(el)
        z = radius * torch.cos(el) * torch.cos(az)
        camera_pos = torch.tensor([x, y, z], device=self.device)

        # 2. Ray Directions
        i, j = torch.meshgrid(
            torch.linspace(-1, 1, W), torch.linspace(-1, 1, H), indexing="xy"
        )
        i, j = i.to(self.device), j.to(self.device)

        # FOV adjustments
        tan_fov = torch.tan(torch.deg2rad(torch.tensor(fov / 2)))
        dirs = torch.stack([i * tan_fov, -j * tan_fov, -torch.ones_like(i)], dim=-1)

        # Rotate rays to look at origin (LookAt matrix simplified)
        # For simple optimization, we assume object is at (0,0,0)
        # We construct a rotation matrix R based on camera position
        z_axis = F.normalize(camera_pos, dim=0)
        x_axis = F.normalize(
            torch.cross(torch.tensor([0.0, 1.0, 0.0], device=self.device), z_axis),
            dim=0,
        )
        y_axis = F.normalize(torch.cross(z_axis, x_axis), dim=0)
        R = torch.stack([x_axis, y_axis, z_axis], dim=1)

        ray_dirs = dirs @ R.T
        ray_origins = camera_pos.expand_as(ray_dirs)

        return ray_origins, ray_dirs

    def render(self, voxel_grid, ray_origins, ray_dirs, num_samples=64):
        """
        Marches rays through the voxel grid (The core rendering loop)
        voxel_grid: [4, D, H, W] tensor (RGBA)
        """
        # 1. Sample points along rays
        # We assume the object fits in a [-1, 1] cube
        z_vals = torch.linspace(
            0.5, 4.5, num_samples, device=self.device
        )  # Near/Far planes

        # [H, W, Samples, 3]
        points = ray_origins.unsqueeze(2) + ray_dirs.unsqueeze(2) * z_vals.unsqueeze(3)

        # 2. Query Voxel Grid (Trilinear Interpolation)
        # Normalize points to [-1, 1] for grid_sample
        grid_coords = points.view(1, -1, 1, 1, 3)  # [1, N_points, 1, 1, 3]

        # Voxel grid expected shape: [1, 4, D, H, W]
        # We permute voxel_grid to [1, 4, D, H, W]
        input_grid = voxel_grid.unsqueeze(0)

        # Sample
        sampled_features = F.grid_sample(input_grid, grid_coords, align_corners=True)
        sampled_features = sampled_features.view(4, -1, num_samples).permute(
            1, 2, 0
        )  # [H*W, Samples, 4]

        rgb = torch.sigmoid(sampled_features[..., :3])
        density = F.softplus(sampled_features[..., 3])  # Density must be positive

        # 3. Volume Rendering Equation (Accumulate color)
        dists = z_vals[1:] - z_vals[:-1]
        dists = torch.cat([dists, torch.tensor([1e10], device=self.device)])

        alpha = 1.0 - torch.exp(-density * dists)
        weights = alpha * torch.cumprod(1.0 - alpha + 1e-10, dim=1)

        # Sum weighted colors
        final_color = (weights.unsqueeze(-1) * rgb).sum(dim=1)

        # Reshape back to image
        H = int(torch.sqrt(torch.tensor(final_color.shape[0])))
        return final_color.view(H, H, 3)
