import torch
import torch.nn as nn

class LearnableVoxelGrid(nn.Module):
    def __init__(self, resolution=64, device="cuda"):
        super().__init__()
        self.resolution = resolution
        
        # 1. Coordinate Grid
        x = torch.linspace(-1, 1, resolution)
        y = torch.linspace(-1, 1, resolution)
        z = torch.linspace(-1, 1, resolution)
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
        
        # 2. Distance
        dist = torch.sqrt(grid_x**2 + grid_y**2 + grid_z**2)
        
        # 3. Density (Solid Sphere)
        init_density = -5.0 + 10.0 * torch.exp(-dist**2 / 0.25)
        init_density = init_density.to(device)
        
        # 4. Color Initialization (FIX: RANDOM NOISE)
        # Random values break the symmetry trap, allowing colors to change.
        init_colors = torch.randn(3, resolution, resolution, resolution, device=device) * 0.1
        
        # Stack
        params = torch.cat([init_colors, init_density.unsqueeze(0)], dim=0)
        
        self.grid = nn.Parameter(params)
    
    def forward(self):
        return self.grid