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
        dist = torch.sqrt(grid_x**2 + grid_y**2 + grid_z**2)
        
        # 2. Density (Solid Sphere)
        init_density = -5.0 + 10.0 * torch.exp(-dist**2 / 0.25)
        self.density = nn.Parameter(init_density.unsqueeze(0).to(device))
        
        # 3. Color (FORCE RED INITIALIZATION)
        # Channel 0 (Red) = 2.0 (High)
        # Channel 1 (Green) = -2.0 (Low)
        # Channel 2 (Blue) = -2.0 (Low)
        # Sigmoid(2.0) ≈ 0.88, Sigmoid(-2.0) ≈ 0.12
        init_colors = torch.zeros(3, resolution, resolution, resolution, device=device)
        init_colors[0] = 2.0 
        init_colors[1] = -2.0
        init_colors[2] = -2.0
        
        # Add small noise so it can still learn
        init_colors += torch.randn_like(init_colors) * 0.1
        
        self.colors = nn.Parameter(init_colors)
        
    def forward(self):
        return torch.cat([self.colors, self.density], dim=0)