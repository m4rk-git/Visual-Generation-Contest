import torch
import torch.nn as nn


class LearnableVoxelGrid(nn.Module):
    def __init__(self, resolution=64, device="cuda"):
        super().__init__()
        self.resolution = resolution
        # 4 Channels: R, G, B, Density
        # Initialize with low density (mostly empty) and random colors
        self.grid = nn.Parameter(
            torch.randn(4, resolution, resolution, resolution, device=device) * 0.1
        )

    def forward(self):
        return self.grid
