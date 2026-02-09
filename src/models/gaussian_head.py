import torch
import torch.nn as nn

class GaussianSplattingHead(nn.Module):
    """
    PhysiGen-WM Specialized Gaussian Head.
    Innovation: Decouples geometric transforms from appearance latent codes
    to ensure more stable physical deformation.
    """
    def __init__(self, latent_dim, num_gaussians=1024):
        super().__init__()
        self.num_gaussians = num_gaussians
        
        # Branch 1: Geometric deformation (position, scale, rotation)
        self.geometry_gen = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, num_gaussians * 10) # 3 xyz, 4 rot, 3 scale
        )
        
        # Branch 2: Appearance & Density (color, opacity)
        self.appearance_gen = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, num_gaussians * 4) # 3 rgb, 1 opacity
        )
        
    def forward(self, z):
        batch_size = z.shape[0]
        
        # 1. Generate Geometry
        geom = self.geometry_gen(z).view(batch_size, self.num_gaussians, 10)
        xyz = geom[..., :3]
        rotation = torch.nn.functional.normalize(geom[..., 3:7], dim=-1)
        scale = torch.exp(geom[..., 7:10] - 5.0) # Bias towards small primitives
        
        # 2. Generate Appearance
        app = self.appearance_gen(z).view(batch_size, self.num_gaussians, 4)
        color = torch.sigmoid(app[..., :3])
        opacity = torch.sigmoid(app[..., 3:4])
        
        return torch.cat([xyz, rotation, scale, color, opacity], dim=-1)
