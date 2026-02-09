import torch
import torch.nn as nn

class CollisionRegularizer(nn.Module):
    """
    Innovation: Neural Collision Avoidance Loss.
    Penalizes Gaussian primitives that overlap beyond a certain density threshold,
    simulating solid-body constraints in the latent space.
    """
    def __init__(self, radius=0.05):
        super().__init__()
        self.radius = radius

    def forward(self, xyz):
        """
        xyz: (Batch, Num_Gaussians, 3)
        """
        # Calculate pairwise distances (vectorized)
        # Result: (Batch, N, N)
        dist_mat = torch.cdist(xyz, xyz, p=2)
        
        # Mask out self-distances
        n = xyz.shape[1]
        mask = torch.eye(n, device=xyz.device).bool()
        dist_mat = dist_mat.masked_fill(mask, float('inf'))
        
        # Penalize distances smaller than radius
        collision_loss = torch.relu(self.radius - dist_mat)
        return torch.mean(collision_loss**2)
