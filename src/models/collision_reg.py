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

    def forward(self, xyz, scales, rotations, velocities=None):
        """
        xyz: (Batch, Num_Gaussians, 3)
        scales: (Batch, Num_Gaussians, 3) - Anisotropic scaling factors
        rotations: (Batch, Num_Gaussians, 4) - Quaternions for orientation
        velocities: (Batch, Num_Gaussians, 3) - Latent velocities (optional)
        """
        B, N, _ = xyz.shape
        diff = xyz.unsqueeze(2) - xyz.unsqueeze(1) # (B, N, N, 3)
        dist_sq = torch.sum(diff**2, dim=-1) + 1e-8
        dist = torch.sqrt(dist_sq)
        direction = diff / dist.unsqueeze(-1) # Normalized unit vectors (B, N, N, 3)

        # 1. Orientation-Aware Projection (OAP)
        # Compute the projection of the ellipsoid covariance along the interaction vector.
        # R is extracted from quaternions q.
        qw, qx, qy, qz = rotations.unbind(-1)
        # Construct rotation matrices (Batch, N, 3, 3)
        R = torch.stack([
            1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw,
            2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw,
            2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2
        ], dim=-1).reshape(B, N, 3, 3)

        # S = diag(scales)
        S = torch.diag_embed(scales)
        # Directional radius squared: r^2 = u^T (R S S R^T) u
        # We compute u^T R S for all interaction directions
        # direction: (B, N, N, 3), R: (B, N, 3, 3)
        u_R = torch.einsum('bnmi,bnij->bnmj', direction, R) 
        u_R_S = u_R * scales.unsqueeze(1) # (B, N, N, 3)
        r_directional = torch.norm(u_R_S, dim=-1) # (B, N, N)

        # 2. Geometric Overlap Constraint
        # Sum of directional radii for the pair (i, j)
        # r_total[b, i, j] = r_directional[b, i, j] + r_directional[b, j, i]
        r_total = r_directional + r_directional.transpose(1, 2)
        
        # 3. Spectral Repulsion & Lorentzian Potential
        overlap = torch.relu(r_total - dist)
        spectral_repulsion = (overlap**2) / (1 + 0.1 * overlap)

        # 4. Multi-Scale Kinetic Dissipation (MSKD)
        # Penalizes relative velocity in the direction of overlap to dampen collisions.
        mskd_loss = torch.tensor(0.0, device=xyz.device)
        if velocities is not None:
            # Relative velocity v_ij = v_i - v_j
            v_rel = velocities.unsqueeze(2) - velocities.unsqueeze(1) # (B, N, N, 3)
            # Dot product with interaction direction: approach speed
            v_approach = torch.sum(v_rel * direction, dim=-1) # (B, N, N)
            # Apply damping only when moving towards each other (v_approach < 0)
            mskd_loss = torch.sum(overlap * torch.relu(-v_approach)) / (B * N * N)
        
        # Mask out self-distances
        mask = torch.eye(N, device=xyz.device).bool().unsqueeze(0)
        spectral_repulsion = spectral_repulsion.masked_fill(mask, 0.0)
        
        return torch.mean(spectral_repulsion) + 0.1 * mskd_loss
