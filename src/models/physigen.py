import torch
import torch.nn as nn
import torch.nn.functional as F
from .gaussian_head import GaussianSplattingHead
from .text_adapter import TextAdapter
from .collision_reg import CollisionRegularizer

class LagrangianODESolver(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.hamiltonian_field = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim)
        )

    def ode_func(self, z):
        return self.hamiltonian_field(z)

    def forward(self, z0, t_steps):
        dt = t_steps[1] - t_steps[0]
        zt = z0
        outputs = []
        for _ in range(len(t_steps)):
            k1 = self.ode_func(zt)
            k2 = self.ode_func(zt + dt/2 * k1)
            k3 = self.ode_func(zt + dt/2 * k2)
            k4 = self.ode_func(zt + dt * k3)
            zt = zt + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            outputs.append(zt)
        return torch.stack(outputs, dim=1)

class PhysiGen3D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config['model']['hidden_dim']
        self.latent_dim = config['model']['latent_dim']
        
        self.text_adapter = TextAdapter(embed_dim=512, latent_dim=self.latent_dim)
        self.physics_engine = LagrangianODESolver(self.latent_dim)
        
        # Upgraded: Dual-Path Attention for Long-Sequence Dependency
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=self.latent_dim, 
            num_heads=4, 
            batch_first=True
        )
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            num_heads=4,
            batch_first=True
        )
        self.conflict_gate = nn.Parameter(torch.ones(1) * 0.1)
        self.temporal_gate = nn.Parameter(torch.ones(1) * 0.05)
        
        self.gaussian_head = GaussianSplattingHead(self.latent_dim, num_gaussians=1024)
        self.collision_reg = CollisionRegularizer(radius=0.05)

    def forward(self, text_embed, time_seq, num_refinement_steps=1):
        z_init, physics_priors = self.text_adapter(text_embed)
        for _ in range(num_refinement_steps):
            z_traj = self.physics_engine(z_init, time_seq)
            consist_err = self.calculate_conservation_loss(z_traj)
            if z_init.requires_grad:
                z_init = z_init - 0.01 * torch.autograd.grad(consist_err, z_init, retain_graph=True)[0]

        # Spatial-Temporal Dual Attention
        s_attn_out, _ = self.spatial_attn(z_traj, z_traj, z_traj)
        t_attn_out, _ = self.temporal_attn(z_traj, z_traj, z_traj) # Could add causal mask here
        
        z_refined = z_traj + torch.tanh(self.conflict_gate) * s_attn_out + torch.tanh(self.temporal_gate) * t_attn_out
        
        scene_evolution = []
        for t in range(z_refined.shape[1]):
            gaussians = self.gaussian_head(z_refined[:, t, :])
            scene_evolution.append(gaussians)
        return torch.stack(scene_evolution, dim=1), physics_priors

    def calculate_conservation_loss(self, z_traj, dt=0.05):
        # Implementation of Causal-Spectral Penalty
        velocity = (z_traj[:, 1:] - z_traj[:, :-1]) / dt
        kinetic_energy = 0.5 * torch.sum(velocity**2, dim=-1)
        potential_energy = torch.norm(z_traj[:, 1:], dim=-1)
        hamiltonian = kinetic_energy + potential_energy
        dH_dt = (hamiltonian[:, 1:] - hamiltonian[:, :-1]) / dt
        
        # Causal weighting: prioritize early time steps to stabilize the ODE
        t_steps = z_traj.shape[1] - 1
        causal_weights = torch.exp(-torch.linspace(0, 1, t_steps)).to(z_traj.device)
        weighted_loss = torch.mean((dH_dt**2) * causal_weights)
        
        return weighted_loss

    def calculate_vco_loss(self, z_traj):
        residuals = z_traj[:, 1:] - z_traj[:, :-1]
        res_var = torch.var(residuals, dim=1)
        return torch.mean(res_var)
