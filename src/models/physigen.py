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
        # Symplectic MLP: Predicts dH/dq (positional change) and dH/dp (momentum change)
        self.hamiltonian_field = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim)
        )
        # Spectral Initialization: Xavier-normalized for harmonic stability
        for m in self.hamiltonian_field:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1e-2)
                nn.init.constant_(m.bias, 0)

    def ode_func(self, z):
        # Symplectic Gradient: J * grad(H)
        # For simplicity, we model the field directly, but enforce J-structure in loss.
        return self.hamiltonian_field(z)

    def forward(self, z_init, t_steps):
        dt = t_steps[1] - t_steps[0]
        zt = z_init
        outputs = []
        for _ in range(len(t_steps)):
            # 4th-order Runge-Kutta (RK4)
            k1 = self.ode_func(zt)
            k2 = self.ode_func(zt + dt/2 * k1)
            k3 = self.ode_func(zt + dt/2 * k2)
            k4 = self.ode_func(zt + dt * k3)
            zt = zt + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            outputs.append(zt)
        
        # New: Latent Phase-Space Drift Correction (LPSDC)
        # Prevents long-term temporal drift by projecting back to the Textual Manifold
        z_stack = torch.stack(outputs, dim=1)
        z_mean = torch.mean(z_stack, dim=1, keepdim=True)
        z_drift_corrected = z_stack - 0.05 * (z_mean - z_init.unsqueeze(1))
        
        # New: Spectral initialization adjustment for periodic consistency
        # Rescale the trajectory by the dominant frequency component to ensure harmonic stability
        fft_z = torch.fft.rfft(z_drift_corrected, dim=1)
        magnitudes = torch.abs(fft_z)
        dominant_freq = torch.argmax(magnitudes, dim=1, keepdim=True)
        # We don't apply hard rescaling yet, just calculate for logging/future spectral loss terms
        
        return z_drift_corrected

    def calculate_jacobian_loss(self, z):
        """
        Enforce Symplectic Structure (J-orthogonality) in the phase-space flow.
        """
        z = z.detach().requires_grad_(True)
        dz_dt = self.ode_func(z)
        
        # Compute Jacobian of the ODE field
        batch_size, dim = z.shape
        jacobian = []
        for i in range(dim):
            grads = torch.autograd.grad(dz_dt[:, i].sum(), z, create_graph=True)[0]
            jacobian.append(grads)
        M = torch.stack(jacobian, dim=1) # (B, dim, dim)
        
        # Symplectic matrix J
        half_dim = dim // 2
        J = torch.zeros((dim, dim), device=z.device)
        J[:half_dim, half_dim:] = torch.eye(half_dim)
        J[half_dim:, :half_dim] = -torch.eye(half_dim)
        
        # Loss: M^T J M - J = 0
        symplectic_err = torch.matmul(torch.matmul(M.transpose(1, 2), J), M) - J
        return torch.mean(symplectic_err**2)

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
        # Implementation of Spectral-Causal Penalty
        velocity = (z_traj[:, 1:] - z_traj[:, :-1]) / dt
        kinetic_energy = 0.5 * torch.sum(velocity**2, dim=-1)
        potential_energy = torch.norm(z_traj[:, 1:], dim=-1)
        hamiltonian = kinetic_energy + potential_energy
        dH_dt = (hamiltonian[:, 1:] - hamiltonian[:, :-1]) / dt
        
        # New: Phase-Space Entropy Regularization
        # Prevents the latent trajectory from collapsing into a single fixed point
        phase_dist = torch.cdist(z_traj, z_traj)
        entropy_reg = -torch.mean(torch.log(phase_dist + 1e-6))

        # Spectral-Causal Refinement: Compute Fourier magnitudes for high-freq damping
        # This suppresses jitter in long-horizon rollout
        fft_z = torch.fft.rfft(z_traj, dim=1)
        spectral_density = torch.abs(fft_z)
        high_freq_penalty = torch.mean(spectral_density[:, -5:]) # Penalize top 5 high-freq bins

        # Causal weighting: prioritize early time steps to stabilize the initial rollout
        t_steps = z_traj.shape[1] - 1
        causal_weights = torch.exp(-torch.linspace(0, 1, t_steps-1)).to(z_traj.device)
        weighted_loss = torch.mean((dH_dt**2) * causal_weights)
        
        # New: Symplectic Jacobian Consistency
        # Checks for preservation of phase-space area/volume
        z_sample = z_traj[:, 0, :]
        symplectic_loss = self.physics_engine.calculate_jacobian_loss(z_sample)
        
        # Total Spectral-Causal Physical Loss
        total_phys_loss = weighted_loss + 0.01 * entropy_reg + 0.1 * high_freq_penalty + 0.2 * symplectic_loss
        
        # New: Temporal-Spectral Coherence Loss (TSCL)
        # Ensure that the temporal gradient in phase-space matches the spectral energy distribution
        grad_z = (z_traj[:, 1:] - z_traj[:, :-1]) / dt
        spectral_grad = torch.abs(torch.fft.rfft(grad_z, dim=1))
        # Align low-frequency gradients with higher weights to stabilize global motion
        tscl = torch.mean(spectral_grad[:, :3]) # Focus on first 3 frequency bins
        
        return total_phys_loss + 0.05 * tscl

    def calculate_vco_loss(self, z_traj):
        residuals = z_traj[:, 1:] - z_traj[:, :-1]
        res_var = torch.var(residuals, dim=1)
        return torch.mean(res_var)
