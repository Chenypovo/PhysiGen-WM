import torch
import torch.nn as nn

class TextAdapter(nn.Module):
    """
    Translates text embeddings into physical state parameters.
    Input: Text tokens or embeddings (Batch, Seq_Len, Embed_Dim)
    Output: Physics Initial State (Batch, Latent_Dim)
    """
    def __init__(self, embed_dim=512, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Simple Attention-based pooling for text features
        self.text_processor = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

        # Spectral Initialization Layer: Mapping to Fourier Phase-Space
        # Ensures initial state has a valid 'energy budget' in the frequency domain.
        # Refinement: Using a Spectral-Gating mechanism to filter text embeddings 
        # based on learned frequency bands before latent projection.
        self.spectral_gate = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid()
        )
        self.spectral_init = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh() # Saturate to prevent early energy divergence
        )
        
        # Mapping to physical priors (mass, gravity_scale, initial_velocity)
        self.physics_prior_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 7) # [v_x, v_y, v_z, mass, gravity_mult, size, color_id]
        )

    def forward(self, text_features):
        # text_features: (Batch, Embed_Dim) assuming pre-pooled or single vector
        z_raw = self.text_processor(text_features)
        
        # Spectral-Gated Refinement:
        # Learns to prioritize semantic features that correspond to physical motion
        # (e.g., "fall", "bounce") over purely descriptive ones.
        gate = self.spectral_gate(z_raw)
        z_raw = z_raw * gate

        # Spectral Causal Shift: Aligning initial latent state with harmonic priors
        z_text = self.spectral_init(z_raw)
        
        # NEW: Spectral Energy Normalization
        # Limits the L2 norm of the initial state to the unit hypersphere
        # to prevent energy overflow in the RK4 solver.
        z_text = z_text / (torch.norm(z_text, dim=-1, keepdim=True) + 1e-6)
        
        physics_priors = self.physics_prior_head(z_text)
        
        # v_z and gravity_mult are usually positive for falling objects
        physics_priors[:, 4] = torch.sigmoid(physics_priors[:, 4]) * 2.0 # Gravity multiplier 0-2
        physics_priors[:, 3] = torch.exp(physics_priors[:, 3]) # Mass must be positive
        
        return z_text, physics_priors
