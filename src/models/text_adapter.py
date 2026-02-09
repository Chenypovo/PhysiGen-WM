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
        
        # Mapping to physical priors (mass, gravity_scale, initial_velocity)
        self.physics_prior_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 7) # [v_x, v_y, v_z, mass, gravity_mult, size, color_id]
        )

    def forward(self, text_features):
        # text_features: (Batch, Embed_Dim) assuming pre-pooled or single vector
        z_text = self.text_processor(text_features)
        physics_priors = self.physics_prior_head(z_text)
        
        # v_z and gravity_mult are usually positive for falling objects
        physics_priors[:, 4] = torch.sigmoid(physics_priors[:, 4]) * 2.0 # Gravity multiplier 0-2
        physics_priors[:, 3] = torch.exp(physics_priors[:, 3]) # Mass must be positive
        
        return z_text, physics_priors
