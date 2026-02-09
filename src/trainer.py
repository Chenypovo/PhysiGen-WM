import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import yaml
import os
from src.models.physigen import PhysiGen3D

class PhysiGenTrainer:
    """
    Research-grade trainer optimized for server environments (A100/H100 compatible).
    Features: Mixed Precision, Conservation Loss Integration, and Checkpointing.
    """
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PhysiGen3D(self.config).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=float(self.config['training']['learning_rate']))
        self.scaler = GradScaler(enabled=self.config['training']['use_amp'])
        self.criterion = nn.MSELoss()
        
        os.makedirs(self.config['training']['save_dir'], exist_ok=True)

    def train_step(self, text_embed, gt_trajectories):
        self.optimizer.zero_grad()
        
        time_seq = torch.linspace(0, 2, self.config['dynamics']['time_steps']).to(self.device)
        
        with autocast(enabled=self.config['training']['use_amp']):
            # Forward pass: Generate 3D Gaussian Sequence
            preds, physics_priors = self.model(text_embed, time_seq)
            
            # 1. Reconstruction Loss (Geometry)
            # Assuming gt_trajectories contains the mapped Gaussian centers/properties
            loss_recon = self.criterion(preds[..., :3], gt_trajectories[..., :3])
            
            # 2. Lagrangian Conservation Loss (The Innovation)
            loss_physics = self.model.calculate_conservation_loss(preds[..., :3])
            
            total_loss = loss_recon + self.config['training']['physics_loss_weight'] * loss_physics

        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return total_loss.item()

    def save_checkpoint(self, epoch):
        path = os.path.join(self.config['training']['save_dir'], f"checkpoint_ep{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"💾 Checkpoint saved: {path}")

if __name__ == "__main__":
    # Integration test for model instantiation
    trainer = PhysiGenTrainer('configs/default.yaml')
    print("🚀 Trainer initialized and ready for server deployment.")
