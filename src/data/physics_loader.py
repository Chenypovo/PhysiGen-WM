import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TrajectoryDataset(Dataset):
    """
    PhysiGen-WM Multi-Scenario Physics Generator.
    Generates: 
    1. Projectile motion (Gravity)
    2. Linear motion (Momentum)
    3. Harmonic oscillation (Spring-like constraints)
    """
    def __init__(self, num_samples=1000, time_steps=50):
        self.num_samples = num_samples
        self.time_steps = time_steps
        
    def generate_batch(self, save_path):
        print(f"🧬 Synthesizing {self.num_samples} physical scenarios...")
        t = torch.linspace(0, 2, self.time_steps)
        g = 9.81
        
        all_data = []
        
        for i in range(self.num_samples):
            scenario_type = np.random.choice(['projectile', 'linear', 'oscillator'])
            
            if scenario_type == 'projectile':
                v0 = torch.rand(1) * 15
                angle = torch.rand(1) * 1.5
                x = v0 * torch.cos(angle) * t
                y = v0 * torch.sin(angle) * t - 0.5 * g * t**2
                z = torch.zeros_like(t)
            elif scenario_type == 'linear':
                v = torch.randn(3) * 5
                p0 = torch.randn(3) * 2
                pos = p0.unsqueeze(1) + v.unsqueeze(1) * t
                x, y, z = pos[0], pos[1], pos[2]
            else: # Oscillator
                freq = torch.rand(1) * 5
                amp = torch.rand(3) * 2
                pos = amp.unsqueeze(1) * torch.sin(freq * t)
                x, y, z = pos[0], pos[1], pos[2]
                
            traj = torch.stack([x, y, z], dim=1)
            all_data.append(traj)
            
        dataset_tensor = torch.stack(all_data)
        torch.save(dataset_tensor, os.path.join(save_path, "physics_train_v1.pt"))
        print(f"✅ Dataset saved to {save_path}")

if __name__ == "__main__":
    import sys
    save_dir = "/Users/starrystark/Desktop/PhysiGen-WM/data/synthetic"
    gen = TrajectoryDataset(num_samples=5000)
    gen.generate_batch(save_dir)
