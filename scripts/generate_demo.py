import torch
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.physics_loader import TrajectoryDataset

def create_static_demo(save_path):
    print("🎨 Generating 3D Trajectory Demo...")
    gen = TrajectoryDataset(num_samples=5)
    t = torch.linspace(0, 2, 50)
    
    # Generate a single sample
    v0 = torch.tensor([5.0])
    angle = torch.tensor([np.pi/4])
    g = 9.81
    x = v0 * torch.cos(angle) * t
    y = v0 * torch.sin(angle) * t - 0.5 * g * t**2
    z = torch.zeros_like(t)
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x.numpy(), z.numpy(), y.numpy(), label='Ground Truth Physics', linewidth=3, color='blue')
    ax.scatter(x.numpy(), z.numpy(), y.numpy(), c=y.numpy(), cmap='viridis', s=20)
    
    ax.set_title("PhysiGen-WM: Newtonian Trajectory Baseline", fontsize=15)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_zlabel("Y (m / Height)")
    ax.legend()
    
    plt.savefig(save_path)
    print(f"✅ Demo plot saved: {save_path}")

if __name__ == "__main__":
    import numpy as np
    os.makedirs("docs/assets", exist_ok=True)
    create_static_demo("docs/assets/trajectory_demo.png")
