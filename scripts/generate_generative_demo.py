import torch
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.physigen import PhysiGen3D

def generate_gaussian_world_preview(save_path):
    print("✨ Initializing PhysiGen-WM Generative Engine...")
    
    # Mock config
    config = {
        'model': {
            'hidden_dim': 512,
            'latent_dim': 128
        }
    }
    
    # Initialize the 3D model
    model = PhysiGen3D(config)
    
    # Simulate a text embedding (as if user typed "A cluster of energy falling")
    text_embed = torch.randn(1, 512)
    time_seq = torch.linspace(0, 1, 10) # 10 time steps
    
    # Generate the 3D World (Sequence of 1024 Gaussians per frame)
    print("🔮 Generating 3D Gaussian collectives...")
    with torch.no_grad():
        # scene_evolution: (Batch, Seq_Len, Num_Gaussians, 14)
        scene_evolution, _ = model(text_embed, time_seq)
    
    # Take the first frame to visualize the "World"
    frame_0 = scene_evolution[0, 0] # (1024, 14)
    xyz = frame_0[:, :3].numpy()
    rgb = frame_0[:, 10:13].numpy()
    opacity = frame_0[:, 13].numpy()
    
    # Create a dense 3D scatter to represent the Gaussian Splats
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot with generated colors and sizes based on opacity
    img = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 
                     c=rgb, 
                     s=opacity * 100, 
                     alpha=0.6,
                     edgecolors='none')
    
    ax.set_title("PhysiGen-WM: Generated 3D Gaussian World (Un-trained Prototype)", fontsize=15)
    ax.set_facecolor('#111111') # Dark theme for research vibe
    fig.patch.set_facecolor('#111111')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.tick_params(colors='white')
    
    # Set view angle
    ax.view_init(elev=20., azim=45)
    
    plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"✅ Generative Demo saved: {save_path}")

if __name__ == "__main__":
    os.makedirs("docs/assets", exist_ok=True)
    generate_gaussian_world_preview("docs/assets/generative_world_preview.png")
