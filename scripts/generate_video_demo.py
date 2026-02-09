import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.physigen import PhysiGen3D

def generate_video_demo(save_path):
    print("🎬 Initializing PhysiGen-WM Video Engine...")
    
    config = {
        'model': {
            'hidden_dim': 512,
            'latent_dim': 128
        }
    }
    
    # Initialize the 3D model
    model = PhysiGen3D(config)
    
    # Text prompt simulation
    text_embed = torch.randn(1, 512)
    # Define a sequence of 40 frames
    seq_len = 40
    time_seq = torch.linspace(0, 2, seq_len) 
    
    print("🧠 Solving Lagrangian ODEs for 40 frames...")
    with torch.no_grad():
        # scene_evolution: (Batch, Seq_Len, Num_Gaussians, 14)
        scene_evolution, _ = model(text_embed, time_seq)
    
    # Prepare Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#000000')
    fig.patch.set_facecolor('#000000')
    
    def update(frame_idx):
        ax.clear()
        # Set persistent axis limits to show motion clearly
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.set_zlim([-15, 5])
        ax.axis('off')
        
        frame_data = scene_evolution[0, frame_idx]
        xyz = frame_data[:, :3].numpy()
        rgb = frame_data[:, 10:13].numpy()
        opacity = frame_data[:, 13].numpy()
        
        # Plot the Gaussian collective
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 
                   c=rgb, 
                   s=opacity * 50, 
                   alpha=0.5,
                   edgecolors='none')
        
        ax.set_title(f"PhysiGen-WM: Latent Physics Evolution [Frame {frame_idx}]", color='white')

    print("📼 Rendering frames to animation...")
    ani = animation.FuncAnimation(fig, update, frames=seq_len, interval=50)
    
    # Save as GIF/MP4
    # Using pillow for GIF as it's more likely to be on the system
    ani.save(save_path, writer='pillow', fps=20)
    print(f"✅ Video Demo saved: {save_path}")

if __name__ == "__main__":
    os.makedirs("docs/assets", exist_ok=True)
    generate_video_demo("docs/assets/physigen_evolution.gif")
