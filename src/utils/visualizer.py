import torch
import numpy as np

def export_to_ply(gaussian_params, file_path):
    """
    Exports generated Gaussian parameters to a .ply file for 3D visualization.
    gaussian_params: Tensor of shape (Num_Gaussians, 14)
    """
    # Extract data
    xyz = gaussian_params[:, :3].detach().cpu().numpy()
    rgb = (gaussian_params[:, 10:13].detach().cpu().numpy() * 255).astype(np.uint8)
    opacity = gaussian_params[:, 13:14].detach().cpu().numpy()

    num_verts = xyz.shape[0]
    
    header = f"""ply
format ascii 1.0
element vertex {num_verts}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property float opacity
end_header
"""
    with open(file_path, 'w') as f:
        f.write(header)
        for i in range(num_verts):
            f.write(f"{xyz[i,0]} {xyz[i,1]} {xyz[i,2]} {rgb[i,0]} {rgb[i,1]} {rgb[i,2]} {opacity[i,0]}\n")
    print(f"📦 3D Visualization file exported: {file_path}")
