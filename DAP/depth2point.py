import numpy as np
import cv2
import os
from plyfile import PlyData, PlyElement
import utils3d   # local utility library
import torch

def spherical_uv_to_directions(uv: np.ndarray):
    theta, phi = (1 - uv[..., 0]) * (2 * np.pi), uv[..., 1] * np.pi
    directions = np.stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ], axis=-1)
    return directions

def spherical_uv_to_directions_torch(uv: torch.Tensor):
    """
    Torch implementation: convert spherical UV coordinates to direction vectors.
    Args:
        uv: UV coords, shape [H, W, 2] or [B, H, W, 2]
    Returns:
        directions: direction vectors, shape [H, W, 3] or [B, H, W, 3]
    """
    theta = (1 - uv[..., 0]) * (2 * torch.pi)
    phi = uv[..., 1] * torch.pi
    directions = torch.stack([
        torch.sin(phi) * torch.cos(theta),
        torch.sin(phi) * torch.sin(theta),
        torch.cos(phi)
    ], dim=-1)
    return directions

def save_3d_points(points: np.array, colors: np.array, mask: np.array, filename: str):
    points = points.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    mask = mask.reshape(-1)

    vertex_data = np.empty(mask.sum(), dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])
    vertex_data['x'] = points[mask, 0]
    vertex_data['y'] = points[mask, 1]
    vertex_data['z'] = points[mask, 2]
    vertex_data['red'] = colors[mask, 0]
    vertex_data['green'] = colors[mask, 1]
    vertex_data['blue'] = colors[mask, 2]

    vertex_element = PlyElement.describe(vertex_data, 'vertex', comments=['point cloud'])
    PlyData([vertex_element], text=True).write(filename)

def depth2pointcloud(depth_path: str, image_path: str, out_ply: str):
    # Read depth map
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"could not read depth map: {depth_path}")

# If the depth was read back as 3-channel, convert to grayscale
    if depth.ndim == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

    depth = depth.astype(np.float32)
    # Normalize depth to a sensible range (assume 8-bit or 16-bit input)
    if depth.dtype == np.uint8:
        depth = depth / 255.0
    elif depth.dtype == np.uint16:
        depth = depth / 65535.0

    h, w = depth.shape
    uv = utils3d.numpy.image_uv(width=w, height=h)   # [H,W,2]
    dirs = spherical_uv_to_directions(uv)           # [H,W,3]
    points = depth[..., None] * dirs                # [H,W,3]

    # Read color image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read color image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[:2] != (h, w):
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

    mask = depth > 0
    save_3d_points(points, image, mask, out_ply)
    print(f"Saved point cloud to {out_ply}")


def depth2pts(depth):
    # Read depth map
    depth.squeeze(1)  # B,1,H,W -> B,H,W (values in [0, 1])

    depth = depth.astype(np.float32)

    b, h, w = depth.shape
    
    # Build UV grid [H, W, 2]
    uv = utils3d.numpy.image_uv(width=w, height=h)
    # Convert to spherical directions [H, W, 3]
    dirs = spherical_uv_to_directions(uv)
    
    # Broadcast to a 3D point cloud [B, H, W, 3]
    points = depth[..., None] * dirs[None, ...]  # [B,H,W,1] * [1,H,W,3] = [B,H,W,3]
    
    return points

def depth2pts_torch(depth):
    """
    Convert a depth map (batched, torch) to a 3D point cloud.
    Args:
        depth: torch tensor of shape [B, H, W] or [B, 1, H, W], values in [0, 1]
    Returns:
        points: torch tensor of shape [B, H, W, 3]
    """
    # Ensure input is float32
    depth = depth.float()
    
    # If input is [B, 1, H, W], squeeze to [B, H, W]
    if depth.dim() == 4 and depth.shape[1] == 1:
        depth = depth.squeeze(1)  # B,1,H,W -> B,H,W
    
    b, h, w = depth.shape
    
    # Build UV grid [H, W, 2]
    uv = utils3d.numpy.image_uv(width=w, height=h)
    # Convert to torch tensor and move to the same device
    uv = torch.from_numpy(uv).to(depth.device).to(depth.dtype)
    
    # Convert to spherical directions [H, W, 3]
    dirs = spherical_uv_to_directions_torch(uv)
    
    # Broadcast to a 3D point cloud [B, H, W, 3]
    points = depth[..., None] * dirs[None, ...]  # [B,H,W,1] * [1,H,W,3] = [B,H,W,3]
    
    return points



if __name__ == "__main__":
    # depth_path = "/path/to/depth.png"   # your depth PNG
    # image_path = "/path/to/rgb.png"     # your color PNG
    # out_ply = "/home/tione/notebook/home/wenxuan/PanDA_dualhead_alltrain/visual_nonormalloss/pts/scene001_points.ply"
    path_ = "/home/tione/notebook/home/wenxuan/DAM_dualhead_alltrain/vis_exp2_insta820/rgb/"
    for file in os.listdir(path_):
        image_path = os.path.join(path_, file)
        depth_path = os.path.join(path_.replace("rgb", "depth"), file.replace("rgb", "depth"))
        os.makedirs(path_.replace("rgb", "pts"), exist_ok=True)
        out_ply = os.path.join(path_.replace("rgb", "pts"), file.replace(".png", ".ply"))
        depth2pointcloud(depth_path, image_path, out_ply)
        
    # os.makedirs("output", exist_ok=True)
    # depth2pointcloud(depth_path, image_path, out_ply)
