"""Panoramic video conditioning utilities.

Provides equi2pers / pers2equi conversion and mask generation for
conditioning Cosmos video generation on perspective video inputs.
Adapted from ARGUS (argus-code/src/pers2equi.py and train.py).

Core workflow:
  1. pers2equi_project: Project perspective video → equirectangular (partial coverage)
  2. generate_equirect_mask: Create spatial mask of which pano pixels are observed
  3. The masked equirectangular video + mask feed into Cosmos video2world
     conditioning (spatial inpainting)
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def rodrigues_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    """Convert a Rodrigues rotation vector to a rotation matrix."""
    theta = torch.norm(rotvec)
    if theta < 1e-8:
        return torch.eye(3, device=rotvec.device, dtype=rotvec.dtype)
    k = rotvec / theta
    K = torch.tensor([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0],
    ], device=rotvec.device, dtype=rotvec.dtype)
    return torch.eye(3, device=rotvec.device, dtype=rotvec.dtype) + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)


def _build_rotation_matrices(
    roll: torch.Tensor,
    pitch: torch.Tensor,
    yaw: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Build batched rotation matrices from roll/pitch/yaw angles (radians).

    Returns:
        R: (B, 3, 3) rotation matrices.
    """
    B = roll.shape[0]
    x_axis = torch.tensor([1.0, 0.0, 0.0], device=device)
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=device)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device)

    R_batch = torch.empty(B, 3, 3, device=device)
    for b in range(B):
        R1 = rodrigues_to_matrix(z_axis * yaw[b])
        R2 = rodrigues_to_matrix((R1 @ y_axis) * (-pitch[b]))
        R3 = rodrigues_to_matrix((R2 @ R1 @ x_axis) * (-roll[b]))
        R_batch[b] = R3 @ R2 @ R1
    return R_batch


def generate_equirect_mask(
    fov_x: float,
    roll: torch.Tensor,
    pitch: torch.Tensor,
    yaw: torch.Tensor,
    height: int = 512,
    width: int = 1024,
    hw_ratio: float = 2.0 / 3.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generate binary mask on equirectangular image showing which pixels
    are visible from a perspective camera with given parameters.

    Ref: ARGUS src/pers2equi.py generate_mask_batch()

    Args:
        fov_x: Horizontal field of view in degrees.
        roll, pitch, yaw: Camera rotation angles (radians), shape (T,).
        height, width: Equirectangular image dimensions.
        hw_ratio: Height/width ratio of the perspective camera.
        device: Torch device.

    Returns:
        mask: (T, 1, H, W) binary mask. 1 = visible from perspective camera.
    """
    roll = roll.float().to(device)
    pitch = pitch.float().to(device)
    yaw = yaw.float().to(device)
    T = roll.shape[0]

    len_x = math.tan(math.radians(fov_x / 2.0))
    fov_y = 2 * math.atan(math.tan(math.radians(fov_x) / 2) * hw_ratio) / math.pi * 180
    len_y = math.tan(math.radians(fov_y / 2.0))

    lon = torch.linspace(-math.pi, math.pi, width, device=device).unsqueeze(0).expand(height, width)
    lat = torch.linspace(math.pi / 2, -math.pi / 2, height, device=device).unsqueeze(1).expand(height, width)

    x_map = torch.cos(lon) * torch.cos(lat)
    y_map = torch.sin(lon) * torch.cos(lat)
    z_map = torch.sin(lat)
    xyz = torch.stack((x_map, y_map, z_map), dim=-1).view(-1, 3).T  # (3, H*W)

    R = _build_rotation_matrices(roll, pitch, yaw, device)
    R_inv = torch.linalg.inv(R)  # (T, 3, 3)

    xyz_rot = torch.matmul(R_inv, xyz).permute(0, 2, 1).view(T, height, width, 3)

    in_front = (xyz_rot[:, :, :, 0] > 0).float()
    xyz_norm = xyz_rot / (xyz_rot[:, :, :, 0:1] + 1e-8)

    in_fov = (
        (xyz_norm[:, :, :, 1].abs() < len_x)
        & (xyz_norm[:, :, :, 2].abs() < len_y)
    ).float()

    mask = in_front * in_fov
    return mask.unsqueeze(1)  # (T, 1, H, W)


def pers2equi_simple(
    pers_frames: torch.Tensor,
    fov_x: float,
    roll: torch.Tensor,
    pitch: torch.Tensor,
    yaw: torch.Tensor,
    equi_h: int = 512,
    equi_w: int = 1024,
    hw_ratio: float = 2.0 / 3.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project perspective frames onto equirectangular canvas.

    Simple grid-sample based projection. For each equirectangular pixel that
    falls within the perspective camera's FOV, sample from the perspective image.

    Args:
        pers_frames: (T, C, Hp, Wp) perspective video frames in [-1, 1].
        fov_x: Horizontal FOV in degrees.
        roll, pitch, yaw: Camera angles (radians), shape (T,).
        equi_h, equi_w: Output equirectangular dimensions.
        hw_ratio: Perspective camera h/w ratio.

    Returns:
        equi_frames: (T, C, equi_h, equi_w) projected equirectangular frames.
        mask: (T, 1, equi_h, equi_w) binary mask of projected pixels.
    """
    device = pers_frames.device
    T, C, Hp, Wp = pers_frames.shape

    mask = generate_equirect_mask(
        fov_x=fov_x, roll=roll, pitch=pitch, yaw=yaw,
        height=equi_h, width=equi_w, hw_ratio=hw_ratio, device=device,
    )

    len_x = math.tan(math.radians(fov_x / 2.0))
    fov_y = 2 * math.atan(math.tan(math.radians(fov_x) / 2) * hw_ratio) / math.pi * 180
    len_y = math.tan(math.radians(fov_y / 2.0))

    lon = torch.linspace(-math.pi, math.pi, equi_w, device=device).unsqueeze(0).expand(equi_h, equi_w)
    lat = torch.linspace(math.pi / 2, -math.pi / 2, equi_h, device=device).unsqueeze(1).expand(equi_h, equi_w)

    x_map = torch.cos(lon) * torch.cos(lat)
    y_map = torch.sin(lon) * torch.cos(lat)
    z_map = torch.sin(lat)
    xyz = torch.stack((x_map, y_map, z_map), dim=-1).view(-1, 3).T  # (3, H*W)

    R = _build_rotation_matrices(roll.to(device), pitch.to(device), yaw.to(device), device)
    R_inv = torch.linalg.inv(R)  # (T, 3, 3)

    xyz_rot = torch.matmul(R_inv, xyz).permute(0, 2, 1).view(T, equi_h, equi_w, 3)

    # perspective projection: map 3D point to 2D perspective image coords
    u = xyz_rot[:, :, :, 1] / (xyz_rot[:, :, :, 0] + 1e-8)  # [-len_x, len_x]
    v = xyz_rot[:, :, :, 2] / (xyz_rot[:, :, :, 0] + 1e-8)  # [-len_y, len_y]

    # normalize to [-1, 1] for grid_sample
    grid_x = u / len_x  # [-1, 1]
    grid_y = -v / len_y  # [-1, 1], negated: grid_sample -1=top but world +z=up
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (T, equi_h, equi_w, 2)

    equi_frames = F.grid_sample(
        pers_frames, grid, mode="bilinear", padding_mode="zeros", align_corners=True,
    )

    equi_frames = equi_frames * mask

    return equi_frames, mask


def create_pano_condition(
    pers_video: torch.Tensor,
    fov_x: float,
    roll: torch.Tensor,
    pitch: torch.Tensor,
    yaw: torch.Tensor,
    equi_h: int = 512,
    equi_w: int = 1024,
    noise_strength: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create panoramic conditioning from a perspective video.

    Projects the perspective video onto an equirectangular canvas and creates
    a conditioning video where:
      - Observed regions contain the projected perspective content
      - Unobserved regions are filled with noise

    Args:
        pers_video: (T, C, Hp, Wp) perspective video in [-1, 1] range.
        fov_x: Horizontal FOV in degrees.
        roll, pitch, yaw: Camera angles (radians), shape (T,).
        equi_h, equi_w: Equirectangular output dimensions.
        noise_strength: Noise level for unobserved regions.

    Returns:
        cond_video: (T, C, equi_h, equi_w) conditioned equirectangular video.
        mask: (T, 1, equi_h, equi_w) spatial mask.
    """
    hw_ratio = pers_video.shape[2] / pers_video.shape[3]

    equi_proj, mask = pers2equi_simple(
        pers_video, fov_x=fov_x,
        roll=roll, pitch=pitch, yaw=yaw,
        equi_h=equi_h, equi_w=equi_w, hw_ratio=hw_ratio,
    )

    noise = torch.randn_like(equi_proj) * noise_strength
    cond_video = torch.where(mask.expand_as(equi_proj) > 0.5, equi_proj, noise)

    return cond_video, mask


def equi2pers(
    equi_frames: torch.Tensor,
    fov_x: float,
    roll: torch.Tensor,
    pitch: torch.Tensor,
    yaw: torch.Tensor,
    out_h: int = 480,
    out_w: int = 640,
) -> torch.Tensor:
    """Extract perspective crops from equirectangular frames.

    Inverse of pers2equi: for each pixel in the output perspective image,
    compute the corresponding equirectangular coordinate and sample.

    Args:
        equi_frames: (T, C, H_equi, W_equi) equirectangular frames in any range.
        fov_x: Horizontal field of view in degrees.
        roll, pitch, yaw: Camera angles (radians), shape (T,).
        out_h, out_w: Output perspective image size.

    Returns:
        pers_frames: (T, C, out_h, out_w) perspective crops.
    """
    device = equi_frames.device
    T, C, H_equi, W_equi = equi_frames.shape

    half_fov_x = math.radians(fov_x / 2.0)
    aspect = out_h / out_w
    half_fov_y = math.atan(math.tan(half_fov_x) * aspect)

    # Build normalized perspective image plane coordinates
    u = torch.linspace(-math.tan(half_fov_x), math.tan(half_fov_x), out_w, device=device)
    v = torch.linspace(-math.tan(half_fov_y), math.tan(half_fov_y), out_h, device=device)
    uu, vv = torch.meshgrid(u, v, indexing="xy")  # (out_h, out_w)

    # 3D direction vectors in camera space: (x=forward, y=right, z=down)
    # Use z_down=True convention matching ARGUS
    xyz_cam = torch.stack([
        torch.ones_like(uu),   # x = forward
        uu,                     # y = right
        vv,                     # z = down
    ], dim=-1)  # (out_h, out_w, 3)
    xyz_cam = xyz_cam.view(-1, 3).T  # (3, out_h*out_w)

    # Build rotation matrices and rotate to world coordinates
    R = _build_rotation_matrices(
        roll.float().to(device),
        pitch.float().to(device),
        yaw.float().to(device),
        device,
    )  # (T, 3, 3)

    # Rotate camera rays to world space
    xyz_world = torch.matmul(R, xyz_cam)  # (T, 3, out_h*out_w)
    xyz_world = xyz_world.permute(0, 2, 1).view(T, out_h, out_w, 3)  # (T, out_h, out_w, 3)

    # Convert 3D directions to equirectangular (lon, lat)
    x, y, z = xyz_world[..., 0], xyz_world[..., 1], xyz_world[..., 2]
    lon = torch.atan2(y, x)                          # [-pi, pi]
    lat = torch.asin(z.clamp(-1 + 1e-6, 1 - 1e-6))  # [-pi/2, pi/2]

    # Normalize to [-1, 1] for grid_sample
    grid_x = lon / math.pi           # [-1, 1] maps to [-pi, pi]
    grid_y = -lat / (math.pi / 2)    # [-1, 1] maps to [pi/2, -pi/2] (top=+pi/2, bottom=-pi/2)
    grid = torch.stack([grid_x, grid_y], dim=-1).to(dtype=equi_frames.dtype)  # (T, out_h, out_w, 2)

    pers_frames = F.grid_sample(
        equi_frames, grid, mode="bilinear", padding_mode="border", align_corners=True,
    )

    return pers_frames


def random_camera_params(
    num_frames: int,
    fov_x_range: Tuple[float, float] = (60.0, 120.0),
    device: torch.device = torch.device("cpu"),
    simulate_motion: bool = True,
    fps: float = 16.0,
) -> Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample random camera parameters for training augmentation.

    When simulate_motion=True, generates realistic camera trajectories
    following ARGUS's strategy: oscillation + linear drift + Gaussian noise,
    mimicking natural handheld camera motion.

    Returns:
        fov_x: Horizontal FOV in degrees.
        roll, pitch, yaw: Camera angles (radians), shape (num_frames,).
    """
    fov_x = np.random.uniform(*fov_x_range)
    base_yaw = np.random.uniform(-np.pi, np.pi)

    if not simulate_motion or num_frames <= 1:
        yaw = torch.full((num_frames,), base_yaw, device=device)
        pitch = torch.zeros(num_frames, device=device)
        roll = torch.zeros(num_frames, device=device)
        return fov_x, roll, pitch, yaw

    t = np.linspace(0, num_frames / fps, num_frames)
    freq = 1.5  # walking frequency in Hz

    # Oscillatory component (random amplitude + phase)
    pitch_arr = float(np.random.rand()) * 0.5 * np.sin(2 * np.pi * (freq * t + np.random.rand()))
    yaw_arr = float(np.random.rand()) * 1.0 * np.sin(2 * np.pi * (freq * t + np.random.rand()))
    roll_arr = float(np.random.rand()) * 0.2 * np.sin(2 * np.pi * (freq * t + np.random.rand()))

    # Linear drift (2/3 chance of having drift)
    if np.random.rand() > 1 / 3:
        yaw_arr += np.random.uniform(-15, 15) * t
    if np.random.rand() > 1 / 3:
        pitch_arr += np.random.uniform(-3, 3) * t
    if np.random.rand() > 1 / 3:
        roll_arr += np.random.uniform(-1, 1) * t

    # Gaussian noise
    pitch_arr += np.random.normal(0, 0.15, size=t.shape)
    yaw_arr += np.random.normal(0, 0.15, size=t.shape)
    roll_arr += np.random.normal(0, 0.1, size=t.shape)

    # Convert to radians and add base yaw offset
    pitch_arr = np.radians(pitch_arr)
    yaw_arr = np.radians(yaw_arr) + base_yaw
    roll_arr = np.radians(roll_arr)

    pitch = torch.tensor(pitch_arr, dtype=torch.float32, device=device)
    yaw = torch.tensor(yaw_arr, dtype=torch.float32, device=device)
    roll = torch.tensor(roll_arr, dtype=torch.float32, device=device)

    return fov_x, roll, pitch, yaw
