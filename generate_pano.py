#!/usr/bin/env python
"""Generate panoramic (equirectangular) videos using Cosmos Predict 2.5.

Supports three modes:
  1. text2pano: Generate pano video from text prompt only
  2. video2pano: Generate pano video conditioned on input pano video/image
     (input is resized to equirectangular, first frame(s) used as temporal condition)
  3. pers2pano (Phase 2): Generate pano from a normal perspective video
     Uses spatial inpainting: perspective frames are projected onto an equirectangular
     canvas, and the model fills in the unobserved regions.

Usage:
  # Text-to-Pano
  python generate_pano.py --prompt "A 360 panoramic view of a mountain" --output_dir output/pano_test

  # Video-to-Pano (pano input, temporal conditioning)
  python generate_pano.py --prompt "A museum interior" --input_path /path/to/pano.mp4

  # Pers-to-Pano (normal perspective input, spatial inpainting)
  python generate_pano.py --prompt "A city street" --pers_input /path/to/normal.mp4 --fov_x 90 --yaw 0

  # With equirectangular RoPE
  python generate_pano.py --prompt "..." --equirect_rope --output_dir output/pano_equirect
"""

import argparse
import json
import math
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from cosmos_predict2._src.imaginaire.utils import distributed, log
from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
from cosmos_predict2.config import MODEL_CHECKPOINTS, MODEL_KEYS, DEFAULT_MODEL_KEY, DEFAULT_NEGATIVE_PROMPT


def parse_args():
    parser = argparse.ArgumentParser(description="Generate panoramic video with Cosmos Predict 2.5")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt")
    parser.add_argument("--input_path", type=str, default=None,
                        help="Input pano video/image (video2pano mode, temporal conditioning)")
    parser.add_argument("--output_dir", type=str, default="output/pano_generation", help="Output directory")

    parser.add_argument("--resolution", type=str, default="none",
                        help="Resolution as H,W (e.g. 704,1408 for 2:1 pano). "
                             "'none' auto-selects based on model: 704,1408 for 14B pano, 512,1024 for 2B pano.")
    parser.add_argument("--num_frames", type=int, default=77, help="Number of video frames to generate")
    parser.add_argument("--guidance", type=int, default=7, help="Classifier-free guidance scale (0-7)")
    parser.add_argument("--num_steps", type=int, default=35, help="Number of diffusion steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    parser.add_argument("--equirect_rope", action="store_true",
                        help="Use equirectangular-aware RoPE position encoding")
    parser.add_argument("--num_input_frames", type=int, default=1,
                        help="Number of latent conditional frames (0=text2pano, 1+=video2pano)")

    # Phase 2: Perspective-to-Pano (spatial inpainting)
    parser.add_argument("--pers_input", type=str, default=None,
                        help="Input perspective (normal) video path for pers2pano mode")
    parser.add_argument("--fov_x", type=float, default=90.0,
                        help="Horizontal FOV of the perspective camera (degrees)")
    parser.add_argument("--yaw", type=float, default=0.0,
                        help="Camera yaw angle (radians, 0=front, pi=back)")
    parser.add_argument("--pitch", type=float, default=0.0,
                        help="Camera pitch angle (radians)")
    parser.add_argument("--roll", type=float, default=0.0,
                        help="Camera roll angle (radians)")

    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_KEY.name,
                        choices=list(MODEL_KEYS.keys()), help="Model variant")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Custom checkpoint path")
    parser.add_argument("--offload_text_encoder", action="store_true", help="Offload text encoder to CPU")
    parser.add_argument("--offload_tokenizer", action="store_true", help="Offload tokenizer to CPU")
    parser.add_argument("--offload_diffusion_model", action="store_true", help="Offload DiT to CPU")
    parser.add_argument("--disable_guardrails", action="store_true", default=True, help="Disable safety guardrails")

    # Phase 3: SigLIP2 image cross-attention (requires fine-tuned model with extra_image_context_dim)
    parser.add_argument("--use_clip", action="store_true",
                        help="Enable SigLIP2 image cross-attention conditioning (requires model trained with extra_image_context_dim)")

    # Fine-tuned checkpoint loading
    parser.add_argument("--finetune_checkpoint", type=str, default=None,
                        help="Path to fine-tuned .pt checkpoint (EMA weights with net.* keys). "
                             "Overlays on top of the base model weights.")

    # V3 pipeline: Image → I2V → Panoramic Video
    parser.add_argument("--v3", action="store_true",
                        help="Use v3 pipeline: Image → base I2V → perspective video → panoramic expansion")
    parser.add_argument("--pano_prompt", type=str, default=None,
                        help="Separate prompt for panoramic expansion (Step 5). "
                             "Use to describe only the environment without foreground subjects "
                             "that are already in the perspective video. Defaults to --prompt.")
    parser.add_argument("--i2v_resolution", type=str, default="480,832",
                        help="Resolution for I2V perspective video generation (H,W). "
                             "Default 480,832 matches Cosmos 480p 16:9 for landscape images.")
    parser.add_argument("--extend_rounds", type=int, default=0,
                        help="Number of temporal extension rounds after initial pano generation. "
                             "Each round uses last frame as temporal condition for forward prediction, "
                             "without spatial mask blending, allowing objects to move beyond FOV.")
    parser.add_argument("--extend_prompt", type=str, default=None,
                        help="Prompt for temporal extension rounds. Should describe fixed camera "
                             "and desired motion. Defaults to pano_prompt + static camera instructions. "
                             "Use '|' to specify different prompts per round (e.g. 'round2|round3|round4'). "
                             "If fewer prompts than rounds, the last prompt is reused for remaining rounds.")
    parser.add_argument("--latent_extension", action="store_true", default=False,
                        help="Use latent-space overlap for extension rounds instead of pixel-space. "
                             "Bypasses VAE encode/decode round-trip between rounds by passing "
                             "raw latent frames directly. Reduces quantization drift but requires "
                             "the model to have been trained with conditional_frames_probs > 0.")

    # ── Panoramic inference optimizations ──
    parser.add_argument("--latent_padding_size", type=int, default=0,
                        help="Circular latent padding columns for seamless left-right boundary "
                             "during VAE decode. Each unit = 8 pixels. Recommended: 2 (16px/side).")
    parser.add_argument("--rotated_denoise", action="store_true", default=False,
                        help="Random horizontal roll at each denoising step to reduce "
                             "position-anchored artifacts in panoramic generation.")
    parser.add_argument("--erp_warp_noise", action="store_true", default=False,
                        help="Apply ERP latitude-aware warping to initial noise: compresses "
                             "horizontal extent near poles to match equirectangular area distortion.")
    parser.add_argument("--cubemap_viz", action="store_true", default=False,
                        help="Save cubemap (6-face) visualization alongside the ERP output.")
    parser.add_argument("--cubemap_face_size", type=int, default=512,
                        help="Resolution of each cubemap face (square). Default 512.")

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Panoramic inference optimization utilities
# ═══════════════════════════════════════════════════════════════════════════════

def erp_warp_latent(latents: torch.Tensor) -> torch.Tensor:
    """Warp latent noise to compensate ERP area distortion.

    Near poles the equirectangular projection over-represents area, so uniform
    Gaussian noise is not uniform on the sphere.  This warps horizontal
    coordinates by sin(colatitude) -- compressing near poles, preserving the
    equator -- and applies a variance-correction step so the output remains
    approximately unit-Gaussian.

    Reference: PanoWan (_erp_warp).

    Args:
        latents: (B, C, T, H, W) tensor of i.i.d. Gaussian noise.
    Returns:
        Warped tensor with the same shape and approximately unit variance.
    """
    B, C, T, H, W = latents.shape
    flat = latents.permute(0, 2, 1, 3, 4).reshape(B * T * C, 1, H, W)

    y = torch.linspace(-1, 1, 2 * H + 1, device=latents.device,
                       dtype=latents.dtype)[1:-1:2, None]
    x = torch.linspace(-1, 1, 2 * W + 1, device=latents.device,
                       dtype=latents.dtype)[None, 1:-1:2]
    x = x * torch.sin((y * 0.5 + 0.5) * math.pi)

    grid = (torch.stack((x.expand(H, W), y.expand(H, W)), dim=-1)
            .unsqueeze(0).expand(flat.shape[0] * 2, -1, -1, -1))

    warped, warped_x2 = F.grid_sample(
        torch.cat([flat, flat ** 2]),
        grid, mode="bilinear", padding_mode="border", align_corners=False,
    ).chunk(2)

    warped = torch.sign(warped) * warped_x2.clamp(min=1e-8).sqrt()
    return warped.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)


def erp_to_cubemap(erp_video: torch.Tensor, face_size: int = 512) -> dict:
    """Convert an ERP video to 6 cubemap face videos.

    Args:
        erp_video: (C, T, H, W) tensor in [0, 1].
        face_size: Resolution of each square face.
    Returns:
        Dict mapping face name to (C, T, face_size, face_size) tensor.
    """
    C, T, H, W = erp_video.shape
    device = erp_video.device

    face_names = ["front", "right", "back", "left", "top", "bottom"]
    rotations = {
        "front":  (0, 0),
        "right":  (0, math.pi / 2),
        "back":   (0, math.pi),
        "left":   (0, -math.pi / 2),
        "top":    (-math.pi / 2, 0),
        "bottom": (math.pi / 2, 0),
    }

    fs = face_size
    u = torch.linspace(-1, 1, fs, device=device)
    v = torch.linspace(-1, 1, fs, device=device)
    grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")

    results = {}
    for name in face_names:
        pitch, yaw = rotations[name]

        x = grid_u
        y = -grid_v
        z = torch.ones_like(grid_u)

        cos_p, sin_p = math.cos(pitch), math.sin(pitch)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)

        x1 = cos_y * x + sin_y * z
        z1 = -sin_y * x + cos_y * z
        y1 = y

        x2 = x1
        y2 = cos_p * y1 - sin_p * z1
        z2 = sin_p * y1 + cos_p * z1

        lon = torch.atan2(x2, z2)
        lat = torch.asin(y2.clamp(-1, 1) / (x2**2 + y2**2 + z2**2).sqrt().clamp(min=1e-6))

        sample_x = lon / math.pi
        sample_y = -lat / (math.pi / 2)

        sample_grid = torch.stack([sample_x, sample_y], dim=-1).unsqueeze(0)
        sample_grid = sample_grid.expand(T, -1, -1, -1)

        frames = erp_video.permute(1, 0, 2, 3)  # (T, C, H, W)
        face_frames = F.grid_sample(
            frames, sample_grid, mode="bilinear",
            padding_mode="border", align_corners=False,
        )
        results[name] = face_frames.permute(1, 0, 2, 3)  # (C, T, fs, fs)

    return results


def save_cubemap_visualization(erp_video: torch.Tensor, output_path: str,
                               face_size: int = 512, fps: int = 16):
    """Render and save cubemap faces from ERP video.

    Saves individual face videos and a combined 3x2 cross layout video.
    """
    from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video

    faces = erp_to_cubemap(erp_video, face_size)

    for name, face_vid in faces.items():
        face_path = f"{output_path}_cube_{name}"
        save_img_or_video(face_vid, face_path, fps=fps)

    C, T, fs, _ = faces["front"].shape
    cross = torch.zeros(C, T, fs * 3, fs * 4, device=erp_video.device)
    layout = {
        "top":    (0, 1),
        "left":   (1, 0),
        "front":  (1, 1),
        "right":  (1, 2),
        "back":   (1, 3),
        "bottom": (2, 1),
    }
    for name, (row, col) in layout.items():
        r0, c0 = row * fs, col * fs
        cross[:, :, r0:r0+fs, c0:c0+fs] = faces[name]

    cross_path = f"{output_path}_cube_cross"
    save_img_or_video(cross, cross_path, fps=fps)
    log.info(f"  Cubemap cross layout: {cross_path}.mp4 ({fs*3}x{fs*4})")


def read_perspective_video(
    video_path: str,
    num_frames: int,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """Read perspective video and return as tensor (T, C, H, W) in [-1, 1]."""
    from decord import VideoReader, cpu

    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)

    if total_frames >= num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        indices = list(range(total_frames))
        indices += [total_frames - 1] * (num_frames - total_frames)

    frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C) uint8
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()  # (T, C, H, W)
    frames = frames / 127.5 - 1.0  # [-1, 1]
    return frames.to(device)


def prepare_pers2pano_condition(
    pipe: Video2WorldInference,
    pers_video: torch.Tensor,
    fov_x: float,
    roll: float,
    pitch: float,
    yaw: float,
    equi_h: int,
    equi_w: int,
) -> dict:
    """Prepare spatial inpainting condition from a perspective video.

    Projects perspective frames onto equirectangular canvas, creates a latent-space
    mask, and returns guided_image / guided_mask for the x0_fn replacement trick.

    Args:
        pipe: Loaded Video2WorldInference pipeline.
        pers_video: (T, C, H, W) perspective video in [-1, 1].
        fov_x: Horizontal FOV in degrees.
        roll, pitch, yaw: Camera angles in radians (constant across frames).
        equi_h, equi_w: Equirectangular resolution.

    Returns:
        dict with "guided_image" and "guided_mask" tensors in latent space.
    """
    from cosmos_predict2._src.predict2.utils.pano_conditioning import (
        pers2equi_simple,
        generate_equirect_mask,
    )

    T_in, C, Hp, Wp = pers_video.shape
    device = pers_video.device
    hw_ratio = Hp / Wp

    roll_t = torch.full((T_in,), roll, device=device)
    pitch_t = torch.full((T_in,), pitch, device=device)
    yaw_t = torch.full((T_in,), yaw, device=device)

    # Project perspective → equirectangular
    equi_proj, pixel_mask = pers2equi_simple(
        pers_video, fov_x=fov_x,
        roll=roll_t, pitch=pitch_t, yaw=yaw_t,
        equi_h=equi_h, equi_w=equi_w, hw_ratio=hw_ratio,
    )
    log.info(f"  Projected perspective ({Hp}x{Wp}) → equirectangular ({equi_h}x{equi_w})")
    log.info(f"  Mask coverage: {pixel_mask.mean().item():.1%} of pixels")

    # Pad/trim to model's required frame count
    model_required_frames = pipe.model.tokenizer.get_pixel_num_frames(pipe.model.config.state_t)
    if T_in < model_required_frames:
        pad = model_required_frames - T_in
        equi_proj = torch.cat([equi_proj, equi_proj[-1:].repeat(pad, 1, 1, 1)], dim=0)
        pixel_mask = torch.cat([pixel_mask, pixel_mask[-1:].repeat(pad, 1, 1, 1)], dim=0)
    elif T_in > model_required_frames:
        equi_proj = equi_proj[:model_required_frames]
        pixel_mask = pixel_mask[:model_required_frames]

    equi_proj_BCTHW = equi_proj.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)
    pixel_mask_11THW = pixel_mask[:, 0:1].permute(1, 0, 2, 3).unsqueeze(0)  # (1, 1, T, H, W)

    # VAE encode → latent space
    log.info("  Encoding projected equirectangular to latent space...")
    tokenizer = pipe.model.tokenizer
    if hasattr(tokenizer, 'encoder') and tokenizer.encoder is not None:
        was_on_cpu = next(tokenizer.encoder.parameters()).device.type == "cpu"
        if was_on_cpu:
            tokenizer.encoder = tokenizer.encoder.to(device)

    equi_uint8 = ((equi_proj_BCTHW + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    with torch.no_grad():
        latent = tokenizer.encode(equi_uint8.to(device))
    guided_image = latent.cpu()  # (1, C_lat, T_lat, H_lat, W_lat) — keep on CPU until needed

    # Offload VAE encoder to free GPU memory for the diffusion model
    if hasattr(tokenizer, 'encoder') and tokenizer.encoder is not None:
        tokenizer.encoder = tokenizer.encoder.to("cpu")
        torch.cuda.empty_cache()
        log.info("  Offloaded VAE encoder to CPU")

    # Downsample pixel mask to exact latent resolution via interpolate
    _, _, T_lat, H_lat, W_lat = guided_image.shape
    guided_mask = F.interpolate(
        pixel_mask_11THW.float(),
        size=(T_lat, H_lat, W_lat),
        mode="nearest",
    )
    guided_mask = (guided_mask > 0.5).float()
    log.info(f"  Latent shape: {guided_image.shape}, mask shape: {guided_mask.shape}")
    log.info(f"  Latent mask coverage: {guided_mask.mean().item():.1%}")

    return {
        "guided_image": guided_image.to(dtype=torch.bfloat16),  # on CPU
        "guided_mask": guided_mask.to(dtype=torch.bfloat16),    # on CPU
        "equi_proj_vis": equi_proj_BCTHW.cpu(),
        "pixel_mask_vis": pixel_mask_11THW.cpu(),
    }

SIGLIP2_MODEL_ID = "google/siglip2-so400m-patch14-384"


def encode_temporal_clip(video_frames_01: torch.Tensor) -> torch.Tensor:
    """Encode T perspective video frames with SigLIP2 So400m/14.

    Args:
        video_frames_01: (T, C, H, W) in [0, 1] range.

    Returns:
        (1, T*729, 1152) flattened temporal SigLIP2 features matching training format.
    """
    from transformers import SiglipVisionModel
    from torchvision.transforms.functional import normalize

    T = video_frames_01.shape[0]
    log.info(f"Loading SigLIP2 for temporal encoding ({T} frames)...")
    clip_model = SiglipVisionModel.from_pretrained(SIGLIP2_MODEL_ID)
    clip_model = clip_model.to(device="cuda", dtype=torch.bfloat16).eval()
    image_size = clip_model.config.image_size  # 384
    hidden_size = clip_model.config.hidden_size  # 1152

    frames = F.interpolate(video_frames_01, size=(image_size, image_size),
                           mode="bicubic", align_corners=False)
    frames = normalize(frames, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    frames = frames.to(dtype=torch.bfloat16, device="cuda")

    all_features = []
    batch_size = 4  # smaller batch due to larger 384px input
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, T, batch_size):
            batch = frames[i:i + batch_size]
            feat = clip_model(pixel_values=batch, output_hidden_states=False).last_hidden_state
            all_features.append(feat)

    temporal_clip = torch.cat(all_features, dim=0)  # (T, 729, 1152)
    tok = temporal_clip.shape[1]
    temporal_clip = temporal_clip.reshape(1, T * tok, hidden_size)

    del clip_model, frames
    torch.cuda.empty_cache()
    log.info(f"Temporal SigLIP2 features: {temporal_clip.shape}")
    return temporal_clip.to(dtype=torch.bfloat16)


def encode_clip_image(input_path: str) -> torch.Tensor:
    """Encode the first frame of an image/video using SigLIP2 So400m/14.

    Returns (1, 729, 1152) patch-level features suitable for image cross-attention.
    The model is loaded, used, and immediately freed to save GPU memory.
    """
    from PIL import Image
    from torchvision import transforms as T
    from torchvision.transforms.functional import normalize

    ext = os.path.splitext(input_path)[1].lower()
    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        img = Image.open(input_path).convert("RGB")
        frame = T.ToTensor()(img).unsqueeze(0)  # (1, 3, H, W) in [0, 1]
    elif ext in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        from decord import VideoReader, cpu
        vr = VideoReader(input_path, ctx=cpu(0))
        first_frame = vr[0].asnumpy()  # (H, W, C) uint8
        frame = torch.from_numpy(first_frame).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    else:
        raise ValueError(f"Unsupported file type for encoding: {ext}")

    log.info(f"SigLIP2 input image: {input_path} ({frame.shape[-2]}x{frame.shape[-1]})")

    from transformers import SiglipVisionModel
    log.info(f"Loading SigLIP2 from {SIGLIP2_MODEL_ID}...")
    clip_model = SiglipVisionModel.from_pretrained(SIGLIP2_MODEL_ID)
    clip_model = clip_model.to(device="cuda", dtype=torch.bfloat16).eval()
    image_size = clip_model.config.image_size  # 384

    frame = F.interpolate(frame, size=(image_size, image_size), mode="bicubic", align_corners=False)
    frame = normalize(frame, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    frame = frame.to(dtype=torch.bfloat16, device="cuda")

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        clip_emb = clip_model(pixel_values=frame, output_hidden_states=False).last_hidden_state

    clip_emb = clip_emb.to(dtype=torch.bfloat16)
    del clip_model
    torch.cuda.empty_cache()
    log.info(f"SigLIP2 image embedding shape: {clip_emb.shape}")
    return clip_emb


def read_perspective_input(
    input_path: str,
    num_frames: int,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """Read a perspective image or video and return (T, C, H, W) in [-1, 1].

    For images, replicates the frame to fill num_frames.
    For videos, delegates to read_perspective_video.
    """
    ext = os.path.splitext(input_path)[1].lower()
    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        from PIL import Image
        from torchvision import transforms as T

        img = Image.open(input_path).convert("RGB")
        frame = T.ToTensor()(img)  # (C, H, W) in [0, 1]
        frame = frame * 2.0 - 1.0  # [-1, 1]
        frames = frame.unsqueeze(0).expand(num_frames, -1, -1, -1)  # (T, C, H, W)
        return frames.to(device)
    else:
        return read_perspective_video(input_path, num_frames, device)


def prepare_reference_latent(
    pipe: Video2WorldInference,
    input_path: str,
    equi_h: int,
    equi_w: int,
    fov_x: float = 90.0,
    noise_strength: float = 0.5,
    yaw_rad: float = 0.0,
) -> tuple:
    """Create ARGUS-style reference latent from a perspective image.

    1. Read perspective image, replicate to T frames
    2. Project to equirectangular via pers2equi_simple
    3. Fill unobserved regions with scaled noise (matching training)
    4. VAE encode → reference_latent (1, 16, T_lat, H_lat, W_lat)

    Returns:
        (reference_latent, equi_pixel_video) where equi_pixel_video is
        (1, 3, T, H, W) uint8 for use as temporal conditioning input.
    """
    from cosmos_predict2._src.predict2.utils.pano_conditioning import (
        pers2equi_simple,
        generate_equirect_mask,
    )

    model_required_frames = pipe.model.tokenizer.get_pixel_num_frames(pipe.model.config.state_t)
    device = torch.device("cuda")

    pers_frames = read_perspective_input(input_path, model_required_frames, device)
    T, C, Hp, Wp = pers_frames.shape
    hw_ratio = Hp / Wp
    log.info(f"  Perspective input: {Hp}x{Wp}, {T} frames, FOV={fov_x}°")

    roll = torch.zeros(T, device=device)
    pitch = torch.zeros(T, device=device)
    yaw = torch.full((T,), yaw_rad, device=device)

    equi_proj, pixel_mask = pers2equi_simple(
        pers_frames, fov_x=fov_x,
        roll=roll, pitch=pitch, yaw=yaw,
        equi_h=equi_h, equi_w=equi_w, hw_ratio=hw_ratio,
    )
    log.info(f"  Projected to {equi_h}x{equi_w}, coverage: {pixel_mask.mean().item():.1%}")

    noise = torch.randn_like(equi_proj) * noise_strength
    masked_equi = torch.where(pixel_mask.expand_as(equi_proj) > 0.5, equi_proj, noise)

    # Prepare clean equi projection for blended diffusion (observed=image, unobserved=0)
    equi_clean = torch.where(pixel_mask.expand_as(equi_proj) > 0.5, equi_proj,
                             torch.zeros_like(equi_proj))
    equi_clean_BCTHW = equi_clean.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)
    spatial_mask_BCTHW = pixel_mask[:, :1, :, :].permute(1, 0, 2, 3).unsqueeze(0)  # (1, 1, T, H, W)
    log.info(f"  Spatial mask for blended diffusion: {spatial_mask_BCTHW.shape}")

    masked_equi_BCTHW = masked_equi.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)

    tokenizer = pipe.model.tokenizer
    if hasattr(tokenizer, 'encoder') and tokenizer.encoder is not None:
        was_on_cpu = next(tokenizer.encoder.parameters()).device.type == "cpu"
        if was_on_cpu:
            tokenizer.encoder = tokenizer.encoder.to(device)

    sigma_data = pipe.model.sigma_data
    with torch.no_grad():
        reference_latent = tokenizer.encode(masked_equi_BCTHW.to(device)) * sigma_data

    log.info(f"  Reference latent shape: {reference_latent.shape}")

    # VAE-encode the CLEAN equi projection for blended diffusion
    with torch.no_grad():
        guided_image_latent = tokenizer.encode(equi_clean_BCTHW.to(device)) * sigma_data

    # Downsample spatial mask to latent space
    _, _, T_lat, H_lat, W_lat = reference_latent.shape
    guided_mask_latent = F.avg_pool3d(
        spatial_mask_BCTHW.float().to(device),
        kernel_size=(max(1, T // T_lat), max(1, equi_h // H_lat), max(1, equi_w // W_lat)),
        stride=(max(1, T // T_lat), max(1, equi_h // H_lat), max(1, equi_w // W_lat)),
    )
    guided_mask_latent = (guided_mask_latent > 0.5).float()
    if guided_mask_latent.shape[2:] != (T_lat, H_lat, W_lat):
        guided_mask_latent = F.interpolate(guided_mask_latent, size=(T_lat, H_lat, W_lat), mode="nearest")
    log.info(f"  Guided image latent: {guided_image_latent.shape}, mask: {guided_mask_latent.shape}")

    if hasattr(tokenizer, 'encoder') and tokenizer.encoder is not None:
        tokenizer.encoder = tokenizer.encoder.to("cpu")
        torch.cuda.empty_cache()
        log.info("  Offloaded VAE encoder to CPU")

    del pers_frames, equi_proj, pixel_mask, masked_equi, masked_equi_BCTHW, noise
    del equi_clean, equi_clean_BCTHW, spatial_mask_BCTHW
    torch.cuda.empty_cache()

    return (
        reference_latent.to(dtype=torch.bfloat16),
        guided_image_latent.to(dtype=torch.bfloat16),
        guided_mask_latent.to(dtype=torch.bfloat16),
    )


def run_v3_pipeline(args):
    """V3 pipeline: Image -> I2V -> Panoramic Video.

    Step 1: Generate perspective video from input image using base Cosmos model
    Step 2: Load fine-tuned panoramic weights + switch to equirectangular RoPE
    Step 3: Encode temporal CLIP features from all perspective frames
    Step 4: Project perspective video to equirectangular + spatial mask + VAE encode
    Step 5: Generate panoramic video with temporal CLIP + blended diffusion
    """
    os.makedirs(args.output_dir, exist_ok=True)

    model_key = MODEL_KEYS[args.model]
    checkpoint = MODEL_CHECKPOINTS[model_key]
    ckpt_path = args.checkpoint_path or checkpoint.path
    experiment = checkpoint.experiment
    config_file = "cosmos_predict2/_src/predict2/configs/video2world/config.py"

    experiment_opts = []
    if args.use_clip and args.finetune_checkpoint:
        experiment_opts.append("++model.config.net.extra_image_context_dim=1152")
        experiment_opts.append("++model.config.net.reference_channels=16")
    # Match training state_t for correct latent temporal dim
    target_T = args.num_frames
    target_T_lat = (target_T - 1) // 4 + 1
    experiment_opts.append(f"++model.config.state_t={target_T_lat}")

    h_pano, w_pano = [int(x) for x in args.resolution.split(",")]

    log.info("=" * 60)
    log.info("V3 Pipeline: Image -> I2V -> Panoramic Video")
    log.info(f"Panoramic resolution: {h_pano}x{w_pano}")
    log.info(f"I2V resolution: {args.i2v_resolution}")
    log.info(f"Input: {args.input_path or args.pers_input}")
    log.info("=" * 60)

    t0_total = time.time()

    # ====== Step 1: Generate or load perspective video ======
    if args.pers_input:
        log.info("Step 1: Reading perspective video from file (skipping I2V)...")
        pers_frames_m1p1 = read_perspective_video(args.pers_input, target_T).to("cuda")
        pers_frames_01 = (pers_frames_m1p1 + 1.0) / 2.0
        log.info(f"Step 1: Loaded perspective video: {pers_frames_m1p1.shape}")
    else:
        log.info("Step 1: Generating perspective video from image (base model I2V)...")
        log.info("Step 1: Loading CLEAN base model (no pano architecture mods)...")
        i2v_h, i2v_w = [int(x) for x in args.i2v_resolution.split(",")]

        base_pipe = Video2WorldInference(
            experiment_name=experiment,
            ckpt_path=ckpt_path,
            s3_credential_path="",
            context_parallel_size=1,
            config_file=config_file,
            experiment_opts=[],
            offload_diffusion_model=args.offload_diffusion_model,
            offload_text_encoder=args.offload_text_encoder,
            offload_tokenizer=args.offload_tokenizer,
        )

        t0 = time.time()
        pers_video_raw = base_pipe.generate_vid2world(
            prompt=args.prompt,
            input_path=args.input_path,
            guidance=args.guidance,
            num_video_frames=args.num_frames,
            num_latent_conditional_frames=1,
            resolution=f"{i2v_h},{i2v_w}",
            seed=args.seed,
            negative_prompt=args.negative_prompt,
            num_steps=args.num_steps,
        )
        log.info(f"Step 1: I2V took {time.time() - t0:.1f}s")

        pers_01_raw = (1.0 + pers_video_raw[0]) / 2.0  # (C, T_raw, H, W)

        # Take first target_T frames (preserve temporal continuity, not subsample)
        T_raw = pers_01_raw.shape[1]
        if T_raw > target_T:
            indices = torch.arange(target_T)
            log.info(f"Step 1: Taking first {target_T} of {T_raw} frames (preserving temporal continuity)")
        else:
            indices = torch.arange(T_raw)
        pers_01 = pers_01_raw[:, indices]  # (C, target_T, H, W)
        pers_m1p1 = pers_video_raw[0][:, indices]

        pers_save = os.path.join(args.output_dir, "v3_step1_perspective")
        save_img_or_video(pers_01, pers_save, fps=16)
        log.info(f"Step 1: Saved perspective video ({pers_01.shape[1]} frames) -> {pers_save}.mp4")

        pers_frames_01 = pers_01.permute(1, 0, 2, 3)       # (T, C, H, W) [0,1]
        pers_frames_m1p1 = pers_m1p1.permute(1, 0, 2, 3)   # (T, C, H, W) [-1,1]
        del pers_video_raw, pers_01_raw, pers_01, pers_m1p1, base_pipe
        import gc; gc.collect()
        torch.cuda.empty_cache()
        log.info("Step 1: Released base model memory")

    # ====== Step 2: Load pano model with fine-tuned weights + equirectangular RoPE ======
    log.info("Step 2: Loading panoramic model (extra_image_context_dim + reference_channels)...")
    pipe = Video2WorldInference(
        experiment_name=experiment,
        ckpt_path=ckpt_path,
        s3_credential_path="",
        context_parallel_size=1,
        config_file=config_file,
        experiment_opts=experiment_opts,
        offload_diffusion_model=args.offload_diffusion_model,
        offload_text_encoder=args.offload_text_encoder,
        offload_tokenizer=args.offload_tokenizer,
    )

    if args.finetune_checkpoint:
        log.info("Step 2: Loading fine-tuned panoramic weights...")
        ft_sd = torch.load(args.finetune_checkpoint, map_location="cpu", weights_only=False)
        net_sd = {k.replace("net.", ""): v for k, v in ft_sd.items() if k.startswith("net.")}
        load_info = pipe.model.net.load_state_dict(net_sd, strict=False)
        log.info(f"Step 2: missing={len(load_info.missing_keys)}, unexpected={len(load_info.unexpected_keys)}")
        del ft_sd, net_sd
        torch.cuda.empty_cache()

    net = pipe.model.net
    if args.equirect_rope:
        if hasattr(net, 'pos_embedder'):
            net.pos_embedder.grid_type = "equirectangular"
        elif hasattr(net, 'rope_position_embedding'):
            net.rope_position_embedding.grid_type = "equirectangular"
            net.rope_position_embedding._is_initialized = False
        log.info("Step 2: Switched to equirectangular RoPE")

    # ── Pano inference optimizations (V3 path) ────────────────────────────────
    if args.latent_padding_size > 0:
        _v3_orig_decode = pipe.model.decode
        _v3_pad = args.latent_padding_size

        @torch.no_grad()
        def _v3_circular_decode(latent):
            left_p = latent[..., -_v3_pad:]
            right_p = latent[..., :_v3_pad]
            padded = torch.cat([left_p, latent, right_p], dim=-1)
            decoded = _v3_orig_decode(padded)
            px_p = _v3_pad * 8
            return decoded[..., px_p:-px_p]

        pipe.model.decode = _v3_circular_decode
        log.info(f"[Pano Opt V3] Circular latent padding: {_v3_pad} cols/side")

    # ====== Step 3: Temporal CLIP encoding ======
    log.info("Step 3: Encoding temporal CLIP from perspective video...")
    temporal_clip_emb = encode_temporal_clip(pers_frames_01.to("cuda"))

    # ====== Step 4: Project perspective -> equirectangular + mask + VAE encode ======
    log.info("Step 4: Projecting perspective video to equirectangular...")
    from cosmos_predict2._src.predict2.utils.pano_conditioning import pers2equi_simple

    T = pers_frames_m1p1.shape[0]
    device = torch.device("cuda")
    Hp, Wp = pers_frames_m1p1.shape[2], pers_frames_m1p1.shape[3]
    hw_ratio = Hp / Wp

    roll_t = torch.zeros(T, device=device)
    pitch_t = torch.zeros(T, device=device)
    yaw_t = torch.full((T,), args.yaw, device=device)

    equi_proj, pixel_mask = pers2equi_simple(
        pers_frames_m1p1.to(device), fov_x=args.fov_x,
        roll=roll_t, pitch=pitch_t, yaw=yaw_t,
        equi_h=h_pano, equi_w=w_pano, hw_ratio=hw_ratio,
    )
    coverage = pixel_mask.mean().item()
    log.info(f"Step 4: {Hp}x{Wp} -> {h_pano}x{w_pano}, coverage: {coverage:.1%}")

    equi_vis = (equi_proj.permute(1, 0, 2, 3) + 1.0) / 2.0
    equi_save = os.path.join(args.output_dir, "v3_step4_equi_projected")
    save_img_or_video(equi_vis, equi_save, fps=16)
    log.info(f"Step 4: Saved projected equirectangular -> {equi_save}.mp4")
    del equi_vis

    equi_BCTHW = equi_proj.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W) [-1,1]
    mask_11THW = pixel_mask[:, :1].permute(1, 0, 2, 3).unsqueeze(0)  # (1, 1, T, H, W)

    # Create masked video for reference_latent: FOV=video, non-FOV=noise (matching training)
    noise_fill = torch.randn_like(equi_BCTHW) * 0.5  # noise_conditioning_strength=0.5
    masked_equi_BCTHW = torch.where(mask_11THW.expand_as(equi_BCTHW) > 0.5,
                                     equi_BCTHW, noise_fill)

    tokenizer = pipe.model.tokenizer
    if hasattr(tokenizer, 'encoder') and tokenizer.encoder is not None:
        if next(tokenizer.encoder.parameters()).device.type == "cpu":
            tokenizer.encoder = tokenizer.encoder.to(device)

    # VAE encode: input must be float [-1, 1] (tokenizer internally does (x+1)/2)
    # NO sigma_data scaling (training uses raw tokenizer.encode output)
    with torch.no_grad():
        guided_image = tokenizer.encode(equi_BCTHW.float().to(device))
        reference_latent = tokenizer.encode(masked_equi_BCTHW.float().to(device))

    _, _, T_lat, H_lat, W_lat = guided_image.shape
    guided_mask = F.interpolate(mask_11THW.float().to(device),
                                size=(T_lat, H_lat, W_lat), mode="nearest")
    guided_mask = (guided_mask > 0.5).float()
    log.info(f"Step 4: guided_image: {guided_image.shape}, reference_latent: {reference_latent.shape}")
    log.info(f"Step 4: Latent mask coverage: {guided_mask.mean().item():.1%}")

    if hasattr(tokenizer, 'encoder') and tokenizer.encoder is not None:
        tokenizer.encoder = tokenizer.encoder.to("cpu")
        torch.cuda.empty_cache()

    del pers_frames_01, pers_frames_m1p1, equi_proj, pixel_mask
    del equi_BCTHW, mask_11THW, noise_fill, masked_equi_BCTHW
    torch.cuda.empty_cache()

    # ====== Step 5: Panoramic generation with temporal CLIP + blended diffusion ======
    log.info("Step 5: Generating panoramic video...")

    _original_forward = net.forward
    _clip_emb = temporal_clip_emb.to(dtype=torch.bfloat16)
    _ref_lat = reference_latent.to(dtype=torch.bfloat16)
    _spatial_mask = guided_mask.to(dtype=torch.bfloat16)

    # CFG calls net.forward twice per step: conditional then unconditional.
    # CLIP must only be in the conditional branch so CFG amplifies the image signal.
    # Spatial mask and reference_latent go in both (structural, not semantic).
    _cfg_call_counter = [0]

    def _patched_forward(*a, **kw):
        dev = a[0].device if a else "cuda"
        kw["reference_latent_B_C_T_H_W"] = _ref_lat.to(device=dev, dtype=torch.bfloat16)
        kw["condition_video_input_mask_B_C_T_H_W"] = _spatial_mask.to(device=dev, dtype=torch.bfloat16)

        is_cond = (_cfg_call_counter[0] % 2 == 0)
        _cfg_call_counter[0] += 1
        if is_cond:
            kw["img_context_emb"] = _clip_emb.to(device=dev, dtype=torch.bfloat16)
        else:
            kw["img_context_emb"] = None

        a = tuple(
            x.to(dtype=torch.bfloat16) if isinstance(x, torch.Tensor) and x.is_floating_point() else x
            for x in a
        )
        for k, v in kw.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                kw[k] = v.to(dtype=torch.bfloat16)
        return _original_forward(*a, **kw)

    net.forward = _patched_forward

    model = pipe.model
    original_generate = model.generate_samples_from_batch
    _g_img = guided_image.to(dtype=torch.bfloat16)
    _g_mask = guided_mask.to(dtype=torch.bfloat16)

    @torch.no_grad()
    def patched_generate_v3(data_batch, guidance=1.5, seed=1, state_shape=None,
                            n_sample=None, is_negative_prompt=False, num_steps=35,
                            shift=5.0, **kwargs):
        import tqdm as tqdm_mod
        from cosmos_predict2._src.imaginaire.utils import misc

        model._normalize_video_databatch_inplace(data_batch)
        model._augment_image_dim_inplace(data_batch)
        is_image_batch = model.is_image_batch(data_batch)
        input_key = model.input_image_key if is_image_batch else model.input_data_key
        if n_sample is None:
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            _T, _H, _W = data_batch[input_key].shape[-3:]
            state_shape = [
                model.config.state_ch,
                model.tokenizer.get_latent_num_frames(_T),
                _H // model.tokenizer.spatial_compression_factor,
                _W // model.tokenizer.spatial_compression_factor,
            ]

        noise = misc.arch_invariant_rand(
            (n_sample,) + tuple(state_shape),
            torch.float32, model.tensor_kwargs["device"], seed,
        )
        seed_g = torch.Generator(device=model.tensor_kwargs["device"])
        seed_g.manual_seed(seed)

        model.sample_scheduler.set_timesteps(
            num_steps, device=model.tensor_kwargs["device"], shift=shift,
            use_kerras_sigma=getattr(model.config, 'use_kerras_sigma_at_inference', False),
        )
        timesteps = model.sample_scheduler.timesteps

        velocity_fn = model.get_velocity_fn_from_batch(
            data_batch, guidance, is_negative_prompt=is_negative_prompt,
        )
        latents = noise

        g_img = _g_img.to(device=latents.device, dtype=latents.dtype)
        g_mask = _g_mask.to(device=latents.device, dtype=latents.dtype)
        if g_mask.shape[2:] != latents.shape[2:]:
            g_mask = F.interpolate(g_mask.float(), size=latents.shape[2:],
                                   mode="nearest").to(dtype=latents.dtype)
        if g_img.shape[2:] != latents.shape[2:]:
            g_img = F.interpolate(g_img.float(), size=latents.shape[2:],
                                  mode="nearest").to(dtype=latents.dtype)

        for step_idx, t in enumerate(tqdm_mod.tqdm(timesteps, desc="V3 panoramic")):
            # Match training: blend CLEAN guide BEFORE forward (denoise line 105-107)
            latents_blended = g_mask * g_img + (1.0 - g_mask) * latents

            _cfg_call_counter[0] = 0  # reset so cond=even, uncond=odd
            timestep = torch.stack([t])
            velocity_pred = velocity_fn(noise, latents_blended, timestep.unsqueeze(0))
            temp_x0 = model.sample_scheduler.step(
                velocity_pred.unsqueeze(0), t, latents_blended[0].unsqueeze(0),
                return_dict=False, generator=seed_g,
            )[0]
            latents = temp_x0.squeeze(0)

        # Final blend to ensure FOV region matches guided data
        latents = g_mask * g_img + (1.0 - g_mask) * latents
        return latents

    model.generate_samples_from_batch = patched_generate_v3

    pano_prompt = args.pano_prompt if args.pano_prompt else args.prompt
    log.info(f"Step 5: Using pano prompt: {pano_prompt[:100]}...")

    t0 = time.time()
    video = pipe.generate_vid2world(
        prompt=pano_prompt,
        input_path=None,
        guidance=args.guidance,
        num_video_frames=args.num_frames,
        num_latent_conditional_frames=0,
        resolution=args.resolution,
        seed=args.seed,
        negative_prompt=args.negative_prompt,
        num_steps=args.num_steps,
    )
    gen_time = time.time() - t0

    model.generate_samples_from_batch = original_generate
    net.forward = _original_forward

    video_out = (1.0 + video[0]) / 2.0
    rope_tag = "equirect" if args.equirect_rope else "linear"
    filename = f"pano_v3_{h_pano}x{w_pano}_{rope_tag}_s{args.seed}"
    output_path = os.path.join(args.output_dir, filename)
    save_img_or_video(video_out, output_path, fps=16)

    if args.cubemap_viz:
        log.info("Rendering cubemap visualization (V3)...")
        save_cubemap_visualization(video_out, output_path,
                                   face_size=args.cubemap_face_size, fps=16)

    info = vars(args).copy()
    info["mode"] = "v3"
    info["total_time_s"] = time.time() - t0_total
    info["pano_gen_time_s"] = gen_time
    with open(f"{output_path}.json", "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    log.success(f"V3 pipeline complete! Output: {output_path}.mp4 (total: {time.time()-t0_total:.1f}s)")

    # ====== Temporal Extension Rounds ======
    if args.extend_rounds > 0:
        from PIL import Image
        import torchvision.transforms.functional as TF

        extend_prompt_raw = args.extend_prompt if args.extend_prompt else pano_prompt
        if '|' in extend_prompt_raw:
            extend_prompts = [p.strip() for p in extend_prompt_raw.split('|')]
        else:
            extend_prompts = [extend_prompt_raw]
        extend_neg = (
            args.negative_prompt
            + ", camera shake, camera movement, camera rotation, handheld, shaky, unstable, panning, tilting, zooming"
        )

        log.info(f"{'='*60}")
        log.info(f"Starting temporal extension: {args.extend_rounds} round(s), {len(extend_prompts)} prompt(s)")
        for pi, ep in enumerate(extend_prompts):
            log.info(f"  Prompt [{pi+1}]: {ep[:100]}...")
        log.info(f"{'='*60}")

        prev_video = video  # (1, C, T, H, W) in [-1,1]
        num_cond_latent = args.num_input_frames
        model_rf = pipe.model.tokenizer.get_pixel_num_frames(pipe.model.config.state_t)

        use_latent_mode = args.latent_extension
        round1_latent = getattr(pipe, '_last_latent', None) if use_latent_mode else None
        if use_latent_mode:
            if round1_latent is not None:
                log.info(f"[LATENT mode] Round 1 latent captured: {round1_latent.shape}")
            else:
                log.warning("[LATENT mode] Round 1 latent not captured — falling back to PIXEL mode")
                use_latent_mode = False
        else:
            log.info("[PIXEL mode] Using pixel-space (decode → PNG → re-encode) for extension")

        prev_latent = round1_latent  # only used in latent mode

        for round_idx in range(args.extend_rounds):
            round_num = round_idx + 2
            mode_tag = "LATENT" if (use_latent_mode and prev_latent is not None) else "PIXEL"
            log.info(f"--- Round {round_num} [{mode_tag}] (cond={num_cond_latent} latent frames) ---")

            last_frame_m1p1 = prev_video[0][:, -1]  # (C, H, W) [-1,1]
            last_frame_01 = (last_frame_m1p1 + 1.0) / 2.0
            last_frame_pil = TF.to_pil_image(last_frame_01.cpu().float().clamp(0, 1))
            last_frame_path = os.path.join(args.output_dir, f"round{round_num}_input.png")
            last_frame_pil.save(last_frame_path)

            use_latent_pass = use_latent_mode and prev_latent is not None
            if use_latent_pass:
                _, C_lat, T_lat, H_lat, W_lat = prev_latent.shape
                cond_latent = torch.zeros(1, C_lat, T_lat, H_lat, W_lat)
                take = min(num_cond_latent, T_lat)
                cond_latent[:, :, :take] = prev_latent[:, :, -take:]
                if take < T_lat:
                    cond_latent[:, :, take:] = prev_latent[:, :, -1:].expand(-1, -1, T_lat - take, -1, -1)
                log.info(f"Round {round_num}: Latent-space conditioning: last {take}/{T_lat} latent frames (zero VAE loss)")

                _original_encode = pipe.model.encode
                _cond_lat = cond_latent.to(dtype=torch.float32)
                def _bypass_encode(state):
                    return _cond_lat.to(device=state.device)
                pipe.model.encode = _bypass_encode
                extend_input = last_frame_path
            else:
                extend_input = last_frame_path
                log.info(f"Round {round_num}: Pixel-space conditioning (decode → PNG → VAE encode)")

            _cfg_call_counter[0] = 0
            clip_for_round = _clip_emb

            def _make_extend_forward(clip_emb, counter, orig_fwd):
                def _fwd(*a, **kw):
                    dev = a[0].device if a else "cuda"
                    is_cond = (counter[0] % 2 == 0)
                    counter[0] += 1
                    if is_cond:
                        kw["img_context_emb"] = clip_emb.to(device=dev, dtype=torch.bfloat16)
                    else:
                        kw["img_context_emb"] = None
                    a = tuple(
                        x.to(dtype=torch.bfloat16) if isinstance(x, torch.Tensor) and x.is_floating_point() else x
                        for x in a
                    )
                    for k, v in kw.items():
                        if isinstance(v, torch.Tensor) and v.is_floating_point():
                            kw[k] = v.to(dtype=torch.bfloat16)
                    return orig_fwd(*a, **kw)
                return _fwd

            net.forward = _make_extend_forward(clip_for_round, _cfg_call_counter, _original_forward)

            round_prompt_idx = min(round_idx, len(extend_prompts) - 1)
            round_prompt = extend_prompts[round_prompt_idx]
            log.info(f"Round {round_num} prompt: {round_prompt[:100]}...")

            t0_round = time.time()
            prev_video = pipe.generate_vid2world(
                prompt=round_prompt,
                input_path=extend_input,
                guidance=args.guidance,
                num_video_frames=args.num_frames,
                num_latent_conditional_frames=num_cond_latent,
                resolution=args.resolution,
                seed=args.seed + round_num,
                negative_prompt=extend_neg,
                num_steps=args.num_steps,
            )
            round_time = time.time() - t0_round

            if use_latent_pass:
                pipe.model.encode = _original_encode
                prev_latent = getattr(pipe, '_last_latent', None)

            round_out = (1.0 + prev_video[0]) / 2.0
            rope_tag = "equirect" if args.equirect_rope else "linear"
            fname = f"pano_v3_{h_pano}x{w_pano}_{rope_tag}_s{args.seed}_round{round_num}"
            round_output = os.path.join(args.output_dir, fname)
            save_img_or_video(round_out, round_output, fps=16)
            log.success(f"Round {round_num}: {round_output}.mp4 ({round_time:.1f}s)")

        net.forward = _original_forward
        log.success(f"All {args.extend_rounds} extension rounds complete! (total: {time.time()-t0_total:.1f}s)")

        # --- Auto-concatenate all rounds into one continuous video ---
        try:
            import torchvision.io as tio_concat
            rope_tag = "equirect" if args.equirect_rope else "linear"
            base_pattern = f"pano_v3_{h_pano}x{w_pano}_{rope_tag}_s{args.seed}"

            round1_path = os.path.join(args.output_dir, f"{base_pattern}.mp4")
            all_round_clips = []
            if os.path.exists(round1_path):
                v1, _, _ = tio_concat.read_video(round1_path)
                all_round_clips.append(v1)
            for ri in range(2, args.extend_rounds + 2):
                rp = os.path.join(args.output_dir, f"{base_pattern}_round{ri}.mp4")
                if os.path.exists(rp):
                    vr, _, _ = tio_concat.read_video(rp)
                    all_round_clips.append(vr)

            if len(all_round_clips) > 1:
                concat_all = torch.cat(all_round_clips, dim=0)
                concat_path = os.path.join(args.output_dir, f"{base_pattern}_concat.mp4")
                tio_concat.write_video(concat_path, concat_all, fps=16)
                log.success(f"Concatenated video: {concat_path} ({concat_all.shape[0]} frames)")
        except Exception as e:
            log.warning(f"Auto-concatenation failed: {e}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.v3:
        if not (args.input_path or args.pers_input):
            raise ValueError("--v3 requires --input_path (image) or --pers_input (video)")
        if args.resolution == "none":
            args.resolution = "256,512"
        run_v3_pipeline(args)
        return

    is_pers2pano = args.pers_input is not None
    is_video2pano = args.input_path is not None and not is_pers2pano

    if is_pers2pano:
        mode_str = "pers2pano"
    elif is_video2pano:
        mode_str = "video2pano"
    else:
        mode_str = "text2pano"

    model_key = MODEL_KEYS[args.model]
    checkpoint = MODEL_CHECKPOINTS[model_key]
    ckpt_path = args.checkpoint_path or checkpoint.path
    experiment = checkpoint.experiment
    config_file = "cosmos_predict2/_src/predict2/configs/video2world/config.py"

    experiment_opts = []
    if args.use_clip and args.finetune_checkpoint:
        experiment_opts.append("++model.config.net.extra_image_context_dim=1152")
        experiment_opts.append("++model.config.net.reference_channels=16")
    if model_key.distilled:
        experiment_opts.append("model.config.init_student_with_teacher=False")
        config_file = "cosmos_predict2/_src/predict2/distill/configs/registry_predict2p5.py"

    log.info("=" * 60)
    log.info("Cosmos Predict 2.5 - Panoramic Video Generation")
    log.info("=" * 60)
    log.info(f"Model: {args.model}")
    log.info(f"Resolution: {args.resolution}")
    log.info(f"Frames: {args.num_frames}")
    log.info(f"Equirect RoPE: {args.equirect_rope}")
    log.info(f"Mode: {mode_str}")
    if is_pers2pano:
        log.info(f"Perspective input: {args.pers_input}")
        log.info(f"Camera: fov_x={args.fov_x}, yaw={args.yaw:.2f}, pitch={args.pitch:.2f}, roll={args.roll:.2f}")
    log.info(f"Prompt: {args.prompt}")
    log.info("=" * 60)

    log.info("Loading model...")
    t0 = time.time()
    pipe = Video2WorldInference(
        experiment_name=experiment,
        ckpt_path=ckpt_path,
        s3_credential_path="",
        context_parallel_size=1,
        config_file=config_file,
        experiment_opts=experiment_opts,
        offload_diffusion_model=args.offload_diffusion_model,
        offload_text_encoder=args.offload_text_encoder,
        offload_tokenizer=args.offload_tokenizer,
    )
    log.info(f"Model loaded in {time.time() - t0:.1f}s")

    # Load fine-tuned checkpoint if provided
    if args.finetune_checkpoint:
        log.info(f"Loading fine-tuned checkpoint: {args.finetune_checkpoint}")
        ft_sd = torch.load(args.finetune_checkpoint, map_location="cpu", weights_only=False)
        ft_net_keys = [k for k in ft_sd if k.startswith("net.")]
        ft_i2v_keys = [k for k in ft_sd if "k_img" in k or "v_img" in k or "img_context_proj" in k]
        log.info(f"  Fine-tuned checkpoint: {len(ft_sd)} keys, {len(ft_net_keys)} net keys, {len(ft_i2v_keys)} I2V keys")

        if ft_i2v_keys and args.use_clip:
            log.info("  I2V keys detected + --use_clip: overlaying weights onto pipeline net (cosmos_v1_2B_pano)")
            net_sd = {k.replace("net.", ""): v for k, v in ft_sd.items() if k.startswith("net.")}
            load_info = pipe.model.net.load_state_dict(net_sd, strict=False)
            log.info(f"  Net load info: missing={len(load_info.missing_keys)}, unexpected={len(load_info.unexpected_keys)}")
            if load_info.missing_keys:
                log.info(f"  Sample missing: {load_info.missing_keys[:5]}")
            if load_info.unexpected_keys:
                log.info(f"  Sample unexpected: {load_info.unexpected_keys[:5]}")
        else:
            load_info = pipe.model.load_state_dict(ft_sd, strict=False)
            if load_info is not None:
                log.info(f"  Overlay load: missing={len(load_info.missing_keys)}, unexpected={len(load_info.unexpected_keys)}")
            else:
                log.info("  Fine-tuned weights overlaid (model.load_state_dict returned None, typical for custom models)")

        del ft_sd
        torch.cuda.empty_cache()
        log.info("  Fine-tuned weights loaded successfully")

    # Auto-resolve resolution for panoramic generation
    if args.resolution == "none":
        model_res = getattr(pipe.model.config, "resolution", "720")
        pano_res_map = {
            "720": "704,1408",
            "512": "512,1024",
            "480": "480,960",
            "256": "256,512",
        }
        args.resolution = pano_res_map.get(model_res, "704,1408")
        log.info(f"Auto-selected pano resolution: {args.resolution} (model config.resolution={model_res})")
    if args.equirect_rope:
        log.info("Switching RoPE position encoding to equirectangular mode...")
        net = pipe.model.net
        if hasattr(net, 'pos_embedder'):
            net.pos_embedder.grid_type = "equirectangular"
            log.info(f"  Set pos_embedder.grid_type = equirectangular (model: {type(net).__name__})")
        elif hasattr(net, 'rope_position_embedding'):
            net.rope_position_embedding.grid_type = "equirectangular"
            net.rope_position_embedding._is_initialized = False
            log.info(f"  Set rope_position_embedding.grid_type = equirectangular (model: {type(net).__name__})")
        else:
            log.warning(f"Could not find position embedding on {type(net).__name__}.")

    # ── Panoramic inference optimizations ──────────────────────────────────────
    _orig_decode_fn = None
    _orig_generate_fn = None

    # (1) Circular Padding: wrap latent columns before VAE decode for seamless L/R
    if args.latent_padding_size > 0:
        _orig_decode_fn = pipe.model.decode
        _circ_pad = args.latent_padding_size

        @torch.no_grad()
        def _circular_decode(latent):
            left_pad = latent[..., -_circ_pad:]
            right_pad = latent[..., :_circ_pad]
            padded = torch.cat([left_pad, latent, right_pad], dim=-1)
            decoded = _orig_decode_fn(padded)
            px_pad = _circ_pad * 8
            return decoded[..., px_pad:-px_pad]

        pipe.model.decode = _circular_decode
        log.info(f"[Pano Opt] Circular latent padding: {_circ_pad} cols/side → {_circ_pad*8} px/side")

    # (2) Rotated Denoising + (3) ERP Warp Noise
    if args.rotated_denoise or args.erp_warp_noise:
        import tqdm as _tqdm_mod
        from cosmos_predict2._src.imaginaire.utils import misc as _misc

        _orig_generate_fn = pipe.model.generate_samples_from_batch
        _do_rotate = args.rotated_denoise
        _do_warp = args.erp_warp_noise
        _model_ref = pipe.model

        @torch.no_grad()
        def _pano_optimized_generate(
            data_batch, guidance=1.5, seed=1, state_shape=None,
            n_sample=None, is_negative_prompt=False, num_steps=35,
            shift=5.0, **kwargs,
        ):
            model = _model_ref
            model._normalize_video_databatch_inplace(data_batch)
            model._augment_image_dim_inplace(data_batch)
            is_image_batch = model.is_image_batch(data_batch)
            input_key = model.input_image_key if is_image_batch else model.input_data_key
            if n_sample is None:
                n_sample = data_batch[input_key].shape[0]
            if state_shape is None:
                _T, _H, _W = data_batch[input_key].shape[-3:]
                state_shape = [
                    model.config.state_ch,
                    model.tokenizer.get_latent_num_frames(_T),
                    _H // model.tokenizer.spatial_compression_factor,
                    _W // model.tokenizer.spatial_compression_factor,
                ]

            noise = _misc.arch_invariant_rand(
                (n_sample,) + tuple(state_shape),
                torch.float32,
                model.tensor_kwargs["device"],
                seed,
            )

            if _do_warp:
                noise = erp_warp_latent(noise)

            seed_g = torch.Generator(device=model.tensor_kwargs["device"])
            seed_g.manual_seed(seed)

            model.sample_scheduler.set_timesteps(
                num_steps, device=model.tensor_kwargs["device"], shift=shift,
                use_kerras_sigma=getattr(model.config, 'use_kerras_sigma_at_inference', False),
            )
            timesteps = model.sample_scheduler.timesteps

            velocity_fn = model.get_velocity_fn_from_batch(
                data_batch, guidance, is_negative_prompt=is_negative_prompt,
            )
            latents = noise

            sum_shift = 0
            desc = "Generating"
            if _do_rotate:
                desc += " (rotated)"
            if _do_warp:
                desc += " (erp-warp)"

            for _, t in enumerate(_tqdm_mod.tqdm(timesteps, desc=desc)):
                timestep = torch.stack([t])
                velocity_pred = velocity_fn(noise, latents, timestep.unsqueeze(0))
                temp_x0 = model.sample_scheduler.step(
                    velocity_pred.unsqueeze(0), t, latents[0].unsqueeze(0),
                    return_dict=False, generator=seed_g,
                )[0]
                latents = temp_x0.squeeze(0)

                if _do_rotate:
                    W_lat = latents.shape[-1]
                    rnd_shift = torch.randint(0, W_lat, (1,)).item()
                    sum_shift += rnd_shift
                    latents = torch.roll(latents, shifts=rnd_shift, dims=-1)

            if _do_rotate and sum_shift != 0:
                latents = torch.roll(latents, shifts=(-sum_shift) % latents.shape[-1], dims=-1)

            return latents

        pipe.model.generate_samples_from_batch = _pano_optimized_generate
        tags = []
        if _do_rotate:
            tags.append("rotated_denoise")
        if _do_warp:
            tags.append("erp_warp_noise")
        log.info(f"[Pano Opt] Patched sampling loop: {', '.join(tags)}")

    # ---- Phase 2: pers2pano with spatial inpainting ----
    if is_pers2pano:
        h, w = [int(x) for x in args.resolution.split(",")]

        log.info("Reading perspective input video...")
        model_required_frames = pipe.model.tokenizer.get_pixel_num_frames(pipe.model.config.state_t)
        pers_video = read_perspective_video(args.pers_input, num_frames=model_required_frames)
        log.info(f"  Perspective video shape: {pers_video.shape}")

        log.info("Preparing spatial inpainting condition...")
        pers_condition = prepare_pers2pano_condition(
            pipe=pipe,
            pers_video=pers_video,
            fov_x=args.fov_x,
            roll=args.roll,
            pitch=args.pitch,
            yaw=args.yaw,
            equi_h=h,
            equi_w=w,
        )

        # Free pers_video GPU memory
        del pers_video
        torch.cuda.empty_cache()

        # Save the projected equirectangular for reference
        equi_vis = (pers_condition["equi_proj_vis"][0] + 1.0) / 2.0
        equi_vis_path = os.path.join(args.output_dir, "pers2equi_projected")
        save_img_or_video(equi_vis, equi_vis_path, fps=16)
        del equi_vis
        log.info(f"  Saved projected equirectangular to {equi_vis_path}.mp4")

        # Detect model type and apply spatial inpainting accordingly
        model = pipe.model
        is_rectified_flow = hasattr(model, 'get_velocity_fn_from_batch') and not hasattr(model, 'get_x0_fn_from_batch')

        if is_rectified_flow:
            log.info("Detected rectified flow model — using per-step latent replacement for spatial inpainting")
            original_generate = model.generate_samples_from_batch

            guided_image = pers_condition["guided_image"]
            guided_mask = pers_condition["guided_mask"]

            @torch.no_grad()
            def patched_generate(data_batch, guidance=1.5, seed=1, state_shape=None,
                                 n_sample=None, is_negative_prompt=False, num_steps=35,
                                 shift=5.0, **kwargs):
                import tqdm as tqdm_mod
                from cosmos_predict2._src.imaginaire.utils import misc
                from einops import rearrange

                model._normalize_video_databatch_inplace(data_batch)
                model._augment_image_dim_inplace(data_batch)
                is_image_batch = model.is_image_batch(data_batch)
                input_key = model.input_image_key if is_image_batch else model.input_data_key
                if n_sample is None:
                    n_sample = data_batch[input_key].shape[0]
                if state_shape is None:
                    _T, _H, _W = data_batch[input_key].shape[-3:]
                    state_shape = [
                        model.config.state_ch,
                        model.tokenizer.get_latent_num_frames(_T),
                        _H // model.tokenizer.spatial_compression_factor,
                        _W // model.tokenizer.spatial_compression_factor,
                    ]

                noise = misc.arch_invariant_rand(
                    (n_sample,) + tuple(state_shape),
                    torch.float32,
                    model.tensor_kwargs["device"],
                    seed,
                )
                seed_g = torch.Generator(device=model.tensor_kwargs["device"])
                seed_g.manual_seed(seed)

                model.sample_scheduler.set_timesteps(
                    num_steps, device=model.tensor_kwargs["device"], shift=shift,
                    use_kerras_sigma=model.config.use_kerras_sigma_at_inference
                    if hasattr(model.config, 'use_kerras_sigma_at_inference') else False,
                )
                timesteps = model.sample_scheduler.timesteps

                velocity_fn = model.get_velocity_fn_from_batch(
                    data_batch, guidance, is_negative_prompt=is_negative_prompt,
                )
                latents = noise

                g_img = guided_image.to(device=latents.device, dtype=latents.dtype)
                g_mask = guided_mask.to(device=latents.device, dtype=latents.dtype)
                if g_mask.shape[2:] != latents.shape[2:]:
                    g_mask = F.interpolate(
                        g_mask.view(1, -1, *g_mask.shape[2:]).float(),
                        size=latents.shape[2:], mode="nearest",
                    ).view(g_mask.shape[0], 1, *latents.shape[2:]).to(dtype=latents.dtype)
                if g_img.shape[2:] != latents.shape[2:]:
                    g_img = F.interpolate(
                        g_img.float(), size=latents.shape[2:], mode="nearest",
                    ).to(dtype=latents.dtype)

                for step_idx, t in enumerate(tqdm_mod.tqdm(timesteps, desc="Generating (pers2pano)")):
                    timestep = torch.stack([t])
                    velocity_pred = velocity_fn(noise, latents, timestep.unsqueeze(0))
                    temp_x0 = model.sample_scheduler.step(
                        velocity_pred.unsqueeze(0), t, latents[0].unsqueeze(0),
                        return_dict=False, generator=seed_g,
                    )[0]
                    latents = temp_x0.squeeze(0)

                    # Spatial replacement: blend guided latent at current noise level
                    t_frac = t.item() / 1000.0
                    noisy_guide = g_img * (1.0 - t_frac) + noise * t_frac
                    latents = g_mask * noisy_guide + (1.0 - g_mask) * latents

                return latents

            model.generate_samples_from_batch = patched_generate
        else:
            log.info("Detected EDM model — using guided_image/guided_mask for spatial inpainting")
            original_get_x0_fn = model.get_x0_fn_from_batch

            def patched_get_x0_fn(data_batch, guidance=1.5, is_negative_prompt=False):
                data_batch["guided_image"] = pers_condition["guided_image"]
                data_batch["guided_mask"] = pers_condition["guided_mask"]
                return original_get_x0_fn(data_batch, guidance, is_negative_prompt)

            model.get_x0_fn_from_batch = patched_get_x0_fn

        log.info("Generating panoramic video with spatial inpainting...")
        t0 = time.time()
        video = pipe.generate_vid2world(
            prompt=args.prompt,
            input_path=None,
            guidance=args.guidance,
            num_video_frames=args.num_frames,
            num_latent_conditional_frames=0,
            resolution=args.resolution,
            seed=args.seed,
            negative_prompt=args.negative_prompt,
            num_steps=args.num_steps,
        )
        gen_time = time.time() - t0

        if is_rectified_flow:
            model.generate_samples_from_batch = original_generate
        else:
            model.get_x0_fn_from_batch = original_get_x0_fn

    # ---- Standard video2pano / text2pano / img2pano ----
    else:
        is_img2pano_reflatent = (
            args.use_clip and args.finetune_checkpoint and args.input_path is not None
        )

        if args.input_path is None:
            num_latent_conditional_frames = 0
        elif is_img2pano_reflatent:
            num_latent_conditional_frames = 1
        else:
            num_latent_conditional_frames = args.num_input_frames

        clip_img_context_emb = None
        ref_latent = None
        guided_img_latent = None
        guided_mask_latent = None

        if is_img2pano_reflatent:
            h, w = [int(x) for x in args.resolution.split(",")]
            log.info("=== ARGUS-style img2pano: CLIP + reference latent + blended diffusion ===")

            log.info("Encoding input with CLIP for image cross-attention...")
            clip_img_context_emb = encode_clip_image(args.input_path)

            log.info("Preparing reference latent + guided latent (pers→equi + VAE)...")
            ref_latent, guided_img_latent, guided_mask_latent = prepare_reference_latent(
                pipe=pipe,
                input_path=args.input_path,
                equi_h=h, equi_w=w,
                fov_x=args.fov_x,
                noise_strength=0.5,
                yaw_rad=args.yaw,
            )
        elif args.use_clip and args.input_path is not None:
            log.info("Encoding input with CLIP for image cross-attention...")
            clip_img_context_emb = encode_clip_image(args.input_path)

        _original_net_forward = None
        if clip_img_context_emb is not None or ref_latent is not None:
            net = pipe.model.net
            _original_net_forward = net.forward
            _clip_emb = clip_img_context_emb.to(dtype=torch.bfloat16) if clip_img_context_emb is not None else None
            _ref_lat = ref_latent.to(dtype=torch.bfloat16) if ref_latent is not None else None
            log.info(f"Monkey-patching net.forward (CLIP={_clip_emb is not None}, ref_latent={_ref_lat is not None})")

            def _patched_net_forward(*args_fwd, **kwargs_fwd):
                if _clip_emb is not None:
                    kwargs_fwd["img_context_emb"] = _clip_emb.to(
                        device=args_fwd[0].device if args_fwd else "cuda"
                    )
                if _ref_lat is not None:
                    kwargs_fwd["reference_latent_B_C_T_H_W"] = _ref_lat.to(
                        device=args_fwd[0].device if args_fwd else "cuda",
                        dtype=torch.bfloat16,
                    )
                args_fwd = tuple(
                    a.to(dtype=torch.bfloat16) if isinstance(a, torch.Tensor) and a.is_floating_point() else a
                    for a in args_fwd
                )
                for k, v in kwargs_fwd.items():
                    if isinstance(v, torch.Tensor) and v.is_floating_point():
                        kwargs_fwd[k] = v.to(dtype=torch.bfloat16)
                return _original_net_forward(*args_fwd, **kwargs_fwd)

            net.forward = _patched_net_forward

        log.info("Generating panoramic video...")
        t0 = time.time()

        if is_img2pano_reflatent and guided_img_latent is not None:
            log.info("  Using blended diffusion: observed regions replaced at each denoising step")
            model = pipe.model

            original_generate = model.generate_samples_from_batch
            _g_img = guided_img_latent
            _g_mask = guided_mask_latent

            @torch.no_grad()
            def patched_generate_argus(data_batch, guidance=1.5, seed=1, state_shape=None,
                                       n_sample=None, is_negative_prompt=False, num_steps=35,
                                       shift=5.0, **kwargs):
                import tqdm as tqdm_mod
                from cosmos_predict2._src.imaginaire.utils import misc

                model._normalize_video_databatch_inplace(data_batch)
                model._augment_image_dim_inplace(data_batch)
                is_image_batch = model.is_image_batch(data_batch)
                input_key = model.input_image_key if is_image_batch else model.input_data_key
                if n_sample is None:
                    n_sample = data_batch[input_key].shape[0]
                if state_shape is None:
                    _T, _H, _W = data_batch[input_key].shape[-3:]
                    state_shape = [
                        model.config.state_ch,
                        model.tokenizer.get_latent_num_frames(_T),
                        _H // model.tokenizer.spatial_compression_factor,
                        _W // model.tokenizer.spatial_compression_factor,
                    ]

                noise = misc.arch_invariant_rand(
                    (n_sample,) + tuple(state_shape),
                    torch.float32,
                    model.tensor_kwargs["device"],
                    seed,
                )
                seed_g = torch.Generator(device=model.tensor_kwargs["device"])
                seed_g.manual_seed(seed)

                model.sample_scheduler.set_timesteps(
                    num_steps, device=model.tensor_kwargs["device"], shift=shift,
                    use_kerras_sigma=getattr(model.config, 'use_kerras_sigma_at_inference', False),
                )
                timesteps = model.sample_scheduler.timesteps

                velocity_fn = model.get_velocity_fn_from_batch(
                    data_batch, guidance, is_negative_prompt=is_negative_prompt,
                )
                latents = noise

                g_img = _g_img.to(device=latents.device, dtype=latents.dtype)
                g_mask = _g_mask.to(device=latents.device, dtype=latents.dtype)
                if g_mask.shape[2:] != latents.shape[2:]:
                    g_mask = F.interpolate(
                        g_mask.float(), size=latents.shape[2:], mode="nearest",
                    ).to(dtype=latents.dtype)
                if g_img.shape[2:] != latents.shape[2:]:
                    g_img = F.interpolate(
                        g_img.float(), size=latents.shape[2:], mode="nearest",
                    ).to(dtype=latents.dtype)

                for step_idx, t in enumerate(tqdm_mod.tqdm(timesteps, desc="Generating (ARGUS blended)")):
                    timestep = torch.stack([t])
                    velocity_pred = velocity_fn(noise, latents, timestep.unsqueeze(0))
                    temp_x0 = model.sample_scheduler.step(
                        velocity_pred.unsqueeze(0), t, latents[0].unsqueeze(0),
                        return_dict=False, generator=seed_g,
                    )[0]
                    latents = temp_x0.squeeze(0)

                    t_frac = t.item() / 1000.0
                    noisy_guide = g_img * (1.0 - t_frac) + noise * t_frac
                    latents = g_mask * noisy_guide + (1.0 - g_mask) * latents

                return latents

            model.generate_samples_from_batch = patched_generate_argus

            video = pipe.generate_vid2world(
                prompt=args.prompt,
                input_path=None,
                guidance=args.guidance,
                num_video_frames=args.num_frames,
                num_latent_conditional_frames=0,
                resolution=args.resolution,
                seed=args.seed,
                negative_prompt=args.negative_prompt,
                num_steps=args.num_steps,
            )
            model.generate_samples_from_batch = original_generate
        else:
            video = pipe.generate_vid2world(
                prompt=args.prompt,
                input_path=args.input_path,
                guidance=args.guidance,
                num_video_frames=args.num_frames,
                num_latent_conditional_frames=num_latent_conditional_frames,
                resolution=args.resolution,
                seed=args.seed,
                negative_prompt=args.negative_prompt,
                num_steps=args.num_steps,
            )
        gen_time = time.time() - t0
        if _original_net_forward is not None:
            pipe.model.net.forward = _original_net_forward
            log.info("Restored original net.forward")

    log.info(f"Generation completed in {gen_time:.1f}s")

    video_out = (1.0 + video[0]) / 2.0

    rope_tag = "equirect" if args.equirect_rope else "linear"
    if "," in args.resolution:
        h, w = args.resolution.split(",")
        res_tag = f"{h}x{w}"
    else:
        res_tag = f"{video_out.shape[-2]}x{video_out.shape[-1]}"
    filename = f"pano_{mode_str}_{res_tag}_{rope_tag}_s{args.seed}"
    output_path = os.path.join(args.output_dir, filename)

    save_img_or_video(video_out, output_path, fps=16)
    log.info(f"Saved to {output_path}.mp4")

    # Cubemap visualization
    if args.cubemap_viz:
        log.info("Rendering cubemap visualization...")
        save_cubemap_visualization(video_out, output_path,
                                   face_size=args.cubemap_face_size, fps=16)

    info = vars(args).copy()
    info["mode"] = mode_str
    info["generation_time_s"] = gen_time
    with open(f"{output_path}.json", "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # Restore monkey-patched functions
    if _orig_decode_fn is not None:
        pipe.model.decode = _orig_decode_fn
        log.info("Restored original model.decode")
    if _orig_generate_fn is not None:
        pipe.model.generate_samples_from_batch = _orig_generate_fn
        log.info("Restored original generate_samples_from_batch")

    log.success(f"Done! Output: {output_path}.mp4")


if __name__ == "__main__":
    main()
