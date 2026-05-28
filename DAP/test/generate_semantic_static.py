"""Generate semantic_static.npy for annotation folders using GroundingSAM.

Uses Grounded SAM 2 (GroundingDINO + SAM2) to detect dynamic objects
(people, vehicles, animals) in the first frame, then classifies each
CoTracker track as semantic-static or semantic-dynamic based on whether
its first-visible position falls inside a detected dynamic-object mask.

Output: semantic_static.npy — (N,) float32 array, 1=static 0=dynamic.
        This can be fused with the geometric track_is_static.npy for
        more robust static/dynamic classification in training.

Environment: Run in the 'gsam' conda environment.

Usage:
    conda run -n gsam python DAP/test/generate_semantic_static.py \
        --ann_dir $PANO_DATA_ROOT/annotations \
        --video_dir $PANO_DATA_ROOT/cosmos_pano_train/videos \
        --gsam_root $GSAM_ROOT  # path to your Grounded-SAM-2 checkout

    # Specific folders only:
    conda run -n gsam python DAP/test/generate_semantic_static.py \
        --ann_dir $PANO_DATA_ROOT/annotations \
        --video_dir $PANO_DATA_ROOT/cosmos_pano_train/videos \
        --gsam_root $GSAM_ROOT  # path to your Grounded-SAM-2 checkout \
        --folders web360_100001 web360_100003
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import box_convert
from tqdm import tqdm

# Dynamic object categories for panoramic indoor/outdoor scenes
DYNAMIC_CLASSES = (
    "person. people. man. woman. child. "
    "car. vehicle. truck. bus. motorcycle. bicycle. "
    "dog. cat. animal. bird."
)

BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def setup_gsam(gsam_root: str):
    """Load GroundingDINO and SAM2 models."""
    gsam_root = Path(gsam_root)

    sys.path.insert(0, str(gsam_root))

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from grounding_dino.groundingdino.util.inference import load_model, predict

    sam2_ckpt = str(gsam_root / "checkpoints" / "sam2.1_hiera_large.pt")
    sam2_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    gdino_cfg = str(gsam_root / "grounding_dino" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py")
    gdino_ckpt = str(gsam_root / "gdino_checkpoints" / "groundingdino_swint_ogc.pth")

    print("Loading SAM2 model...")
    sam2_model = build_sam2(sam2_cfg, sam2_ckpt, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    print("Loading GroundingDINO model...")
    gdino_model = load_model(
        model_config_path=gdino_cfg,
        model_checkpoint_path=gdino_ckpt,
        device=DEVICE,
    )

    return sam2_predictor, gdino_model, predict


def extract_frames(video_path: str, indices: list[int] | None = None) -> list[np.ndarray]:
    """Extract specific frames from a video file. Returns list of BGR frames.
    If indices is None, extracts only the first frame."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if indices is None:
        indices = [0]
    frames = []
    for idx in sorted(set(indices)):
        if idx >= total:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"Failed to read any frames from {video_path}")
    return frames


def get_dynamic_masks(
    frame_bgr: np.ndarray,
    sam2_predictor,
    gdino_model,
    predict_fn,
    text_prompt: str = DYNAMIC_CLASSES,
) -> np.ndarray:
    """Run GroundingSAM on a single frame, return a combined binary mask
    of all detected dynamic objects. Shape: (H, W) bool."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]

    from grounding_dino.groundingdino.util.inference import load_image
    import tempfile

    tmp_path = os.path.join(tempfile.gettempdir(), "_gsam_tmp_frame.jpg")
    cv2.imwrite(tmp_path, frame_bgr)
    image_source, image_tensor = load_image(tmp_path)

    with torch.no_grad():
        boxes, confidences, labels = predict_fn(
            model=gdino_model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=DEVICE,
        )

    if len(boxes) == 0:
        return np.zeros((h, w), dtype=bool)

    boxes_abs = boxes * torch.Tensor([w, h, w, h]).to(boxes.device)
    input_boxes = box_convert(boxes=boxes_abs, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()

    sam2_predictor.set_image(frame_rgb)

    with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

    if masks.ndim == 4:
        masks = masks.squeeze(1)

    combined_mask = masks.any(axis=0)
    return combined_mask


def classify_tracks_semantic(
    tracks_2d: np.ndarray,
    visibility: np.ndarray,
    dynamic_mask: np.ndarray,
    dilate_px: int = 15,
    hit_ratio_thresh: float = 0.3,
) -> np.ndarray:
    """Classify each track as static/dynamic using vectorized mask lookup.

    A track is marked dynamic if >= hit_ratio_thresh of its visible-frame
    positions fall inside the (dilated) dynamic mask.

    Args:
        tracks_2d: (N, T, 2) pixel coordinates
        visibility: (N, T) float, >0.5 means visible
        dynamic_mask: (H, W) bool, True = dynamic object region
        dilate_px: dilation radius in pixels (accounts for mask boundary errors)
        hit_ratio_thresh: fraction of visible frames that must hit the mask

    Returns:
        semantic_static: (N,) float32, 1=static, 0=dynamic
    """
    N, T, _ = tracks_2d.shape
    H, W = dynamic_mask.shape

    if dilate_px > 0 and dynamic_mask.any():
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
        )
        dilated = cv2.dilate(dynamic_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    else:
        dilated = dynamic_mask

    vis_mask = visibility > 0.5  # (N, T)
    px = np.clip(np.round(tracks_2d[:, :, 0]).astype(int), 0, W - 1)  # (N, T)
    py = np.clip(np.round(tracks_2d[:, :, 1]).astype(int), 0, H - 1)  # (N, T)
    in_dynamic = dilated[py, px]  # (N, T) bool via advanced indexing
    hits = (in_dynamic & vis_mask).sum(axis=1)  # (N,)
    vis_count = vis_mask.sum(axis=1).clip(min=1)  # (N,)
    hit_ratio = hits / vis_count
    semantic_static = np.where(hit_ratio >= hit_ratio_thresh, 0.0, 1.0).astype(np.float32)

    return semantic_static


N_SAMPLE_FRAMES = 3  # first, middle, last


def process_folder(
    ann_dir: str,
    video_dir: str,
    folder_name: str,
    sam2_predictor,
    gdino_model,
    predict_fn,
    force: bool = False,
):
    """Process a single annotation folder.

    Runs GroundingSAM on up to N_SAMPLE_FRAMES (first/middle/last) and
    unions the dynamic masks to maximize recall for dynamic objects.
    """
    ann_path = os.path.join(ann_dir, folder_name)
    out_path = os.path.join(ann_path, "semantic_static.npy")

    if os.path.exists(out_path) and not force:
        return "skip"

    meta_path = os.path.join(ann_path, "meta.json")
    tracks_path = os.path.join(ann_path, "tracks_2d.npy")
    vis_path = os.path.join(ann_path, "visibility.npy")

    if not all(os.path.exists(p) for p in [meta_path, tracks_path, vis_path]):
        return "missing"

    with open(meta_path) as f:
        meta = json.load(f)
    video_res = meta.get("video_resolution", [1024, 512])
    n_frames = meta.get("n_frames", 20)
    W_vid, H_vid = video_res[0], video_res[1]

    video_path = os.path.join(video_dir, f"{folder_name}.mp4")
    if not os.path.exists(video_path):
        return "no_video"

    frame_indices = [0, n_frames // 2, max(n_frames - 1, 0)]
    frame_indices = sorted(set(frame_indices))
    frames = extract_frames(video_path, frame_indices)

    combined_dynamic = np.zeros((H_vid, W_vid), dtype=bool)
    for frame_bgr in frames:
        h_f, w_f = frame_bgr.shape[:2]
        if (h_f, w_f) != (H_vid, W_vid):
            frame_bgr = cv2.resize(frame_bgr, (W_vid, H_vid))
        mask = get_dynamic_masks(
            frame_bgr, sam2_predictor, gdino_model, predict_fn
        )
        combined_dynamic |= mask

    tracks_2d = np.load(tracks_path)
    visibility = np.load(vis_path)

    semantic_static = classify_tracks_semantic(
        tracks_2d, visibility, combined_dynamic
    )

    np.save(out_path, semantic_static)

    n_dynamic = int((semantic_static < 0.5).sum())
    n_total = len(semantic_static)
    return f"ok ({n_dynamic}/{n_total} dynamic)"


def main():
    parser = argparse.ArgumentParser(description="Generate semantic_static.npy via GroundingSAM")
    parser.add_argument("--ann_dir", required=True, help="Root annotation directory")
    parser.add_argument("--video_dir", required=True, help="Directory containing .mp4 videos")
    parser.add_argument("--gsam_root", required=True, help="Path to Grounded-SAM-2 repo")
    parser.add_argument("--folders", nargs="*", default=None, help="Specific folders to process")
    parser.add_argument("--force", action="store_true", help="Overwrite existing semantic_static.npy")
    args = parser.parse_args()

    sam2_predictor, gdino_model, predict_fn = setup_gsam(args.gsam_root)

    if args.folders:
        folders = args.folders
    else:
        folders = sorted([
            d for d in os.listdir(args.ann_dir)
            if os.path.isdir(os.path.join(args.ann_dir, d))
        ])

    print(f"\nProcessing {len(folders)} annotation folders...")
    stats = {"ok": 0, "skip": 0, "missing": 0, "no_video": 0, "error": 0}

    for folder_name in tqdm(folders, desc="GroundingSAM semantic"):
        try:
            result = process_folder(
                args.ann_dir, args.video_dir, folder_name,
                sam2_predictor, gdino_model, predict_fn,
                force=args.force,
            )
            if result.startswith("ok"):
                stats["ok"] += 1
            else:
                stats[result] += 1
        except Exception as e:
            stats["error"] += 1
            tqdm.write(f"  ERROR [{folder_name}]: {e}")

    print(f"\nDone! Results: {stats}")


if __name__ == "__main__":
    main()
