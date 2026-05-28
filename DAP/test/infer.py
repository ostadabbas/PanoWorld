from __future__ import absolute_import, division, print_function

import os
import sys
import cv2
import torch
import yaml
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from networks.models import make  # prefer `make` over `import *`

import matplotlib

def colorize_depth_fixed(depth_u8: np.ndarray, cmap: str = "Spectral") -> np.ndarray:
    """
    depth_u8: uint8, 0~255
    return: RGB uint8
    """
    disp = depth_u8.astype(np.float32) / 255.0
    colored = matplotlib.colormaps[cmap](disp)[..., :3]
    colored = (colored * 255).astype(np.uint8)
    return np.ascontiguousarray(colored)

def ensure_dir_for_file(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def load_model(config):
    model_path = os.path.join(config["load_weights_dir"], "model.pth")
    print(f"🔹 Loading model weights from: {model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(model_path, map_location=device)

    m = make(config["model"])
    if any(k.startswith("module") for k in state.keys()):
        m = nn.DataParallel(m)

    m = m.to(device)
    m_state = m.state_dict()
    m.load_state_dict({k: v for k, v in state.items() if k in m_state}, strict=False)
    m.eval()

    print("✅ Model loaded successfully.\n")
    return m, device

def infer_raw(model, device, img_rgb_u8: np.ndarray) -> np.ndarray:
    """
    img_rgb_u8: HWC uint8 RGB
    return: pred float32 (H,W)
    """
    img = img_rgb_u8.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(tensor)

        if isinstance(outputs, dict) and "pred_depth" in outputs:
            if "pred_mask" in outputs:
                mask = 1 - outputs["pred_mask"]
                mask = mask > 0.5
                outputs["pred_depth"][~mask] = 1
            pred = outputs["pred_depth"][0].detach().cpu().squeeze().numpy()
        else:
            pred = outputs[0].detach().cpu().squeeze().numpy()

    return pred.astype(np.float32)

def pred_to_vis(pred: np.ndarray, vis_range: str = "100m", cmap: str = "Spectral"):
    """
    return:
      depth_gray_u8: (H,W) uint8
      depth_color_rgb: (H,W,3) uint8 RGB
    """
    if vis_range == "100m":
        pred_clip = np.clip(pred, 0.0, 1.0)
        depth_gray = (pred_clip * 255).astype(np.uint8)
    elif vis_range == "10m":
        pred_clip = np.clip(pred, 0.0, 0.1)
        depth_gray = (pred_clip * 10.0 * 255).astype(np.uint8)
    else:
        raise ValueError(f"Unknown vis_range: {vis_range} (use '100m' or '10m')")

    depth_color = colorize_depth_fixed(depth_gray, cmap=cmap)
    return depth_gray, depth_color

def infer_and_save(model, device, img_path, out_root, idx, vis_range="100m", cmap="Spectral"):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"⚠️ Cannot read image: {img_path}")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    pred = infer_raw(model, device, img_rgb)

    depth_gray, depth_color_rgb = pred_to_vis(pred, vis_range=vis_range, cmap=cmap)

    filename = f"{idx:06d}"

    pred_npy_path = os.path.join(out_root, "depth_npy", filename + ".npy")
    gray_png_path = os.path.join(out_root, f"depth_vis_gray_{vis_range}", filename + ".png")
    color_png_path = os.path.join(out_root, f"depth_vis_color_{vis_range}", filename + ".png")

    ensure_dir_for_file(pred_npy_path)
    ensure_dir_for_file(gray_png_path)
    ensure_dir_for_file(color_png_path)

    np.save(pred_npy_path, pred)

    cv2.imwrite(gray_png_path, depth_gray)

    cv2.imwrite(color_png_path, cv2.cvtColor(depth_color_rgb, cv2.COLOR_RGB2BGR))


def main(config_path, txt_path, out_root, vis_range="100m", cmap="Spectral"):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print("✅ Config loaded.")

    model, device = load_model(config)

    with open(txt_path, "r") as f:
        img_list = [l.strip() for l in f.readlines() if l.strip()]

    print(f"🔹 Total images to infer: {len(img_list)}")
    print(f"🔹 Visualization: {vis_range}, colormap: {cmap}\n")

    for idx, img_path in enumerate(tqdm(img_list, desc="Inferencing"), start=1):
        infer_and_save(model, device, img_path, out_root, idx, vis_range=vis_range, cmap=cmap)

    print("\nInference done.")
    print(f"   depth npy: {out_root}/depth_npy")
    print(f"   depth gray png: {out_root}/depth_vis_gray_{vis_range}")
    print(f"   depth color png: {out_root}/depth_vis_color_{vis_range}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/infer.yaml")
    parser.add_argument("--txt", default="datasets/test.txt")
    parser.add_argument("--output", default="test_output")
    parser.add_argument("--gpu", default="0", help="CUDA device index to use")

    parser.add_argument("--vis", default="10m", choices=["100m", "10m"], help="visualization range; only affects PNG export, not the NPY")
    parser.add_argument("--cmap", default="Spectral", help="matplotlib colormap name, e.g. Spectral, Turbo, Viridis")

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    main(args.config, args.txt, args.output, vis_range=args.vis, cmap=args.cmap)
