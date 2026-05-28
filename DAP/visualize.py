"""
DAP Visualization Script
Generates:
  1. Side-by-side comparison video  (original frame | colorized depth)
  2. 3D equirectangular point cloud  for a chosen frame (saved as PNG + shown)
  3. Depth statistics summary plot
"""

import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from tqdm import tqdm

# ── paths ──────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR  = os.path.join(BASE, "datasets", "video_frames")
NPY_DIR     = os.path.join(BASE, "output", "video_1000050115", "depth_npy")
COLOR_DIR   = os.path.join(BASE, "output", "video_1000050115", "depth_vis_color_100m")
OUT_DIR     = os.path.join(BASE, "output", "video_1000050115", "viz")
os.makedirs(OUT_DIR, exist_ok=True)

CMAP        = "Spectral_r"
MAX_DEPTH_M = 100.0          # model scale: 1.0 → 100 m


# ── helpers ────────────────────────────────────────────────────────────────

def sorted_files(directory, ext):
    return sorted([f for f in os.listdir(directory) if f.endswith(ext)])


def load_frame(path):
    bgr = cv2.imread(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def colorize_depth(depth_m, vmin=0, vmax=20):
    """Return (H,W,3) uint8 RGB using CMAP, clipped to [vmin, vmax] metres."""
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    rgba = matplotlib.colormaps[CMAP](norm(depth_m))
    return (rgba[..., :3] * 255).astype(np.uint8)


# ── 1. side-by-side video ──────────────────────────────────────────────────

def make_comparison_video(fps=6, max_depth_m=None):
    frame_files = sorted_files(FRAMES_DIR, ".png")
    npy_files   = sorted_files(NPY_DIR,    ".npy")
    if not frame_files:
        print("No frames found"); return

    # ── determine far-clip: explicit value or P99 across all frames ──
    if max_depth_m is not None:
        clip = max_depth_m
    else:
        sample_depths = [
            np.load(os.path.join(NPY_DIR, nf)).ravel() * MAX_DEPTH_M
            for nf in npy_files
        ]
        clip = float(np.percentile(np.concatenate(sample_depths), 99))
    print(f"   Depth range: 0 – {clip:.1f} m (consistent across all frames)")

    sample = load_frame(os.path.join(FRAMES_DIR, frame_files[0]))
    H, W   = sample.shape[:2]

    out_path = os.path.join(OUT_DIR, "comparison.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(out_path, fourcc, fps, (W * 2, H))

    print(f"\n[1/3] Building comparison video → {out_path}")
    for ff, nf in tqdm(zip(frame_files, npy_files), total=len(frame_files)):
        rgb   = load_frame(os.path.join(FRAMES_DIR, ff))
        depth = np.load(os.path.join(NPY_DIR, nf)) * MAX_DEPTH_M

        depth_vis = colorize_depth(depth, vmin=0, vmax=clip)

        for img, label in [(rgb, "Original"), (depth_vis, f"Depth  0 – {clip:.0f} m")]:
            cv2.putText(img, label, (14, 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, label, (14, 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 0), 1, cv2.LINE_AA)

        combined = np.concatenate([rgb, depth_vis], axis=1)
        writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"   Saved: {out_path}")


# ── 2. 3-D point cloud ─────────────────────────────────────────────────────

def erp_to_xyz(depth_m, subsample=8):
    """
    Back-project equirectangular depth to 3-D Cartesian (camera = origin).
    subsample: take every Nth pixel to keep point count manageable.
    """
    H, W    = depth_m.shape
    ys, xs  = np.meshgrid(np.arange(0, H, subsample),
                          np.arange(0, W, subsample), indexing="ij")
    d       = depth_m[ys, xs]

    lon = (xs / W) * 2 * np.pi - np.pi          # -π … +π
    lat = (ys / H) * np.pi       - np.pi / 2    # -π/2 … +π/2

    X = d * np.cos(lat) * np.cos(lon)
    Y = d * np.sin(lat)
    Z = d * np.cos(lat) * np.sin(lon)
    return X.ravel(), Y.ravel(), Z.ravel(), d.ravel()


def make_pointcloud(frame_idx=14, subsample=3, max_depth_m=None):
    """
    subsample   : take every Nth pixel (3 → ~58k pts, 2 → ~131k pts, 1 → full res)
    max_depth_m : hard far-clip in metres; None = auto (P99 of frame)
    """
    npy_files   = sorted_files(NPY_DIR, ".npy")
    frame_files = sorted_files(FRAMES_DIR, ".png")
    frame_idx   = min(frame_idx, len(npy_files) - 1)

    depth_m = np.load(os.path.join(NPY_DIR, npy_files[frame_idx])) * MAX_DEPTH_M
    rgb_img = load_frame(os.path.join(FRAMES_DIR, frame_files[frame_idx]))

    # ── build pixel grid at chosen resolution ──
    H, W = depth_m.shape
    ys, xs = np.meshgrid(np.arange(0, H, subsample),
                         np.arange(0, W, subsample), indexing="ij")
    colors = rgb_img[ys, xs].reshape(-1, 3) / 255.0

    X, Y, Z, D = erp_to_xyz(depth_m, subsample)

    # ── far-clip: use explicit value or auto P99 ──
    clip = max_depth_m if max_depth_m is not None else np.percentile(D, 99)
    mask = D <= clip
    X, Y, Z, D, colors = X[mask], Y[mask], Z[mask], D[mask], colors[mask]

    n_pts = mask.sum()
    print(f"\n[2/3] Rendering 3-D point cloud for frame {frame_idx + 1:02d} "
          f"({n_pts:,} points, far-clip={clip:.1f} m) ...")

    # point size: scale down slightly for very dense clouds
    pt_size = max(0.15, 1.8 - n_pts / 60_000)

    fig = plt.figure(figsize=(18, 8))
    fig.patch.set_facecolor("#0d0d0d")

    def _style_ax(ax, title):
        ax.set_facecolor("#0d0d0d")
        ax.set_title(title, color="white", fontsize=11, pad=6)
        ax.set_xlabel("X (m)", color="#aaa", fontsize=7, labelpad=2)
        ax.set_ylabel("Z (m)", color="#aaa", fontsize=7, labelpad=2)
        ax.set_zlabel("Y (m)", color="#aaa", fontsize=7, labelpad=2)
        ax.tick_params(colors="#555", labelsize=6)
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor("#222")
        ax.grid(True, color="#222", linewidth=0.5)

    # ── left: coloured by original RGB ──
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(X, Z, Y, c=colors, s=pt_size, linewidths=0, alpha=0.85)
    _style_ax(ax1, "Point Cloud (RGB texture)")

    # ── right: coloured by depth value ──
    ax2 = fig.add_subplot(122, projection="3d")
    sc = ax2.scatter(X, Z, Y, c=D, cmap=CMAP, s=pt_size, linewidths=0,
                     alpha=0.85, vmin=D.min(), vmax=clip)
    _style_ax(ax2, "Point Cloud (depth-coloured)")

    cbar = fig.colorbar(sc, ax=ax2, pad=0.08, shrink=0.55, orientation="vertical")
    cbar.ax.yaxis.set_tick_params(color="#aaa", labelsize=7)
    cbar.set_label("Depth (m)", color="#aaa", fontsize=8)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#aaa")

    plt.suptitle(
        f"DAP – Panoramic 3-D Point Cloud  (frame {frame_idx + 1:02d} | "
        f"{n_pts:,} pts | subsample={subsample} | far-clip={clip:.1f} m)",
        color="white", fontsize=12, y=0.98,
    )
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, f"pointcloud_frame{frame_idx + 1:02d}.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"   Saved: {out_path}")


# ── 3. depth statistics summary ────────────────────────────────────────────

def make_stats_plot():
    npy_files = sorted_files(NPY_DIR, ".npy")
    print(f"\n[3/3] Building depth statistics plot ...")

    means, medians, p10s, p90s = [], [], [], []
    for nf in npy_files:
        d = np.load(os.path.join(NPY_DIR, nf)).ravel() * MAX_DEPTH_M
        means.append(d.mean())
        medians.append(np.median(d))
        p10s.append(np.percentile(d, 10))
        p90s.append(np.percentile(d, 90))

    frames = np.arange(1, len(npy_files) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#111")
    for ax in axes:
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="#ccc")
        ax.xaxis.label.set_color("#ccc")
        ax.yaxis.label.set_color("#ccc")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    # ── left: per-frame stats ──
    ax = axes[0]
    ax.fill_between(frames, p10s, p90s, alpha=0.25, color="#4fc3f7",
                    label="P10–P90")
    ax.plot(frames, medians, color="#4fc3f7", lw=1.8, label="Median")
    ax.plot(frames, means,   color="#ff8a65", lw=1.5, ls="--", label="Mean")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Per-frame Depth Statistics")
    ax.legend(facecolor="#222", edgecolor="#444", labelcolor="white", fontsize=9)
    ax.grid(True, color="#2a2a2a", linewidth=0.6)

    # ── right: histogram of all depth values ──
    ax = axes[1]
    all_depths = np.concatenate([
        np.load(os.path.join(NPY_DIR, nf)).ravel() * MAX_DEPTH_M
        for nf in npy_files
    ])
    ax.hist(all_depths, bins=120, color="#4fc3f7", alpha=0.85, edgecolor="none")
    ax.set_xlabel("Depth (m)")
    ax.set_ylabel("Pixel count")
    ax.set_title("Depth Distribution (all frames)")
    ax.grid(True, color="#2a2a2a", linewidth=0.6)
    ax.set_yscale("log")

    plt.suptitle("DAP – Depth Estimation Statistics  |  video_1000050115",
                 color="white", fontsize=13, y=1.01)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "depth_stats.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"   Saved: {out_path}")


# ── entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAP Visualization")
    parser.add_argument("--frame",     type=int,   default=14,
                        help="0-based frame index for point cloud (default: 14)")
    parser.add_argument("--subsample", type=int,   default=3,
                        help="Point cloud pixel subsampling factor (default: 3; 2=denser, 1=full)")
    parser.add_argument("--max-depth", type=float, default=None,
                        help="Far-clip in metres for point cloud (default: auto P99). "
                             "Use e.g. 10 for indoor scenes.")
    parser.add_argument("--fps",       type=int,   default=6,
                        help="FPS for comparison video (default: 6)")
    parser.add_argument("--skip-video", action="store_true",
                        help="Skip video generation")
    args = parser.parse_args()

    if not args.skip_video:
        make_comparison_video(fps=args.fps, max_depth_m=args.max_depth)
    make_pointcloud(frame_idx=args.frame, subsample=args.subsample,
                    max_depth_m=args.max_depth)
    make_stats_plot()

    print(f"\nAll outputs saved to: {OUT_DIR}")
