"""
Extension Quality Evaluation
=============================
Three-part evaluation for panoramic video temporal extension:

  1. Concatenate round videos into one seamless clip
  2. Depth temporal stability analysis (per-round + global)
  3. Person motion trajectory tracking (CoTracker3 or manual mask)

Usage:
  # Full evaluation on an output directory
  python eval_extension.py --output_dir output/v5_geo_lobby_latent_v2 \
      --rounds 4 --eval_depth --eval_motion

  # Just concatenate
  python eval_extension.py --output_dir output/v5_geo_lobby_latent_v2 \
      --rounds 4 --concat_only

  # Depth evaluation only (skip concat if already done)
  python eval_extension.py --output_dir output/v5_geo_lobby_latent_v2 \
      --rounds 4 --eval_depth
"""

import os
import sys
import json
import argparse
import time
import glob

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.io as tio
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ═══════════════════════════════════════════════════════════════
#  Part 1: Video Concatenation
# ═══════════════════════════════════════════════════════════════

def find_round_videos(output_dir, max_rounds):
    """Find round video files in the output directory."""
    pattern_base = os.path.join(output_dir, "pano_v3_*_s*.mp4")
    candidates = sorted(glob.glob(pattern_base))

    round1 = None
    round_files = {}
    for c in candidates:
        bn = os.path.basename(c)
        if "_round" in bn:
            try:
                rn = int(bn.split("_round")[1].split(".")[0])
                round_files[rn] = c
            except ValueError:
                pass
        elif "round" not in bn and bn.startswith("pano_v3"):
            round1 = c

    if round1 is None:
        raise FileNotFoundError(f"No round 1 video found in {output_dir}")

    result = {1: round1}
    for r in range(2, max_rounds + 1):
        if r in round_files:
            result[r] = round_files[r]
    return result


def compute_overlap_frames(vid_a, vid_b, max_overlap=8):
    """
    Estimate the number of overlapping frames between the end of vid_a
    and the start of vid_b by finding the minimum MSE alignment.
    Returns (best_overlap, mse_curve).
    """
    T_a = vid_a.shape[0]
    T_b = vid_b.shape[0]
    check = min(max_overlap, T_a, T_b)

    mses = []
    for k in range(1, check + 1):
        tail = vid_a[-k:].float()
        head = vid_b[:k].float()
        mse = ((tail - head) ** 2).mean().item()
        mses.append(mse)

    if not mses:
        return 0, []
    best_k = int(np.argmin(mses)) + 1
    best_mse = mses[best_k - 1]

    if best_mse > 500:
        return 0, mses
    return best_k, mses


def concatenate_rounds(round_videos, output_path, overlap_mode="auto"):
    """
    Concatenate round videos into one seamless clip.
    overlap_mode: 'auto' (detect), 'none' (no overlap removal), or int (fixed frames to skip)
    """
    videos = []
    for r in sorted(round_videos.keys()):
        v, _, info = tio.read_video(round_videos[r])
        videos.append(v)
        fps = info["video_fps"]

    frames_list = [videos[0]]
    overlap_info = []

    for i in range(1, len(videos)):
        if overlap_mode == "auto":
            overlap, mses = compute_overlap_frames(videos[i - 1], videos[i])
        elif overlap_mode == "none":
            overlap = 0
            mses = []
        else:
            overlap = int(overlap_mode)
            mses = []

        overlap_info.append({
            "transition": f"round{i}→round{i+1}",
            "overlap_frames": overlap,
            "mse_curve": [float(m) for m in mses] if mses else [],
        })
        frames_list.append(videos[i][overlap:])

    concat = torch.cat(frames_list, dim=0)
    print(f"Concatenated: {concat.shape[0]} frames ({concat.shape[1]}x{concat.shape[2]})")

    tio.write_video(output_path, concat, fps=int(fps))
    print(f"Saved: {output_path}")

    info_path = output_path.replace(".mp4", "_info.json")
    info = {
        "total_frames": int(concat.shape[0]),
        "fps": float(fps),
        "duration_s": float(concat.shape[0] / fps),
        "rounds": len(round_videos),
        "per_round_frames": [int(v.shape[0]) for v in videos],
        "overlaps": overlap_info,
    }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    return concat, fps, info


# ═══════════════════════════════════════════════════════════════
#  Part 2: Depth Stability Analysis
# ═══════════════════════════════════════════════════════════════

def load_dap_model():
    """Load DAP depth model. Must chdir to DAP root for relative paths."""
    DAP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DAP")
    sys.path.insert(0, DAP_ROOT)
    sys.path.insert(0, os.path.join(DAP_ROOT, "test"))

    import yaml
    from infer import load_model

    prev_cwd = os.getcwd()
    os.chdir(DAP_ROOT)

    config_path = os.path.join(DAP_ROOT, "config", "infer.yaml")
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model, device = load_model(config)

    os.chdir(prev_cwd)
    return model, device


def estimate_depth_batch(model, device, frames_np):
    """
    Run DAP depth on a batch of frames.
    frames_np: list of (H, W, 3) uint8 RGB arrays
    Returns: list of (H, W) float32 depth maps
    """
    from infer import infer_raw
    depths = []
    for frame in frames_np:
        d = infer_raw(model, device, frame)
        depths.append(d)
    return depths


def analyze_depth_stability(depths, round_boundaries, output_dir):
    """
    Analyze temporal depth stability.

    Metrics computed per-frame:
      - mean_depth: average depth value
      - temporal_diff: L1 diff between consecutive depth maps
      - structural_sim: SSIM-like structural metric between consecutive frames

    Outputs:
      - Per-frame metrics CSV
      - Stability plots (per-round colored)
      - Summary statistics JSON
    """
    os.makedirs(output_dir, exist_ok=True)
    n = len(depths)

    mean_depths = []
    std_depths = []
    temporal_diffs = []
    temporal_rel_diffs = []

    for i in range(n):
        d = depths[i]
        valid = d[d > 0.01]
        mean_depths.append(float(np.mean(valid)) if len(valid) > 0 else 0)
        std_depths.append(float(np.std(valid)) if len(valid) > 0 else 0)

        if i > 0:
            diff = np.abs(depths[i] - depths[i - 1])
            temporal_diffs.append(float(np.mean(diff)))
            denom = np.maximum(np.abs(depths[i - 1]), 0.01)
            temporal_rel_diffs.append(float(np.mean(diff / denom)))
        else:
            temporal_diffs.append(0.0)
            temporal_rel_diffs.append(0.0)

    # --- Plot ---
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 1, hspace=0.35)

    colors = plt.cm.tab10(np.linspace(0, 1, len(round_boundaries)))

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    for ri, (start, end) in enumerate(round_boundaries):
        c = colors[ri]
        label = f"Round {ri + 1}"
        frames = range(start, min(end, n))
        ax1.plot(frames, mean_depths[start:end], color=c, label=label, linewidth=1.2)
        ax2.plot(frames, temporal_diffs[start:end], color=c, label=label, linewidth=1.2)
        ax3.plot(frames, temporal_rel_diffs[start:end], color=c, label=label, linewidth=1.2)

    for start, _ in round_boundaries[1:]:
        for ax in [ax1, ax2, ax3]:
            ax.axvline(x=start, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    ax1.set_ylabel("Mean Depth")
    ax1.set_title("Depth Stability Analysis — Mean Depth per Frame")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("Temporal L1 Diff")
    ax2.set_title("Frame-to-Frame Depth Change (Absolute)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3.set_ylabel("Relative Depth Change")
    ax3.set_title("Frame-to-Frame Depth Change (Relative)")
    ax3.set_xlabel("Frame Index")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3)

    plot_path = os.path.join(output_dir, "depth_stability.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Depth stability plot: {plot_path}")

    per_round_stats = []
    for ri, (start, end) in enumerate(round_boundaries):
        end = min(end, n)
        diffs_r = temporal_diffs[start + 1:end]
        rel_r = temporal_rel_diffs[start + 1:end]
        per_round_stats.append({
            "round": ri + 1,
            "frames": end - start,
            "mean_depth_avg": float(np.mean(mean_depths[start:end])),
            "mean_depth_std": float(np.std(mean_depths[start:end])),
            "temporal_l1_avg": float(np.mean(diffs_r)) if diffs_r else 0,
            "temporal_l1_max": float(np.max(diffs_r)) if diffs_r else 0,
            "temporal_rel_avg": float(np.mean(rel_r)) if rel_r else 0,
            "temporal_rel_max": float(np.max(rel_r)) if rel_r else 0,
        })

    summary = {
        "total_frames": n,
        "per_round": per_round_stats,
        "global_mean_depth": float(np.mean(mean_depths)),
        "global_temporal_l1_avg": float(np.mean(temporal_diffs[1:])),
        "degradation_trend": {
            "round1_l1": per_round_stats[0]["temporal_l1_avg"] if per_round_stats else 0,
            "last_round_l1": per_round_stats[-1]["temporal_l1_avg"] if per_round_stats else 0,
            "ratio": (per_round_stats[-1]["temporal_l1_avg"] /
                      max(per_round_stats[0]["temporal_l1_avg"], 1e-8))
                     if len(per_round_stats) > 1 else 1.0,
        },
    }

    stats_path = os.path.join(output_dir, "depth_stability_stats.json")
    with open(stats_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Depth stats: {stats_path}")

    np.savez_compressed(
        os.path.join(output_dir, "depth_metrics.npz"),
        mean_depths=np.array(mean_depths),
        temporal_diffs=np.array(temporal_diffs),
        temporal_rel_diffs=np.array(temporal_rel_diffs),
        std_depths=np.array(std_depths),
    )

    return summary


def visualize_depth_comparison(depths, round_boundaries, output_dir, sample_frames=None):
    """Save a side-by-side depth comparison for selected frames across rounds."""
    os.makedirs(output_dir, exist_ok=True)

    if sample_frames is None:
        sample_frames = []
        for ri, (start, end) in enumerate(round_boundaries):
            mid = (start + end) // 2
            sample_frames.append(mid)

    n_samples = len(sample_frames)
    fig, axes = plt.subplots(2, n_samples, figsize=(4 * n_samples, 6))
    if n_samples == 1:
        axes = axes[:, np.newaxis]

    vmin = min(d.min() for d in depths)
    vmax = np.percentile(np.concatenate([d.flatten() for d in depths]), 95)

    for i, fi in enumerate(sample_frames):
        if fi >= len(depths):
            continue
        d = depths[fi]
        ri = next(r for r, (s, e) in enumerate(round_boundaries) if s <= fi < e)

        axes[0, i].imshow(d, cmap="Spectral", vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f"R{ri+1} F{fi}", fontsize=9)
        axes[0, i].axis("off")

        if fi > 0:
            diff = np.abs(depths[fi] - depths[fi - 1])
            axes[1, i].imshow(diff, cmap="hot", vmin=0, vmax=vmax * 0.3)
            axes[1, i].set_title(f"Δ(F{fi}-F{fi-1})", fontsize=9)
        else:
            axes[1, i].set_title("(first frame)", fontsize=9)
        axes[1, i].axis("off")

    plt.suptitle("Depth Maps & Temporal Differences", fontsize=12)
    comp_path = os.path.join(output_dir, "depth_comparison.png")
    plt.savefig(comp_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Depth comparison: {comp_path}")


# ═══════════════════════════════════════════════════════════════
#  Part 3: Person Motion Trajectory
# ═══════════════════════════════════════════════════════════════

def track_with_cotracker(video_tensor, grid_size=20, output_dir=None):
    """
    Run CoTracker3 on the video to get dense tracks.
    video_tensor: (T, H, W, 3) uint8 tensor
    Returns: pred_tracks (1, T, N, 2), pred_visibility (1, T, N)
    """
    COTRACKER_ROOT = os.environ.get("COTRACKER_ROOT", "")
    if COTRACKER_ROOT:
        sys.path.insert(0, COTRACKER_ROOT)
    from cotracker.predictor import CoTrackerPredictor

    cotracker_ckpt = os.environ.get(
        "COTRACKER_CKPT",
        os.path.expanduser("~/.cache/torch/hub/checkpoints/scaled_offline.pth"),
    )
    device = torch.device("cuda")
    model = CoTrackerPredictor(checkpoint=cotracker_ckpt).to(device)

    video = video_tensor.permute(0, 3, 1, 2).unsqueeze(0).float().to(device)

    pred_tracks, pred_visibility = model(video, grid_size=grid_size)
    return pred_tracks.cpu(), pred_visibility.cpu()


def analyze_motion_from_tracks(pred_tracks, pred_visibility, round_boundaries,
                               output_dir, H, W, motion_threshold=3.0):
    """
    Analyze tracked point motions. Identifies high-motion points (likely the person)
    and plots their trajectories.
    """
    os.makedirs(output_dir, exist_ok=True)

    tracks = pred_tracks[0].numpy()   # (T, N, 2)
    vis = pred_visibility[0].numpy()  # (T, N)
    T, N, _ = tracks.shape

    displacement = np.zeros(N)
    for t in range(1, T):
        mask = vis[t] * vis[t - 1]
        d = np.linalg.norm(tracks[t] - tracks[t - 1], axis=-1) * mask
        displacement += d

    avg_disp = displacement / max(T - 1, 1)

    high_motion = avg_disp > motion_threshold
    n_high = high_motion.sum()
    print(f"High-motion points (>{motion_threshold} px/frame): {n_high}/{N}")

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    colors_round = plt.cm.tab10(np.linspace(0, 1, len(round_boundaries)))

    ax = axes[0]
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.set_title(f"All Tracks (N={N}, high-motion={n_high})")

    for i in range(N):
        alpha = 0.6 if high_motion[i] else 0.1
        lw = 1.5 if high_motion[i] else 0.3
        color = "red" if high_motion[i] else "blue"
        valid = vis[:, i] > 0.5
        ax.plot(tracks[valid, i, 0], tracks[valid, i, 1],
                color=color, alpha=alpha, linewidth=lw)

    ax = axes[1]
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.set_title("High-Motion Tracks (colored by round)")

    for i in range(N):
        if not high_motion[i]:
            continue
        for ri, (start, end) in enumerate(round_boundaries):
            seg = tracks[start:min(end, T), i]
            seg_vis = vis[start:min(end, T), i]
            valid = seg_vis > 0.5
            if valid.sum() > 1:
                ax.plot(seg[valid, 0], seg[valid, 1],
                        color=colors_round[ri], linewidth=1.5, alpha=0.8,
                        label=f"R{ri+1}" if i == np.where(high_motion)[0][0] else "")

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8)

    track_path = os.path.join(output_dir, "motion_tracks.png")
    plt.savefig(track_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Motion tracks plot: {track_path}")

    per_round_motion = []
    for ri, (start, end) in enumerate(round_boundaries):
        end = min(end, T)
        round_disp = []
        for t in range(start + 1, end):
            mask = vis[t] * vis[t - 1] * high_motion
            if mask.sum() > 0:
                d = np.linalg.norm(tracks[t] - tracks[t - 1], axis=-1)
                round_disp.append(float((d * mask).sum() / mask.sum()))
        per_round_motion.append({
            "round": ri + 1,
            "avg_motion_px_per_frame": float(np.mean(round_disp)) if round_disp else 0,
            "max_motion_px_per_frame": float(np.max(round_disp)) if round_disp else 0,
        })

    fig, ax = plt.subplots(figsize=(10, 4))
    frame_motions = []
    for t in range(1, T):
        mask = vis[t] * vis[t - 1] * high_motion
        if mask.sum() > 0:
            d = np.linalg.norm(tracks[t] - tracks[t - 1], axis=-1)
            frame_motions.append(float((d * mask).sum() / mask.sum()))
        else:
            frame_motions.append(0)

    for ri, (start, end) in enumerate(round_boundaries):
        end = min(end, T)
        seg_start = max(start - 1, 0)
        seg_end = min(end - 1, len(frame_motions))
        ax.plot(range(seg_start + 1, seg_end + 1), frame_motions[seg_start:seg_end],
                color=colors_round[ri], linewidth=1.2, label=f"Round {ri+1}")

    for start, _ in round_boundaries[1:]:
        ax.axvline(x=start, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("Frame")
    ax.set_ylabel("Avg Motion (px/frame)")
    ax.set_title("Person Motion Speed Over Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    motion_plot_path = os.path.join(output_dir, "motion_speed.png")
    plt.savefig(motion_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Motion speed plot: {motion_plot_path}")

    motion_summary = {
        "total_tracks": int(N),
        "high_motion_tracks": int(n_high),
        "motion_threshold_px": float(motion_threshold),
        "per_round": per_round_motion,
    }
    motion_stats_path = os.path.join(output_dir, "motion_stats.json")
    with open(motion_stats_path, "w") as f:
        json.dump(motion_summary, f, indent=2)
    print(f"Motion stats: {motion_stats_path}")

    return motion_summary


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate panoramic video extension quality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output_dir", required=True,
                        help="Directory containing round MP4 files")
    parser.add_argument("--rounds", type=int, default=4,
                        help="Number of rounds to include (1=initial only)")
    parser.add_argument("--concat_only", action="store_true",
                        help="Only concatenate, skip all evaluation")
    parser.add_argument("--eval_depth", action="store_true",
                        help="Run depth stability analysis")
    parser.add_argument("--eval_motion", action="store_true",
                        help="Run person motion trajectory analysis")
    parser.add_argument("--overlap_mode", default="auto",
                        help="Overlap mode: 'auto', 'none', or integer")
    parser.add_argument("--cotracker_grid", type=int, default=30,
                        help="CoTracker grid size for motion tracking")
    parser.add_argument("--motion_threshold", type=float, default=3.0,
                        help="Pixels/frame threshold for high-motion points")
    args = parser.parse_args()

    eval_dir = os.path.join(args.output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # --- Step 1: Find and concatenate ---
    print("=" * 60)
    print(f"  Extension Quality Evaluation")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Rounds: {args.rounds}")
    print("=" * 60)

    round_videos = find_round_videos(args.output_dir, args.rounds)
    print(f"\nFound {len(round_videos)} round videos:")
    for r, p in sorted(round_videos.items()):
        print(f"  Round {r}: {os.path.basename(p)}")

    concat_path = os.path.join(eval_dir, "concatenated.mp4")
    concat_video, fps, concat_info = concatenate_rounds(
        round_videos, concat_path,
        overlap_mode=args.overlap_mode,
    )

    if args.concat_only:
        print("\n[concat_only] Done.")
        return

    T_total = concat_video.shape[0]
    round_boundaries = []
    offset = 0
    for ri, nf in enumerate(concat_info["per_round_frames"]):
        if ri > 0 and ri - 1 < len(concat_info["overlaps"]):
            overlap = concat_info["overlaps"][ri - 1]["overlap_frames"]
        else:
            overlap = 0
        actual = nf - (overlap if ri > 0 else 0)
        round_boundaries.append((offset, offset + actual))
        offset += actual

    print(f"\nRound boundaries (frame indices): {round_boundaries}")

    # --- Step 2: Depth stability ---
    if args.eval_depth:
        print("\n" + "=" * 60)
        print("  Depth Stability Analysis")
        print("=" * 60)

        depth_dir = os.path.join(eval_dir, "depth")
        os.makedirs(depth_dir, exist_ok=True)

        cache_path = os.path.join(depth_dir, "depths_cache.npz")
        if os.path.exists(cache_path):
            print("Loading cached depth maps...")
            data = np.load(cache_path)
            depths = [data[f"frame_{i}"] for i in range(len(data.files))]
        else:
            print("Loading DAP model...")
            model, device = load_dap_model()

            print(f"Running depth estimation on {T_total} frames...")
            frames_np = [concat_video[t].numpy() for t in range(T_total)]
            from tqdm import tqdm
            depths = []
            for i in tqdm(range(T_total), desc="DAP inference"):
                frame_rgb = frames_np[i]
                from infer import infer_raw
                d = infer_raw(model, device, frame_rgb)
                depths.append(d)

            save_dict = {f"frame_{i}": depths[i] for i in range(len(depths))}
            np.savez_compressed(cache_path, **save_dict)
            print(f"Depth maps cached: {cache_path}")

            del model
            torch.cuda.empty_cache()

        depth_summary = analyze_depth_stability(depths, round_boundaries, depth_dir)
        visualize_depth_comparison(depths, round_boundaries, depth_dir)

        print("\n--- Depth Stability Summary ---")
        for rs in depth_summary["per_round"]:
            print(f"  Round {rs['round']}: mean_depth={rs['mean_depth_avg']:.3f} "
                  f"temporal_L1={rs['temporal_l1_avg']:.4f} "
                  f"temporal_rel={rs['temporal_rel_avg']:.4f}")
        dt = depth_summary["degradation_trend"]
        print(f"  Degradation ratio (last/first): {dt['ratio']:.2f}x")

    # --- Step 3: Motion trajectory ---
    if args.eval_motion:
        print("\n" + "=" * 60)
        print("  Person Motion Trajectory Analysis")
        print("=" * 60)

        motion_dir = os.path.join(eval_dir, "motion")

        cotracker_ckpt = os.environ.get(
            "COTRACKER_CKPT",
            os.path.expanduser("~/.cache/torch/hub/checkpoints/scaled_offline.pth"),
        )
        if not os.path.exists(cotracker_ckpt):
            print(f"WARNING: CoTracker checkpoint not found at {cotracker_ckpt}")
            print("Skipping motion analysis. Please provide the checkpoint or use manual masks.")
        else:
            print(f"Running CoTracker3 (grid_size={args.cotracker_grid})...")
            pred_tracks, pred_vis = track_with_cotracker(
                concat_video, grid_size=args.cotracker_grid, output_dir=motion_dir
            )

            H, W = concat_video.shape[1], concat_video.shape[2]
            motion_summary = analyze_motion_from_tracks(
                pred_tracks, pred_vis, round_boundaries, motion_dir,
                H, W, motion_threshold=args.motion_threshold,
            )

            print("\n--- Motion Summary ---")
            for ms in motion_summary["per_round"]:
                print(f"  Round {ms['round']}: "
                      f"avg_motion={ms['avg_motion_px_per_frame']:.2f} px/frame "
                      f"max_motion={ms['max_motion_px_per_frame']:.2f} px/frame")

    print("\n" + "=" * 60)
    print("  Evaluation complete!")
    print(f"  Results in: {eval_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
