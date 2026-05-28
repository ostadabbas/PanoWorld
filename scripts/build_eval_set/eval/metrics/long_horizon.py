"""Long-horizon metrics. Currently DEFERRED — we focus on traj-PSNR + APD/ATE.

When re-enabled, this module will provide:

* loop_consistency  — OmniRoam-style CLIP-similarity to first frame on closed-
  loop trajectories (requires Habitat re-render with explicit loop subset, or
  Argus clips that already contain loops).
* nvs_psnr_window   — PSNR over far-out frame windows (e.g. 615-635 from
  OmniRoam) for sequences > 100 frames.

For the current 93-frame setup the per-window PSNR in trajectory.py already
covers what we need, so this module returns an empty dict.
"""

from __future__ import annotations


def eval_long_horizon(pred_path: str, gt_path: str, **ctx) -> dict[str, float]:
    return {}
