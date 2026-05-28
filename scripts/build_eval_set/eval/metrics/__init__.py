"""Metric registry.

Every metric is a callable
    metric(pred_path, gt_path, **ctx) -> dict[metric_name, float | NaN]

`pred_path` is a method's output (mp4 / npy / dir of npys depending on the
metric). `gt_path` is the master.csv-derived ground-truth pointer for the same
clip. `ctx` carries strata info, n_frames, fps, etc.

Each function returns a *dict* because most metrics produce multiple variants
(raw / sky-masked / texture-weighted, or per-window PSNR).
"""

from importlib import import_module

REGISTRY = {
    "visual": "metrics.visual:eval_visual",
    "depth": "metrics.depth:eval_depth",
    "trajectory": "metrics.trajectory:eval_trajectory",
    "tracks": "metrics.tracks:eval_tracks",
    "long_horizon": "metrics.long_horizon:eval_long_horizon",
    "caption_alignment": "metrics.caption_alignment:eval_caption_alignment",
    "self_consistency": "metrics.self_consistency:eval_self_consistency",
}


def get(name: str):
    if name not in REGISTRY:
        raise KeyError(f"unknown metric group: {name}; have {list(REGISTRY)}")
    mod_path, fn_name = REGISTRY[name].split(":")
    return getattr(import_module(mod_path), fn_name)
