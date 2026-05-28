"""Smoke tests for metrics/_common.py spatial+temporal align logic.

Run directly:
    python -m metrics.test_align
or with pytest if available:
    pytest test_set_pkg/eval/metrics/test_align.py
"""
from __future__ import annotations

import numpy as np

from _common import (
    align_pair_to_eval_grid,
    align_to_eval_grid,
    resize_thwc,
)


def _make_video(T: int, H: int, W: int, *, color: int = 0) -> np.ndarray:
    arr = np.full((T, H, W, 3), color, dtype=np.uint8)
    # diagonal stripe so we can detect resize artefacts visually
    for t in range(T):
        for i in range(min(H, W)):
            arr[t, i, i, :] = 255
    return arr


def test_resize_noop_when_match():
    """resize_thwc should be a no-op (identity) if H,W already match target."""
    v = _make_video(4, 512, 1024, color=42)
    out = resize_thwc(v, 512, 1024)
    assert out.shape == v.shape, f"shape mismatch: {out.shape} vs {v.shape}"
    assert np.array_equal(out, v), "no-op resize should be bit-identical"
    print("[PASS] resize_thwc is no-op when shape already matches")


def test_resize_upscale_omniroam_case():
    """960x480 -> 1024x512 (omniroam_pers vs GT). Must preserve T, dtype, range."""
    v = _make_video(8, 480, 960, color=128)
    out = resize_thwc(v, 512, 1024)
    assert out.shape == (8, 512, 1024, 3), f"got {out.shape}"
    assert out.dtype == np.uint8, f"got {out.dtype}"
    # Center region should still hold the constant fill (resize artefacts only at diagonal).
    assert int(out[3, 256, 512, 0]) >= 100, "center pixel value should remain near fill (128)"
    print("[PASS] resize_thwc upscales 960x480 -> 1024x512 correctly")


def test_align_pair_matching_shape_is_unchanged():
    """When pred and gt share resolution, align_pair_to_eval_grid result is
    bit-identical for the spatial dim (only temporal axis may change)."""
    pred = _make_video(80, 512, 1024, color=10)
    gt   = _make_video(80, 512, 1024, color=20)
    pred_a, gt_a = align_pair_to_eval_grid(pred, gt, pred_fps=16.0, gt_fps=16.0)
    assert pred_a.shape == (80, 512, 1024, 3)
    assert gt_a.shape   == (80, 512, 1024, 3)
    assert np.array_equal(pred_a, pred), "spatially identical pred should be untouched"
    assert np.array_equal(gt_a,   gt),   "spatially identical gt should be untouched"
    print("[PASS] align_pair_to_eval_grid is no-op spatial when shapes match")


def test_align_pair_omniroam_resizes_pred_to_gt():
    """omniroam case: pred=960x480, gt=1024x512. After fix:
    - pred is resized UP to 1024x512 (matching GT)
    - gt is left at 1024x512 (no crop)
    Both outputs are 1024x512.
    """
    pred = _make_video(81, 480, 960, color=50)
    gt   = _make_video(80, 512, 1024, color=200)
    pred_a, gt_a = align_pair_to_eval_grid(pred, gt, pred_fps=16.0, gt_fps=16.0)
    assert pred_a.shape == (80, 512, 1024, 3), f"pred should be 1024x512 after resize, got {pred_a.shape}"
    assert gt_a.shape   == (80, 512, 1024, 3), f"gt should stay 1024x512, got {gt_a.shape}"
    # GT center pixel must remain the original fill (no crop happened).
    assert int(gt_a[40, 256, 512, 0]) == 200, "GT center should still be 200 (no crop)"
    print("[PASS] align_pair resizes omniroam 960x480 -> GT 1024x512, no crop on GT")


def test_align_pair_explicit_target_hw():
    """Explicit target_hw overrides GT-shape default."""
    pred = _make_video(80, 480, 960, color=50)
    gt   = _make_video(80, 512, 1024, color=200)
    pred_a, gt_a = align_pair_to_eval_grid(
        pred, gt, pred_fps=16.0, gt_fps=16.0,
        target_hw=(256, 512),
    )
    assert pred_a.shape == (80, 256, 512, 3)
    assert gt_a.shape   == (80, 256, 512, 3)
    print("[PASS] target_hw=(256,512) override works")


def test_align_to_eval_grid_temporal_only():
    """Single video temporal align is unchanged (no spatial work)."""
    v = _make_video(81, 480, 960, color=77)
    out = align_to_eval_grid(v, src_fps=16.0)
    # 81 frames @ 16fps = 5.0625s ; eval grid 5.0s @ 16fps = 80f.
    assert out.shape == (80, 480, 960, 3), f"got {out.shape}"
    print("[PASS] align_to_eval_grid is spatially untouched (single-video path)")


def main():
    test_resize_noop_when_match()
    test_resize_upscale_omniroam_case()
    test_align_pair_matching_shape_is_unchanged()
    test_align_pair_omniroam_resizes_pred_to_gt()
    test_align_pair_explicit_target_hw()
    test_align_to_eval_grid_temporal_only()
    print("\nAll align tests passed.")


if __name__ == "__main__":
    main()
