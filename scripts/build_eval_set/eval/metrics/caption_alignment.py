"""Caption alignment metrics — placeholder; primary metric (CLIP-T per frame)
already lives in metrics/visual.py:eval_visual.

This module will host *additional* caption-side metrics later, e.g.

* CIDEr / SPICE (only if we caption the generated videos with the same VLM
  and compare to GT captions — not currently planned).
* DSGTextScore (semantic graph match).

For now it returns {} so the registry stays consistent.
"""

from __future__ import annotations


def eval_caption_alignment(pred_path: str, gt_path: str, **ctx) -> dict[str, float]:
    return {}
