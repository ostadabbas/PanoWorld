"""Shared helpers for inference runners.

Conventions:
  master.csv  -> $PANO_DATA_ROOT/test/master.csv
  per-method results layout:
      results_dir/<method_id>/<clip_id>/video.mp4
      results_dir/<method_id>/<clip_id>/depth/0000.npy ...   (optional)
      results_dir/<method_id>/<clip_id>/camera_poses.npy     (optional)
      results_dir/<method_id>/<clip_id>/run.json             (provenance)

clip_id format in master.csv is "<split>::<clip_basename>" (e.g.
"habitat_ood::hab_0017"). We replace "::" with "__" for filesystem-safe
output dirs.
"""
from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterator

import numpy as np


# ───────────────────────────── master.csv ─────────────────────────────
_PANO_DATA_ROOT = os.environ.get(
    "PANO_DATA_ROOT", os.path.join(os.path.expanduser("~"), "pano_video_data")
)
MASTER_DEFAULT = Path(_PANO_DATA_ROOT) / "test" / "master.csv"


def read_master(path: Path | str = MASTER_DEFAULT) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def clip_id_to_dirname(clip_id: str) -> str:
    """`split::clip` -> `split__clip` (filesystem-safe)."""
    return clip_id.replace("::", "__")


# ───────────────────────────── caption ────────────────────────────────
def load_caption_with_style(caption_path: str | None,
                            style: str = "long") -> tuple[str, str]:
    """Read a caption JSON and return (text, style_used).

    Caption files are typically `{model_key: {short, medium, long}}` (e.g.
    gpt-5-mini → {short, medium, long}); we also handle the flat
    `{short, medium, long}` shape.

    Fallback order: requested `style` -> long -> medium -> short.

    `style_used` is one of {"long","medium","short"} on hit, and "" when
    the file is missing/unparseable/empty (in which case `text` is also "").
    Runners persist `style_used` to run.json so the eval pipeline / paper
    appendix can audit which caption text actually drove each method.
    """
    if not caption_path or not Path(caption_path).is_file():
        return "", ""
    try:
        d = json.loads(Path(caption_path).read_text())
    except Exception:
        return "", ""

    candidates: list[dict] = []
    if isinstance(d, dict):
        candidates.append(d)
        candidates.extend(v for v in d.values() if isinstance(v, dict))
    elif isinstance(d, list) and d and isinstance(d[0], dict):
        candidates.append(d[0])

    fallbacks = [s for s in (style, "long", "medium", "short")]
    seen = set()
    fallbacks = [s for s in fallbacks if not (s in seen or seen.add(s))]

    for cand in candidates:
        for s in fallbacks:
            v = cand.get(s)
            if isinstance(v, str) and v.strip():
                return v.strip(), s
            if isinstance(v, list) and v and isinstance(v[0], str) and v[0].strip():
                return v[0].strip(), s
    return "", ""


def load_caption_text(caption_path: str | None,
                      style: str = "long") -> str:
    """Backward-compat wrapper. Prefer `load_caption_with_style` in new code."""
    return load_caption_with_style(caption_path, style)[0]


# ───────────────────────────── frame I/O ──────────────────────────────
def read_first_frame_rgb_uint8(video_path: str) -> np.ndarray:
    """Return frame 0 of an mp4 as (H, W, 3) uint8."""
    from decord import VideoReader, cpu
    vr = VideoReader(video_path, ctx=cpu(0))
    arr = vr[0].asnumpy()  # (H, W, 3)
    return arr.astype(np.uint8)


def save_image_png(arr_hwc_uint8: np.ndarray, out_path: str | Path) -> str:
    """Save (H, W, 3) uint8 -> PNG."""
    from PIL import Image
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_hwc_uint8).save(out_path, format="PNG")
    return out_path


def erp_to_perspective_image(
    erp_uint8_hwc: np.ndarray,
    fov_x_deg: float = 90.0,
    yaw_rad: float = 0.0,
    pitch_rad: float = 0.0,
    out_h: int = 480,
    out_w: int = 640,
    device: str = "cuda",
) -> np.ndarray:
    """Render a perspective crop from a single ERP frame using cosmos's equi2pers.

    Returns (out_h, out_w, 3) uint8. Falls back to CPU if CUDA unavailable.
    """
    import torch
    from cosmos_predict2._src.predict2.utils.pano_conditioning import equi2pers
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    frame = (
        torch.from_numpy(erp_uint8_hwc).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    )
    yaw   = torch.tensor([yaw_rad], device=device)
    pitch = torch.tensor([pitch_rad], device=device)
    roll  = torch.zeros(1, device=device)
    pers = equi2pers(
        frame, fov_x=fov_x_deg, roll=roll, pitch=pitch, yaw=yaw,
        out_h=out_h, out_w=out_w,
    )  # (1, 3, H, W)
    arr = (pers[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return arr


def make_static_video_from_image(
    image_path_or_arr,
    out_mp4: str | Path,
    n_frames: int = 25,
    fps: int = 25,
) -> str:
    """Replicate one image into an mp4 of `n_frames`. Used for the "static-
    repeat video" parity strategy when a baseline insists on video input.

    Accepts either a path (PNG/JPG) or a (H,W,3) uint8 ndarray.
    """
    import imageio.v3 as iio
    from PIL import Image
    out_mp4 = str(out_mp4)
    Path(out_mp4).parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image_path_or_arr, np.ndarray):
        arr = image_path_or_arr
    else:
        arr = np.asarray(Image.open(str(image_path_or_arr)).convert("RGB"))
    frames = np.stack([arr] * int(n_frames), axis=0)
    iio.imwrite(out_mp4, frames, fps=fps, codec="libx264",
                pixelformat="yuv420p", macro_block_size=1)
    return out_mp4


# ───────────────────────────── output layout ──────────────────────────
def per_clip_outdir(results_dir: str | Path, method_id: str, clip_id: str) -> Path:
    return Path(results_dir) / method_id / clip_id_to_dirname(clip_id)


def write_run_json(out_dir: Path, payload: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "run.json"
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return p


def have_output(out_dir: Path) -> bool:
    """A clip is considered done if video.mp4 exists and is non-empty."""
    v = out_dir / "video.mp4"
    return v.is_file() and v.stat().st_size > 0


# ───────────────────────────── subprocess ─────────────────────────────
def run_subprocess(
    cmd: list[str],
    log_path: Path | str,
    cwd: str | Path | None = None,
    env_extra: dict | None = None,
) -> int:
    """Run a command, tee'ing stdout+stderr to a log file. Return exit code."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    with open(log_path, "wb") as f:
        proc = subprocess.run(
            cmd, cwd=str(cwd) if cwd else None, env=env,
            stdout=f, stderr=subprocess.STDOUT,
        )
    return proc.returncode


CONDA_SH_DEFAULT = "$HOME/anaconda3/etc/profile.d/conda.sh"


def wrap_with_conda(
    cmd: list[str],
    conda_env: str | None,
    *,
    conda_sh: str = CONDA_SH_DEFAULT,
) -> list[str]:
    """If conda_env is given, prepend `conda activate` so the command runs
    inside that env. Returns a bash -lc invocation list.

    If conda_env is None or empty, the input cmd is returned unchanged.
    """
    if not conda_env:
        return cmd
    quoted = " ".join(_sh_quote(c) for c in cmd)
    bash_line = f"source {conda_sh} && conda activate {conda_env} && {quoted}"
    return ["bash", "-lc", bash_line]


def _sh_quote(s: str) -> str:
    """Minimal POSIX shell quoting (single quotes; escape internal singles)."""
    return "'" + s.replace("'", "'\\''") + "'"


# ───────────────────────────── output discovery ───────────────────────
def find_first_mp4(work_dir: Path | str,
                   exclude_substrings: tuple[str, ...] = ()) -> Path | None:
    """Walk `work_dir` recursively, return the first non-trivial mp4 found
    whose name does NOT contain any of `exclude_substrings`. Useful when a
    baseline writes outputs under nested subfolders with timestamps."""
    work_dir = Path(work_dir)
    if not work_dir.is_dir():
        return None
    for p in sorted(work_dir.rglob("*.mp4")):
        name = p.name.lower()
        if any(s.lower() in name for s in exclude_substrings):
            continue
        if p.stat().st_size > 0:
            return p
    return None


def normalize_output_video(
    src_mp4: Path,
    dst_mp4: Path,
    *,
    target_h: int | None = None,
    target_w: int | None = None,
) -> Path:
    """Move/copy `src_mp4` to `dst_mp4`. If target_h/w provided AND ffmpeg is
    on PATH, also re-encode to the target spatial size (used to coerce a
    baseline's native output resolution to ERP 512×1024 if needed). Otherwise
    just shutil.move."""
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    if target_h and target_w and shutil.which("ffmpeg"):
        # Ensure fresh
        if dst_mp4.exists():
            dst_mp4.unlink()
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(src_mp4),
            "-vf", f"scale={int(target_w)}:{int(target_h)}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-an", str(dst_mp4),
        ]
        rc = subprocess.run(cmd).returncode
        if rc == 0 and dst_mp4.is_file() and dst_mp4.stat().st_size > 0:
            return dst_mp4
        # fall through to plain move on failure
    shutil.move(str(src_mp4), str(dst_mp4))
    return dst_mp4


# ───────────────────────────── filtering ──────────────────────────────
def select_clips(
    rows: list[dict],
    splits: list[str] | None = None,
    only_clips: list[str] | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Subset master rows by split / explicit clip_id list / limit."""
    out = rows
    if splits:
        sset = set(splits)
        out = [r for r in out if r.get("split") in sset]
    if only_clips:
        cset = set(only_clips)
        out = [r for r in out if r.get("clip_id") in cset]
    if limit:
        out = out[: int(limit)]
    return out


__all__ = [
    "MASTER_DEFAULT",
    "read_master",
    "clip_id_to_dirname",
    "load_caption_text",
    "load_caption_with_style",
    "read_first_frame_rgb_uint8",
    "save_image_png",
    "erp_to_perspective_image",
    "make_static_video_from_image",
    "per_clip_outdir",
    "write_run_json",
    "have_output",
    "run_subprocess",
    "wrap_with_conda",
    "find_first_mp4",
    "normalize_output_video",
    "select_clips",
]
