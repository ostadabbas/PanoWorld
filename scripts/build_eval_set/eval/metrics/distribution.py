"""Distributional video / image quality metrics.

These are the *set-level* metrics that complement the per-clip ones in
``metrics/visual.py``: they cannot be computed from a single (pred, gt) pair,
but only from sets of pred videos and sets of GT videos.

Implemented:

  * ``vq_fvd``  — Fréchet Video Distance using ``torchvision`` R(2+1)D-18
                  pretrained on Kinetics-400 (replaces the original Sport-1M
                  I3D from the FVD paper; reproducible because weights ship
                  with torchvision and don't depend on broken Dropbox links).
  * ``vq_faed`` — Fréchet "Aux Encoder" Distance using a Swin3D-T transformer
                  also pretrained on Kinetics-400. Different architectural
                  inductive bias from R(2+1)D so when both numbers move
                  together they reinforce each other; when they diverge it's
                  a signal worth investigating.
  * ``vq_fid``  — Fréchet Inception Distance, image-level. Per video we
                  extract per-frame Inception-v3 ``pool3`` features for 8
                  uniformly-sampled frames and average them, giving one
                  2048-d feature per video. FID is then computed over the
                  resulting set.

Usage from ``compute_distributional.py``:

    feats_pred = extract_set_embeddings(pred_paths, model="r3d18", cache=...)
    feats_gt   = extract_set_embeddings(gt_paths,   model="r3d18", cache=...)
    fvd        = frechet_distance(feats_pred, feats_gt)

Heavy models are loaded lazily and cached per-process. Per-clip embeddings
are cached on disk so re-running aggregation across new method outputs is
O(K_new) extractions, not O(K_total).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

try:
    from ._common import (
        DEFAULT_TARGET_FPS, DEFAULT_TARGET_SECS, DEFAULT_TARGET_FRAMES,
        align_to_eval_grid, read_video_uint8,
    )
except ImportError:
    from _common import (  # type: ignore
        DEFAULT_TARGET_FPS, DEFAULT_TARGET_SECS, DEFAULT_TARGET_FRAMES,
        align_to_eval_grid, read_video_uint8,
    )


# ---------------------------------------------------------------- model spec
# Each entry is:
#   "model_key": (loader_fn_name, sample_frames, spatial, mode)
#       loader_fn_name : module-private fn that returns (torch.nn.Module, dim)
#       sample_frames  : how many frames the encoder consumes (T axis)
#       spatial        : H == W after resize before encoder
#       mode           : "video3d" or "image_per_frame"
_SPECS: dict[str, dict] = {
    "r3d18":     dict(loader="_load_r3d18",     T=16, S=224, mode="video3d",
                       mean=(0.43216, 0.394666, 0.37645),
                       std =(0.22803, 0.22145,  0.216989)),
    "swin3d_t":  dict(loader="_load_swin3d_t",  T=16, S=224, mode="video3d",
                       mean=(0.485, 0.456, 0.406),
                       std =(0.229, 0.224, 0.225)),
    "inception": dict(loader="_load_inception", T=8,  S=299, mode="image_per_frame",
                       mean=(0.485, 0.456, 0.406),
                       std =(0.229, 0.224, 0.225)),
}

_MODEL_CACHE: dict[str, tuple] = {}


def _get_torch():
    import torch
    return torch


# ---------------------------------------------------------------- model loaders
def _load_r3d18():
    torch = _get_torch()
    from torchvision.models.video import r3d_18, R3D_18_Weights
    net = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    # strip classification head → (B, 512) global features
    net.fc = torch.nn.Identity()
    return net.eval(), 512


def _load_swin3d_t():
    torch = _get_torch()
    from torchvision.models.video import swin3d_t, Swin3D_T_Weights
    net = swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1)
    net.head = torch.nn.Identity()  # → (B, 768)
    return net.eval(), 768


def _load_inception():
    torch = _get_torch()
    from torchvision.models import inception_v3, Inception_V3_Weights
    net = inception_v3(
        weights=Inception_V3_Weights.IMAGENET1K_V1,
        aux_logits=True,           # required by the pretrained checkpoint
    )
    net.fc = torch.nn.Identity()   # → (B, 2048) pool3 features
    if hasattr(net, "AuxLogits"):
        net.AuxLogits = None
    return net.eval(), 2048


def _get_model(model_key: str, device: str = "cuda"):
    if model_key in _MODEL_CACHE:
        entry = _MODEL_CACHE[model_key]
        if entry is None:
            raise RuntimeError(f"model {model_key} previously failed to load")
        return entry
    spec = _SPECS[model_key]
    fn = globals()[spec["loader"]]
    try:
        net, dim = fn()
    except Exception as e:
        # Cache failure so we fail fast on subsequent calls instead of
        # retrying network downloads N times.
        _MODEL_CACHE[model_key] = None
        raise RuntimeError(f"failed to load {model_key}: {e}") from e
    torch = _get_torch()
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    net = net.to(device)
    _MODEL_CACHE[model_key] = (net, dim, device, spec)
    return _MODEL_CACHE[model_key]


# ---------------------------------------------------------------- I/O + cache
def _video_signature(path: str | Path) -> str:
    """Cheap fingerprint of a video file (size + mtime + path tail) used as
    cache invalidation key. NOT a content hash — we only need to detect
    'this video changed' between runs, not to detect cross-file collisions."""
    p = Path(path)
    try:
        st = p.stat()
        s = f"{p.name}|{st.st_size}|{int(st.st_mtime)}"
    except Exception:
        s = str(p)
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def _read_and_normalize(path: str, src_fps: float,
                        target_frames: int = DEFAULT_TARGET_FRAMES,
                        target_fps:    int = DEFAULT_TARGET_FPS,
                        target_secs: float = DEFAULT_TARGET_SECS,
                        ) -> np.ndarray | None:
    v = read_video_uint8(path)
    if v is None or v.ndim != 4 or v.shape[0] == 0:
        return None
    return align_to_eval_grid(
        v, src_fps,
        target_fps=target_fps, target_secs=target_secs,
        target_frames=target_frames,
    )


# ---------------------------------------------------------------- preprocess
def _to_model_tensor(video_thwc: np.ndarray, spec: dict, torch_mod):
    """uint8 (T,H,W,3) -> normalized tensor in the shape the model expects.

    For 'video3d' mode: returns (1, 3, T, S, S) float32.
    For 'image_per_frame' mode: returns (T, 3, S, S) float32.
    """
    T_target = spec["T"]
    S = spec["S"]

    T_src = video_thwc.shape[0]
    if T_src >= T_target:
        idx = np.linspace(0, T_src - 1, T_target).round().astype(np.int64)
    else:
        idx = np.clip(np.arange(T_target), 0, T_src - 1)
    v = video_thwc[idx]                                     # (T,H,W,3) uint8

    x = torch_mod.from_numpy(v).float() / 255.0             # (T,H,W,3) [0,1]
    x = x.permute(0, 3, 1, 2).contiguous()                  # (T,3,H,W)
    x = torch_mod.nn.functional.interpolate(
        x, size=(S, S), mode="bilinear", align_corners=False
    )                                                       # (T,3,S,S)

    mean = torch_mod.tensor(spec["mean"]).view(1, 3, 1, 1)
    std  = torch_mod.tensor(spec["std"]).view(1, 3, 1, 1)
    x = (x - mean) / std

    if spec["mode"] == "video3d":
        x = x.permute(1, 0, 2, 3).unsqueeze(0).contiguous()  # (1,3,T,S,S)
    return x


def _embed_one(video_thwc: np.ndarray, model_key: str,
               device: str = "cuda") -> np.ndarray:
    """Embed a single clip. Kept as a thin wrapper around _embed_batch for
    callers that don't have a batch in hand."""
    return _embed_batch([video_thwc], model_key, device=device)[0]


def _embed_batch(videos: list[np.ndarray], model_key: str,
                 device: str = "cuda") -> np.ndarray:
    """Embed a batch of clips. Returns (N, D) float32.

    For video3d backbones the model batch axis carries N clips directly.
    For image-per-frame backbones (Inception) we flatten N×T frames into a
    single forward pass, then mean-pool back per clip — this gives one
    feature vector per clip for FID, but keeps the GPU well-fed.
    """
    if not videos:
        return np.zeros((0, 0), dtype=np.float32)
    torch = _get_torch()
    net, dim, device, spec = _get_model(model_key, device=device)
    xs = [_to_model_tensor(v, spec, torch) for v in videos]

    with torch.no_grad():
        if spec["mode"] == "video3d":
            x = torch.cat(xs, dim=0).to(device)         # (N, 3, T, S, S)
            f = net(x).flatten(1)                       # (N, D)
            arr = f.detach().cpu().numpy()
        else:
            T = xs[0].shape[0]
            x = torch.cat(xs, dim=0).to(device)         # (N*T, 3, S, S)
            f = net(x)                                  # (N*T, D)
            if isinstance(f, tuple):                    # InceptionOutputs
                f = f[0]
            f = f.view(len(xs), T, -1).mean(dim=1)      # (N, D)
            arr = f.detach().cpu().numpy()
    return arr.astype(np.float32)


# ---------------------------------------------------------------- public
def extract_set_embeddings(
    paths: list[str],
    fps_list: list[float],
    *,
    model_key: str,
    cache_root: Path | None = None,
    cache_tag:  str = "",
    device: str = "cuda",
    progress_every: int = 25,
    name: str = "",
    batch_size: int = 8,
) -> tuple[np.ndarray, list[str]]:
    """Compute and stack embeddings over a list of videos.

    Two-pass implementation:

      1. Cache lookup pass — for every (path, model) signature that has a
         cached .npy on disk, load it and skip embedding entirely. GT cache
         is shared across methods so once GT is embedded once, every
         subsequent ``compute_distributional.py`` invocation reuses it.

      2. Batched-embed pass — videos that missed the cache are read,
         resampled to the eval grid, and pushed through the encoder in
         batches of ``batch_size``. Video3D backbones see (B, 3, T, S, S);
         image-per-frame Inception sees (B*T, 3, S, S) flattened then
         mean-pooled back per clip. Empirically this is ~2-3× single-clip
         on an A100 (per-call CUDA launch overhead amortizes well).

    Returns
    -------
    feats : (N_kept, D) float32. May be < len(paths) if some videos failed
            to read or the model failed.
    used  : list of paths that contributed (same order as feats rows).
    """
    # First pass: collect cache hits + queue cache misses (preserve order).
    n = len(paths)
    feats_by_idx: dict[int, np.ndarray] = {}
    miss_indices: list[int] = []
    miss_caches:  dict[int, Path | None] = {}

    for i, (p, _) in enumerate(zip(paths, fps_list)):
        cache_path: Path | None = None
        if cache_root is not None:
            sig = _video_signature(p)
            cache_path = (Path(cache_root) / cache_tag / model_key /
                          f"{sig}.npy")
            if cache_path.is_file():
                try:
                    feats_by_idx[i] = np.load(cache_path)
                    continue
                except Exception:
                    pass
        miss_indices.append(i)
        miss_caches[i] = cache_path

    if feats_by_idx and progress_every:
        print(f"[dist] {name} {model_key}: {len(feats_by_idx)}/{n} cache hits",
              flush=True)

    # Second pass: embed in batches.
    embed_done = 0
    for start in range(0, len(miss_indices), batch_size):
        chunk_idx = miss_indices[start:start + batch_size]

        videos:    list[np.ndarray] = []
        video_idx: list[int]        = []
        for i in chunk_idx:
            v_arr = _read_and_normalize(paths[i], fps_list[i])
            if v_arr is None:
                print(f"[dist] WARN {name} {model_key}: read fail on "
                      f"{paths[i]}", flush=True)
                continue
            videos.append(v_arr)
            video_idx.append(i)

        if not videos:
            continue

        try:
            embs = _embed_batch(videos, model_key, device=device)
        except Exception as e:
            # Fall back to per-video so a single bad clip doesn't drop a whole
            # batch (e.g. corrupted decode that survives _read_and_normalize).
            print(f"[dist] WARN {name} {model_key}: batch fail "
                  f"({len(videos)} clips): {e}; retrying per-clip",
                  flush=True)
            embs_list = []
            for v_arr in videos:
                try:
                    embs_list.append(_embed_one(v_arr, model_key, device=device))
                except Exception as e2:
                    print(f"[dist] WARN {name} {model_key}: embed fail: {e2}",
                          flush=True)
                    embs_list.append(None)
            embs = embs_list  # type: ignore[assignment]

        for k, i in enumerate(video_idx):
            emb = embs[k]
            if emb is None:
                continue
            feats_by_idx[i] = np.asarray(emb, dtype=np.float32)
            cache_path = miss_caches.get(i)
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    np.save(cache_path, emb)
                except Exception:
                    pass

        embed_done += len(video_idx)
        if progress_every and embed_done % progress_every < batch_size:
            print(f"[dist] {name} {model_key}: embedded "
                  f"{embed_done}/{len(miss_indices)}", flush=True)

    # Materialize in original order.
    if not feats_by_idx:
        return np.zeros((0, 1), dtype=np.float32), []
    keep = sorted(feats_by_idx.keys())
    feats = np.stack([feats_by_idx[i] for i in keep], axis=0).astype(np.float32)
    used  = [paths[i] for i in keep]
    return feats, used


# ---------------------------------------------------------------- Fréchet
def frechet_distance(feats_a: np.ndarray, feats_b: np.ndarray,
                     eps: float = 1e-6) -> float:
    """Standard Fréchet distance between two empirical Gaussian fits.

    d^2 = ||mu_a - mu_b||^2 + tr(Sa + Sb - 2 sqrt(Sa @ Sb))

    Returns NaN when either set has < 2 samples (covariance undefined).
    Numerical safety: matrix sqrt eats a tiny diagonal jitter and any tiny
    imaginary residue is dropped.
    """
    if feats_a.shape[0] < 2 or feats_b.shape[0] < 2:
        return float("nan")
    if feats_a.shape[1] != feats_b.shape[1]:
        return float("nan")

    mu_a = feats_a.mean(0)
    mu_b = feats_b.mean(0)
    Sa = np.cov(feats_a, rowvar=False)
    Sb = np.cov(feats_b, rowvar=False)

    diff = mu_a - mu_b

    # Standard FID-style sqrtm with retry on numerical failure (heavily
    # rank-deficient cov when N << D; we have ~50-150 clips × 512-2048 dim).
    from scipy.linalg import sqrtm
    I = np.eye(Sa.shape[0])
    covmean, _ = sqrtm(Sa.dot(Sb), disp=False)
    if (covmean is None) or (not np.isfinite(covmean).all()):
        # retry with a stronger diagonal jitter
        for jitter in (1e-6, 1e-3, 1e-1):
            covmean, _ = sqrtm((Sa + jitter * I).dot(Sb + jitter * I),
                               disp=False)
            if np.isfinite(covmean).all():
                break
    if (covmean is None) or (not np.isfinite(covmean).all()):
        return float("nan")
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fd = float(diff.dot(diff) + np.trace(Sa) + np.trace(Sb)
               - 2.0 * np.trace(covmean))
    # Mathematically non-negative; any tiny negative is sqrtm jitter on a
    # near-rank-deficient cov (we have ~150 clips << feature dim) and is
    # safe to clamp to 0. Don't hide LARGE negatives: they indicate the
    # sqrtm branch is wrong and the result is unreliable -> return NaN.
    if fd < 0:
        if abs(fd) < 1.0:
            return 0.0
        return float("nan")
    return fd


# ---------------------------------------------------------------- orchestration
MODEL_TO_METRIC = {
    "r3d18":     "vq_fvd",
    "swin3d_t":  "vq_faed",
    "inception": "vq_fid",
}


def compute_bucket_metrics(
    pred_paths: list[str], pred_fps: list[float],
    gt_paths:   list[str], gt_fps:   list[float],
    *,
    cache_root: Path | None,
    pred_cache_tag: str,                  # e.g. "pred/<method_id>"
    gt_cache_tag:   str = "gt",
    device: str = "cuda",
    min_samples: int = 8,
    models: tuple[str, ...] = ("r3d18", "swin3d_t", "inception"),
    batch_size: int = 8,
) -> dict[str, float]:
    """Run all distributional metrics for one (pred set, gt set) bucket.

    Returns ``{metric_name: value}`` with NaN for buckets below
    ``min_samples`` (defaults to 8).
    """
    out: dict[str, float] = {m: float("nan") for m in MODEL_TO_METRIC.values()}
    if (len(pred_paths) < min_samples) or (len(gt_paths) < min_samples):
        return out

    for mk in models:
        feats_pred, _ = extract_set_embeddings(
            pred_paths, pred_fps,
            model_key=mk,
            cache_root=cache_root, cache_tag=pred_cache_tag,
            device=device, name=f"pred[{pred_cache_tag}]",
            batch_size=batch_size,
        )
        feats_gt, _ = extract_set_embeddings(
            gt_paths, gt_fps,
            model_key=mk,
            cache_root=cache_root, cache_tag=gt_cache_tag,
            device=device, name="gt",
            batch_size=batch_size,
        )
        out[MODEL_TO_METRIC[mk]] = frechet_distance(feats_pred, feats_gt)
    return out


__all__ = [
    "MODEL_TO_METRIC",
    "extract_set_embeddings",
    "frechet_distance",
    "compute_bucket_metrics",
]
