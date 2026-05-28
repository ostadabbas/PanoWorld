# PanoWorld Evaluation Pipeline

> Philosophy: **record everything, decide later.** Every run produces a single
> *fat* CSV (`results_long.csv`) where rows are
> `(clip_id × method × metric_name × strata)`. Aggregation and "which slice goes
> into the paper" is a post-processing step (`aggregate.py`).

## Layout

```
eval/
├── README.md                    ← this file
├── eval_config.yaml             ← single source of truth for paths + flags
├── auto_stratify.py             ← caption-based scene-type classifier → strata.csv
├── run_eval.py                  ← main entry (per-clip pass): master.csv + results_dir → results_long.csv
├── compute_distributional.py    ← post-pass: FVD/FAED/FID over (method × bucket); appends to same csv
├── aggregate.py                 ← results_long.csv → publication-ready tables
├── metrics/
│   ├── __init__.py              ← registry of metric callables
│   ├── visual.py                ← LPIPS / SSIM / PSNR / CLIP-T (per-clip)
│   ├── distribution.py          ← FVD / FAED / FID model loaders + Fréchet distance
│   ├── depth.py                 ← AbsRel / δ<1.25 / sky-masked / texture-weighted
│   ├── trajectory.py            ← per-window PSNR @ GT-traj (OmniRoam) + ATE / APD
│   ├── tracks.py                ← AJ / OA / δ_avg (TAP-Vid) + 3D-track consistency
│   ├── long_horizon.py          ← NVS-PSNR / loop-consistency [DEFERRED]
│   └── caption_alignment.py     ← CLIP-T (text↔video) + caption-content scores
├── runners/                     ← per-method inference wrappers
│   ├── infer_panoworld.py      ← ours (perspective + ERP-init modes)
│   ├── infer_omniroam.py        ← TODO
│   ├── infer_360dvd.py          ← TODO
│   ├── infer_imagine360.py      ← TODO
│   ├── infer_argus.py           ← TODO
│   └── infer_matrix3d.py        ← deferred (added once others are running)
└── strata/
    └── strata.csv               ← OUTPUT of auto_stratify.py
```

## Strata schema (strata.csv)

| col          | values                                                                |
|--------------|-----------------------------------------------------------------------|
| clip_id      | matches master.csv                                                    |
| scene_type   | `textured_indoor` / `outdoor_with_sky` / `mixed_low_texture`          |
| sky_pct      | float ∈ [0,1] — sky pixel fraction (heuristic + caption-overridden)   |
| motion_kind  | `static` / `walk` / `rotate` / `walk_rotate` (copied from master)     |
| n_frames     | int                                                                   |
| split        | `self_iid` / `argus_ood` / `habitat_ood`                              |

Multiple strata are **orthogonal** — when reporting we can group by any column or
combination (e.g. `outdoor × dynamic × Argus`).

## results_long.csv schema (THE fat csv)

| col          | type | example                                           |
|--------------|------|---------------------------------------------------|
| clip_id      | str  | `argus_ood::abc123_clip001`                       |
| split        | str  | `argus_ood`                                       |
| method       | str  | `panoworld_pers` / `panoworld_erp` / `omniroam` |
| input_kind   | str  | `pers_image` / `pers_video` / `erp_image` / `text`|
| metric_name  | str  | `depth_abs_rel/sky_masked` / `traj_psnr@25-30`    |
| value        | float| 0.137                                             |
| strata_json  | json | `{"scene_type": "outdoor_with_sky", ...}`         |
| run_id       | str  | timestamp + git-hash for traceability             |

Rules:
* **One row per metric value.** Variants of a metric (sky-masked vs raw, frame
  window 25-30 vs 55-60) are separate rows with different `metric_name`.
* **NaN allowed.** A metric that doesn't apply (e.g. ATE on static clip)
  produces a NaN row, not a missing row, so we always know what we tried.
* **Append-only.** Each `run_eval.py` invocation appends; aggregator dedupes
  on `(clip_id, method, metric_name, run_id)`.

## Input parity policy (READ THIS FIRST)

PanoWorld is **free-form** generation: it takes only `{image, caption}`
and the camera goes wherever the model has learned to take it. OmniRoam
and Matrix-3D are **controllable**, also consuming a per-frame camera
trajectory; Imagine360 and ARGUS are **video-conditioned**, consuming a
perspective video. To prevent any of these extra inputs from giving the
respective method an unearned advantage, we lock the input modality:

| Stage   | Allowed inputs                                  |
|---------|-------------------------------------------------|
| Stage 1 | perspective image + caption                     |
| Stage 2 | ERP image + caption                             |

Methods whose architecture demands extra inputs are handicapped, never
helped:

* OmniRoam / Matrix-3D → fed a **fixed forward trajectory** (NOT GT).
* Imagine360 / ARGUS → fed a **static-repeat video** (the single
  perspective image replicated 25 times). We deliberately do NOT
  generate a synthetic video with any auxiliary model (e.g. Cosmos-base)
  because that would inject content not available to PanoWorld and
  break parity.

### Stage-by-stage metric matrix

Without trajectory control, "GT frame _t_ ↔ generated frame _t_"
correspondence does not exist past frame 0. So any metric that compares a
GT frame at time _t_ with a generated frame at time _t_ measures "did your
hallucinated camera happen to match GT's camera", not generation quality.
We therefore split metrics into:

| Class | Examples | Stage 1 | Stage 2 | Strictly fair on Stage 1? |
|---|---|---|---|---|
| Correspondence-free | FVD, FAED, FID, CLIP-T, depth-distribution, track-self-quality, motion-smoothness | ✅ | ✅ | ✅ |
| Correspondence-required | PSNR, SSIM, LPIPS, Depth-AbsRel/δ, Tap-AJ/OA/δ_avg | ✅ recorded | ✅ | ⚠️ noisy (frames not aligned) |

Stage 2 anchors frame 0 (real ERP first frame ≈ GT first frame), so
correspondence-required metrics make sense there even though later-frame
camera drift inflates the numbers — that drift is itself a meaningful
"camera-path stability" signal.

On Stage 1 the correspondence-required metrics are recorded too — they're
not strictly fair (cameras roam differently across methods) but if our
method produces systematically more GT-like hallucination they may trend
in our favor over 150 clips. We keep them in the fat csv and decide later
whether to publish.

Trajectory-controllability metrics (traj-PSNR, ATE, APD, loop-consistency)
remain DISABLED throughout — PanoWorld is free-form, we don't claim that
axis.

### Output parity (spatial)

Different baselines emit different ERP resolutions:

| baseline           | native output | how we handle it                  |
|--------------------|---------------|-----------------------------------|
| 360DVD / ARGUS / FYC / Imagine360 / PanoWorld | 1024×512   | direct compare, no rescale        |
| **OmniRoam Preview**   | **960×480**   | bilinearly resized **up to 1024×512 (GT)** before pixel metrics |
| OmniRoam Refine    | 1440×720      | resized **down to 1024×512 (GT)** before pixel metrics |

Implementation: `metrics/_common.py::align_pair_to_eval_grid`. Pred is resized
to GT's H×W (default) before any pixel-level metric (PSNR/SSIM/LPIPS/CLIP-T).
This replaces an earlier crop-to-min behavior which would silently drop the
GT pole/equator regions for any baseline whose output was smaller than GT
(biasing those baselines' scores in their favor since the hardest-to-render
ERP boundaries were never compared).

Distribution metrics (FVD/FAED/FID) are unaffected — their backbones already
internally rescale every frame to the model's input size (e.g. 224×224).

Temporal fps (`method_fps_overrides` in `run_eval.py`) is the **semantic** fps
of the generator's training, not the mp4 container's `r_frame_rate`. For
example OmniRoam writes `fps=30` to the container but its underlying Wan-2.1
backbone has `sample_fps=16`, so we pass 16 to `align_pair_to_eval_grid`.

## PanoWorld two-stage inference

This matches the paper's rollout-rounds description (`method.tex` ¶252).

```
Stage 1 (perspective in):
   pers_image + caption
     --[Round 1: PanoWorld]-->  ERP first frame (hallucinated)
     --[Round 2: PanoWorld]-->  5s ERP video  (evaluated)

Stage 2 (ERP in):
   real ERP first frame (from GT clip) + caption
     --[Round 2: PanoWorld]-->  5s ERP video  (evaluated)
```

Why this layout:
* Whole inference loop is PanoWorld end-to-end — no Cosmos-base or any
  other auxiliary model contaminates the Stage-1 pipeline.
* Round 2 has full panoramic freedom (camera not pinned to the input
  perspective FOV).
* Stage 1 → Stage 2 quality gap = "Round-1 hallucination cost", a clean
  ablation we can read off the same data.
* Each baseline gets the equivalent treatment (image + caption only;
  static repeat / fixed forward as needed).

## Stage-1 vs Stage-2 tables

`aggregate.py` produces:

**Stage 1 — perspective input, all baselines**
```
methods   = [panoworld_pers, omniroam_pers, 360dvd, imagine360, argus, matrix3d]
input     = perspective image + caption (omniroam/matrix3d also get fixed forward traj)
strata    = ALL groupings recorded; pick the best slice at write-up
```

**Stage 2 — ERP input, our model vs OmniRoam (only)**
```
methods   = [panoworld_erp, omniroam_erp]
input     = ERP image + caption (omniroam also gets fixed forward traj)
notes     = supplementary; included to demonstrate ERP-init also works
```

## Running

```bash
# 1. (one-time) classify all 150 clips into strata
python auto_stratify.py \
    --master $PANO_DATA_ROOT/test/master.csv \
    --out strata/strata.csv

# 2. (per method, per split) launch inference → write videos to results_dir
python runners/infer_panoworld.py --master ... --out results/panoworld_pers/

# 3. compute every PER-CLIP metric for every (clip, method) under results_dir
python run_eval.py \
    --master $PANO_DATA_ROOT/test/master.csv \
    --strata strata/strata.csv \
    --results results/ \
    --out results_long.csv

# 3b. compute DISTRIBUTIONAL metrics (FVD / FAED / FID) — per (method × bucket)
#     Buckets emitted: "all" + per-split + per-scene_type. Appends to the
#     same results_long.csv as bucket-level rows (clip_id="").
python compute_distributional.py \
    --master $PANO_DATA_ROOT/test/master.csv \
    --strata strata/strata.csv \
    --results results/ \
    --out results_long.csv

# 4. produce paper tables (averaged + stratified)
python aggregate.py --in results_long.csv --out paper_tables/
```

### Distributional metrics — what they measure & how they're computed

| metric    | direction | encoder                                   | feature dim | granularity |
|-----------|-----------|-------------------------------------------|-------------|-------------|
| `vq_fvd`  | ↓         | R(2+1)D-18 pretrained on Kinetics-400     | 512         | clip-level (16 frames sampled across 80f eval grid) |
| `vq_faed` | ↓         | Swin3D-T pretrained on Kinetics-400       | 768         | clip-level (16 frames, transformer) |
| `vq_fid`  | ↓         | Inception-v3 pool3 (ImageNet)             | 2048        | per-frame (8 frames averaged → one feature per clip) |

For each (method × bucket):
1. Each video is resampled onto the eval grid (80f@16fps), encoded with the
   chosen backbone, and reduced to one feature vector per video.
2. Pred-set and GT-set features are each fit to a Gaussian (mean + cov).
3. Fréchet distance: `||μ_p−μ_g||² + tr(Σ_p+Σ_g − 2·sqrt(Σ_p·Σ_g))`.
4. Per-clip embeddings are cached under `<results_dir>/_dist_cache/...` so
   re-running after only adding new method outputs is O(K_new) extractions.

### Multi-GPU sharding (optional, for fast eval on multi-GPU nodes)

Single-GPU run on ~150 clips × 6 methods × 3 backbones takes ~60 min wall
time (mostly mp4 decode, not GPU). To go faster on a multi-GPU node, shard
**by method** across GPUs — distinct methods write to disjoint cache
directories so there's no contention between workers. Run the GT pre-pass
**once** to populate the shared GT cache so per-method workers don't all
race on it:

```bash
# 1. one-time GT pre-pass on a single GPU (~5 min)
python compute_distributional.py --gt-only --device cuda:0 \
    --master ... --strata ... --results ...

# 2. per-method workers in parallel; each pins to its own GPU
for i in 0 1 2 3 4 5; do
    methods=(panoworld_pers omniroam_pers dvd_360 imagine360 argus matrix3d)
    CUDA_VISIBLE_DEVICES=$i python compute_distributional.py \
        --methods ${methods[$i]} --device cuda \
        --master ... --strata ... --results ... &
done; wait
```

Total wall time on 6+ GPU node: ~10 min instead of ~60.

`--batch-size` (default 8) is also exposed; raise it for larger GPUs (the
encoder forward is the cheap part; 16 is fine on 40 GB A100). Numerically
batch-invariant within FP32 noise (verified bit-exact on Inception,
≤1e-6 on R(2+1)D-18 / Swin3D-T).

### NaN floors

Buckets where N_clips < `--min-samples` (default 8) emit NaN — Fréchet on
< 8 samples is too noisy to publish. Strata buckets are:

* `all`                       (~150 clips)
* `split::self_iid` / `split::argus_ood` / `split::habitat_ood`
* `scene::textured_indoor` / `scene::outdoor_with_sky` / `scene::mixed_low_texture`

Backbone choice notes:

* We use torchvision's R(2+1)D-18 (Kinetics-400) in place of the original
  Sport-1M I3D from Unterthiner et al. 2018 because torchvision weights
  ship via official package channels (no broken Dropbox links). Recent
  works (e.g. StyleGAN-V follow-ups, MAGVITv2) report comparable rankings
  with this substitution.
* `vq_faed` uses Swin3D-T as a transformer-architecture aux encoder so
  agreement / disagreement with R(2+1)D `vq_fvd` is informative: when both
  rank methods the same we have a robust signal; divergence is worth a
  footnote.
* With ~150 clips and 512-2048-d features, every covariance matrix is
  rank-deficient. Numbers are still **comparative** (rank-consistent across
  methods), they're just not directly comparable across papers / corpora.

## OmniRoam protocol notes (NOT adopted)

OmniRoam (SIGGRAPH'26) reports per-window PSNR @ GT-trajectory as their
trajectory-controllability metric. This protocol is **not adopted** here
because PanoWorld has no trajectory conditioning — feeding everyone the
same GT trajectory only measures "how good is your traj-conditioning
module," which is a question we don't claim to answer.

What we *do* take from OmniRoam:
* the FAED metric (Fréchet AutoEncoder Distance) — robust to ERP distortion.
* their treatment of methods that don't accept a trajectory (Imagine360 is
  reported only on visual quality in their Table 1) — we do the same in the
  opposite direction: trajectory-requiring methods are reduced to a fixed
  forward trajectory.

What we deliberately *don't* take:
* GT-trajectory conditioning (would unfairly hand the answer to OmniRoam).
* loop consistency (out-of-scope; see metrics/long_horizon.py docstring).
