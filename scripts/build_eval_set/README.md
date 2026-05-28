# PanoWorld Evaluation Toolkit

This directory bundles everything needed to **reproduce the PanoWorld paper's quantitative results** and to **build new panoramic evaluation test sets**.

```
scripts/build_eval_set/
├── README.md              ← you are here
├── build_master_csv.py    ← index a directory of clips into the eval-set master CSV
├── habitat/               ← synthetic ERP renderer (Habitat-Sim)
│   ├── render_habitat_erp.py
│   └── compute_habitat_tracks_analytic.py
├── eval/                  ← main evaluation framework
│   ├── README.md          ← detailed metric + runner documentation
│   ├── run_eval.py        ← top-level driver
│   ├── aggregate.py
│   ├── auto_stratify.py
│   ├── compute_distributional.py
│   ├── make_balanced_tables.py
│   ├── eval_config.yaml
│   ├── eval_config_geom_only.yaml
│   ├── metrics/           ← depth / tracks / trajectory / visual / caption_alignment / ...
│   ├── runners/           ← per-method inference drivers (PanoWorld + baselines)
│   └── strata/strata.csv
└── docs/
    └── argus_manifest.csv ← the 50 Argus real-world clips in the released test set
```

## What's in our paper's evaluation set

The PanoWorld test set bundles **150 clips total** across three strata:

| Stratum | Clips | Source |
|---|---|---|
| `self_iid` | 50 | Held-out clips from PanoGeo (matches training distribution) |
| `argus_ood` | 50 | Real YouTube 360° clips, length-normalized to 5 s @ 1024×512 |
| `habitat_ood` | 50 | Synthetic Habitat-Sim renders (perfect GT depth + trajectory) |

The pre-built test set is published on Google Drive — see [`docs/DATASET.md`](../../docs/DATASET.md) for the download command.

## What you can do with this toolkit

### 1. Reproduce paper numbers with the released checkpoint

```bash
# 1) Download the test set (one-time)
export PANO_DATA_ROOT=$HOME/pano_video_data
rclone copy gdrive:PanoWorld/test/ $PANO_DATA_ROOT/test/ -P

# 2) Run PanoWorld chained inference (~20 h on a single H100)
python scripts/build_eval_set/eval/runners/infer_panoworld_chained.py \
    --master $PANO_DATA_ROOT/test/master.csv \
    --results eval_results \
    --finetune_checkpoint checkpoints/panoworld_main/model_ema_bf16.pt \
    --method_id panoworld_main --round1_reuse_dir "" --scene_first

# 3) Score
python scripts/build_eval_set/eval/run_eval.py \
    --results_root eval_results/ \
    --master $PANO_DATA_ROOT/test/master.csv \
    --methods panoworld_main \
    --output eval_results/scores_panoworld_main.csv
```

Full step-by-step in [`docs/PANOWORLD_INFERENCE_GUIDE.md`](../../docs/PANOWORLD_INFERENCE_GUIDE.md).

### 2. Compare to baselines

`eval/runners/` ships drop-in inference drivers for the baselines reported in the paper:

| Method | Runner | Notes |
|---|---|---|
| **PanoWorld** (ours) | `infer_panoworld.py`, `infer_panoworld_chained.py`, `infer_panoworld_batch.py` | This repo |
| Argus | `infer_argus.py` | clone the baseline separately |
| 360DVD | `infer_360dvd.py` | clone the baseline separately |
| Follow-Your-Canvas | `infer_fyc.py` | clone the baseline separately |
| Imagine360 | `infer_imagine360.py` | clone the baseline separately |
| OmniRoam | `infer_omniroam.py` | clone the baseline separately |

See [`eval/runners/BASELINE_SETUP.md`](eval/runners/BASELINE_SETUP.md) for how to install the upstream code of each baseline. Pre-baked baseline outputs are also published on Google Drive (see [Model Zoo](../../docs/MODEL_ZOO.md)).

### 3. Build a new evaluation test set

To extend or replace our test set with your own panoramic clips:

```bash
# 1) Drop your panoramic .mp4 clips under a flat directory:
#      my_eval_clips/
#        clip_001.mp4
#        clip_002.mp4
#        ...
# 2) Annotate (depth + tracks) with the main repo's annotation pipeline:
python generate_annotations.py \
    --train_data_dir my_eval_clips \
    --output_dir     my_eval_clips/annotations

# 3) Index the directory into a master.csv
python scripts/build_eval_set/build_master_csv.py \
    --root my_eval_clips \
    --output my_eval_clips/master.csv
```

You can now plug `my_eval_clips/master.csv` into any `infer_*` runner.

### 4. Generate synthetic ERP clips with perfect GT (Habitat-Sim)

```bash
# requires habitat-sim 0.3+ and a Replica-CAD or HM3D scene
python scripts/build_eval_set/habitat/render_habitat_erp.py \
    --scene  path/to/replica/apartment_0/apartment_0.glb \
    --output my_synth_clips/ \
    --num_clips 50 --resolution 1024,512 --num_frames 80
```

Produces ERP `.mp4` clips alongside ground-truth depth maps + camera trajectories that you can run against `compute_habitat_tracks_analytic.py` to get analytical 3D tracks for the trajectory-consistency metric.

## Metrics

The evaluation produces per-clip and per-method scores for:

- **Depth consistency** (`metrics/depth.py`) — pseudo-GT panoramic depth alignment.
- **Trajectory consistency** (`metrics/tracks.py`, `metrics/trajectory.py`) — 3D world-frame point trajectories over time.
- **Long-horizon consistency** (`metrics/long_horizon.py`) — drift across the 5 s window.
- **Self-consistency** (`metrics/self_consistency.py`) — seam closure at longitude ±180°.
- **Visual quality** (`metrics/visual.py`) — VBench-style aesthetic / motion smoothness scores.
- **Distributional** (`metrics/distribution.py`, `compute_distributional.py`) — FVD-like FID over Inflated-3D features.
- **Caption alignment** (`metrics/caption_alignment.py`) — CLIPScore between caption and rendered ERP frames.

See [`eval/README.md`](eval/README.md) for definitions, formulas, and per-metric runtime budgets.

## Pre-baked outputs on Google Drive

To make every paper number reproducible *without* re-running 6+ baselines, we also publish the inference outputs (`<clip>/video.mp4`) as `.tar.gz` archives. See [Model Zoo](../../docs/MODEL_ZOO.md) for the GD links. Drop them into `eval_results/` and you can re-run `run_eval.py` to verify the published metrics in ~30 minutes (rather than ~5 days of GPU time).
