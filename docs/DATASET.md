# PanoWorld Dataset Guide

This document covers the two things people typically want to do with PanoWorld data:

1. **Download our evaluation test set** to reproduce paper numbers (Section 1).
2. **Build your own panoramic training set** from your own 360° videos using the Gemini caption pipeline (Section 2).

For API-token setup (Gemini etc.), see [`TOKENS.md`](TOKENS.md).

---

## 1. Download the evaluation test set

The PanoWorld benchmark consists of **150 clips** (≈ 14 GiB, evenly balanced across 3 splits) sampled from three sources:

| Split | Clips | Source |
|---|---|---|
| `self_iid/` | 50 | Self-collected panoramic clips (in-distribution) |
| `argus_ood/` | 50 | [Argus](https://argus-360.github.io/) panoramic videos (out-of-distribution real) |
| `habitat_ood/` | 50 | Habitat-Sim rendered ERP videos (out-of-distribution synthetic) |

📂 **Google Drive folder (release):** <https://drive.google.com/drive/folders/1Db7O2enPfuugamwd9mdE0IR6facOwVG0>

The test set is shipped as a single tarball (`panoworld_testset_150.tar`, ~14 GiB) in that folder, alongside the checkpoint tarball.

After [setting up rclone](#setting-up-rclone) with a `gdrive:` remote:

```bash
# Choose any local directory as the data root
export PANOWORLD_TEST_ROOT=$HOME/panoworld_test_set
mkdir -p "$PANOWORLD_TEST_ROOT" && cd "$PANOWORLD_TEST_ROOT"

# Pull the test-set tarball (~14 GiB)
rclone copy gdrive:panoworld_testset_150.tar . \
    --drive-root-folder-id=1Db7O2enPfuugamwd9mdE0IR6facOwVG0 \
    --retries 20 --retries-sleep 60s --low-level-retries 30 -P

# Unpack — creates self_iid/, argus_ood/, habitat_ood/, master.csv, README.md
tar -xf panoworld_testset_150.tar && rm panoworld_testset_150.tar

# Then expand the two per-split annotation bundles
# (videos/captions are already uncompressed; habitat_ood/annotations/ is already a directory)
tar -xf self_iid/annotations.tar           -C self_iid/
tar -xf argus_ood/annotations.tar          -C argus_ood/
tar -xf argus_ood/camera_trajectories.tar  -C argus_ood/
```

### Layout after extraction

```
$PANOWORLD_TEST_ROOT/
├── README.md                            ← drive-side documentation (mirrors this section)
├── master.csv                           ← 150 rows + 1 header, relative paths
├── self_iid/                            50 clips — real ERP, indoor
│   ├── videos/                          50× 1024×512 mp4
│   ├── captions/<cid>.json
│   ├── manifest.csv
│   └── annotations/<cid>/               unpacked from annotations.tar
├── argus_ood/                           50 clips — Argus subset (real, OOD)
│   ├── videos/                          50× 1024×512 mp4
│   ├── captions/<cid>.json
│   ├── manifest.csv
│   ├── dropped_non_2to1.csv
│   ├── annotations/<cid>/               unpacked from annotations.tar
│   └── camera_trajectories/             unpacked from camera_trajectories.tar
└── habitat_ood/                         50 clips — Habitat-Sim + Replica (synthetic, OOD)
    ├── videos/                          50× 1024×512 mp4
    ├── captions/<cid>.json
    └── annotations/<cid>/               already unpacked
```

### Verify

```bash
wc -l   $PANOWORLD_TEST_ROOT/master.csv                  # 151 (150 + header)
ls      $PANOWORLD_TEST_ROOT/self_iid/videos/   | wc -l  # 50
ls      $PANOWORLD_TEST_ROOT/argus_ood/videos/  | wc -l  # 50
ls      $PANOWORLD_TEST_ROOT/habitat_ood/videos/| wc -l  # 50
```

> **Paths in `master.csv` are relative.** The eval drivers resolve them against
> `$PANOWORLD_TEST_ROOT` (the directory holding `master.csv`). No `sed` fix-up
> needed when you move the dataset around.

### Run the chained eval

After the test set + a checkpoint are in place:

```bash
mkdir -p logs eval_results_panoworld
python scripts/build_eval_set/eval/runners/infer_panoworld_chained.py \
    --master  $PANOWORLD_TEST_ROOT/master.csv \
    --results eval_results_panoworld \
    --finetune_checkpoint checkpoints/panoworld_main/model_ema_bf16.pt \
    --method_id panoworld_main \
    --round1_reuse_dir "" \
    --scene_first
```

Run-time is ~20 hours on a single H100 80 GB. See [`PANOWORLD_INFERENCE_GUIDE.md`](PANOWORLD_INFERENCE_GUIDE.md) for the full walk-through.

---

## 2. Build your own panoramic training set

PanoWorld training data is `(clip.mp4, caption.txt)` pairs. We auto-generate captions with **Gemini 2.5 Pro** using a 4-view perspective projection (VLMs handle perspective images far better than raw equirectangular crops).

### 2a. Source video requirements

- **Format**: `.mp4`, equirectangular 2:1 aspect ratio (e.g. 1920×960, 3840×1920).
- **Length**: any length ≥ 5 s. Long videos are auto-cut into 5 s clips.
- **FPS**: any; we resample to 16 fps internally.
- **Content**: anything; both indoor scenes and outdoor driving worked in our experiments.

Organize by scene (each scene = one folder of related videos):

```
$DATA_ROOT/
├── my_scene1/
│   ├── VID_001.mp4
│   ├── VID_002.mp4
│   └── ...
└── my_scene2/
    └── VID_001.mp4
```

### 2b. Cut + caption

```bash
# 1. Set your Gemini key (one of two ways, see TOKENS.md)
export GEMINI_API_KEY="AIzaSy..."

# 2. Run the end-to-end pipeline
SOURCE_ROOT=$DATA_ROOT \
SCENES="my_scene1 my_scene2" \
CLIPS_DIR=$DATA_ROOT/self_collected_clips \
CAPTION_MODEL=gemini-2.5-pro \
bash scripts/pano_caption/run_all.sh
```

What this does, stage by stage:

| Stage | Script | Output |
|---|---|---|
| Cut | `scripts/pano_caption/cut_clips.py` | `$CLIPS_DIR/<scene>__<source_stem>__clip<NNN>.mp4` (5 s each) |
| Caption | `scripts/pano_caption/caption_with_gemini.py` | `<basename>.txt` next to each clip |
| Register | `prepare_pano_data.py --datasets self` | symlinks + `train.csv` / `val.csv` |

Full step-by-step (manual) variant:

```bash
# Stage 1 — cut only
python -m scripts.pano_caption.cut_clips \
    --source_root $DATA_ROOT \
    --scenes my_scene1 my_scene2 \
    --output_dir $DATA_ROOT/self_collected_clips \
    --clip_seconds 5

# Stage 2 — caption only (idempotent; skips already-captioned clips)
python -m scripts.pano_caption.caption_with_gemini \
    --clip_dir $DATA_ROOT/self_collected_clips \
    --model gemini-2.5-pro --workers 8

# Stage 3 — register into a cosmos-pano training package
python prepare_pano_data.py \
    --self_collected_dir $DATA_ROOT/self_collected_clips \
    --output_dir         $DATA_ROOT/cosmos_pano_train \
    --datasets self
```

### 2c. Cost estimate (Gemini 2.5 Pro, April 2026 pricing)

| Item | Value |
|---|---|
| 1 source video, ~30 s long | → ~6 clips |
| Per clip: 12 images (~258 tok each) + prompt | ≈ 3300 in tok + 60 out tok |
| Per clip (`$2.00`/M input + `$8`/M output) | ≈ **$0.005** |
| 1 000 clips | ≈ **$5** |
| 6 000 clips (full PanoWorld training set) | ≈ **$30** |

For a quick-and-dirty check use `gemini-2.5-flash` (~5× cheaper, ~$7 total for 6 k clips) — often indistinguishable for scene-level captions.

### 2d. Output layout

```
$DATA_ROOT/cosmos_pano_train/
├── videos/self_<basename>.mp4    # symlinks → ../self_collected_clips/<basename>.mp4
├── metas/self_<basename>.txt     # caption (copied)
├── train.csv                     # scene-aware split (clips from same source video stay together)
└── val.csv
```

This package is the format directly consumed by the training pipeline.

### 2e. Tuning caption quality

The prompt template is in `scripts/pano_caption/caption_with_gemini.py :: CAPTION_PROMPT`. Edit it then re-run with `--skip_existing=False` (or `rm` specific `.txt` files) to regenerate. Useful knobs:

- `--n_timestamps 4` (vs default 3) — better activity recognition
- `--fov_x 105` — slightly overlapping perspective views, more scene context per image
- `--model gemini-2.5-pro` — best quality (default)

---

## 3. Releasing your own dataset

If you produce a dataset you want to share publicly, the recommended structure is:

```
my_dataset/
├── README.md          (origin, license, clip counts, citation)
├── master.csv         (rows: clip_id, video_path, caption_path, annotation_path, ...)
├── clips/             (.mp4 files)
├── captions/          (.txt files, one line per clip)
└── annotations/       (optional .json/.npz with depth, tracks, geometry, ...)
```

`master.csv` should use **relative paths** so the dataset is portable. The PanoWorld inference and eval drivers will resolve them against the master.csv directory.

---

## Setting up rclone

If you've never used rclone with Google Drive:

```bash
# 1. install
curl https://rclone.org/install.sh | sudo bash

# 2. configure (interactive)
rclone config
#   n) > New remote
#   name> gdrive
#   Storage> drive
#   client_id / client_secret: leave blank (uses rclone's defaults)
#   scope: 1 (full access)
#   then follow the URL in a browser to authorize

# 3. sanity check
rclone lsd gdrive:PanoWorld/
```

For headless servers, use `rclone authorize "drive"` on a machine with a browser and paste the resulting token back to the server. Full docs: https://rclone.org/drive/
