<div align="center">

# PanoWorld: Geometry-Consistent Panoramic Video World Modeling

[![arXiv](https://img.shields.io/badge/arXiv-2605.15391-b31b1b.svg)](https://arxiv.org/abs/2605.15391)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://zzzura-secure.duckdns.org/pano)
[![Release](https://img.shields.io/badge/Release-Google_Drive-4285F4.svg)](https://drive.google.com/drive/folders/1Db7O2enPfuugamwd9mdE0IR6facOwVG0)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

**Le Jiang**<sup>1</sup> · **Xiangyu Bai**<sup>1</sup> · **Bishoy Galoaa**<sup>1</sup> · **Shayda Moezzi**<sup>1</sup> · **Caleb James Lee**<sup>1</sup> · **Tooba Imtiaz**<sup>1</sup> · **Edmund Yeh**<sup>1</sup> · **Jennifer Dy**<sup>1</sup> · **Yanzhi Wang**<sup>1</sup> · **Sarah Ostadabbas**<sup>1</sup>

<sup>1</sup>Northeastern University

</div>

---

PanoWorld is a **panoramic (360° equirectangular) video world model** that takes a single perspective image plus a text prompt and generates a temporally consistent, **geometry-consistent** panoramic video. Existing panoramic video methods optimize primarily for visual realism and do not explicitly constrain the underlying 3D scene state, producing outputs that appear plausible yet exhibit inconsistent depth, broken correspondences, and implausible motion across the spherical surface. PanoWorld treats panoramic video generation as a geometry- and dynamics-consistent latent-state modeling problem rather than pure visual synthesis.

> **TL;DR.** A single perspective image + a 360° caption → 5 seconds of full-sphere video, with depth, trajectories, and pole regions that stay self-consistent.

<div align="center">

🎬 **A live demo and qualitative gallery is on the [project page](https://zzzura-secure.duckdns.org/pano).**

</div>

---

## ✨ Highlights

- 🌐 **Equirectangular-aware position encoding.** Position embeddings are made spherically continuous so the model treats the left/right boundary as the same column and respects pole compression — no more vertical seams or pole tearing.
- 📐 **Depth supervision.** A panorama-adapted Depth Anything V2 backbone (DAP) provides per-frame depth used during fine-tuning, suppressing depth flicker that pure-pixel objectives never see.
- 🎯 **Motion supervision.** Point trajectories regularize how the scene moves through time. The result: people, vehicles, and dynamic foreground objects follow physically plausible paths across the full sphere.
- 🔁 **Two-round chained inference.** Round 1 is FOV-locked so the input perspective stays anchored; Round 2 takes only the first frame of Round 1 and synthesizes free 360° motion. Together they avoid the "frozen-FOV" failure mode of naive single-shot ERP generation.
- 🎬 **PanoGeo dataset.** ~5 k panoramic clips ( WEB360 + 360x_HR + self-collected ) with auto-generated 4-view Gemini captions, depth, and 2D/3D point tracks — the full data pipeline ships in this repo so you can build your own.

---

## 📋 Release status

| Asset                                      | Status                                                                                              |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------- |
| 🌐 Project page                            | [zzzura-secure.duckdns.org/pano](https://zzzura-secure.duckdns.org/pano) — live                     |
| 📄 arXiv preprint                          | [arXiv:2605.15391](https://arxiv.org/abs/2605.15391) — live                                         |
| 🛠️ Inference code                          | this repo — live                                                                                    |
| 🤖 Pre-trained checkpoint                  | [Google Drive](https://drive.google.com/drive/folders/1Db7O2enPfuugamwd9mdE0IR6facOwVG0) — live ([Model Zoo](docs/MODEL_ZOO.md))    |
| 🧪 Evaluation test set (150 clips)         | [Google Drive](https://drive.google.com/drive/folders/1Db7O2enPfuugamwd9mdE0IR6facOwVG0) — live ([docs/DATASET.md](docs/DATASET.md)) |
| 🎬 PanoGeo training set (videos + annotations) | uploading soon                                                                                  |
| 🏋️ Training code                           | uploading soon                                                                                      |
| 🤗 Hugging Face mirror of checkpoint       | planned                                                                                             |

---

## 🤖 Models

PanoWorld releases a single end-to-end checkpoint built on top of NVIDIA Cosmos-Predict2.5 (2B).

| Name             | Description                                                                                                                                        | Files                                                                                       | Download                                                                                       |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **panoworld_main** | Main PanoWorld model — geometry-aware fine-tuning of Cosmos-Predict2.5-2B for 360° equirectangular video generation.        | `model_ema_bf16.pt` (4.6 GB) — used for inference <br/> `model_ema_fp32.pt` (9.2 GB) <br/> `model.pt` (13.8 GB) | [`panoworld_main.tar` on Google Drive](https://drive.google.com/drive/folders/1Db7O2enPfuugamwd9mdE0IR6facOwVG0) (26 GB) |

Drop the EMA file at `checkpoints/panoworld_main/model_ema_bf16.pt` and pass it to `--finetune_checkpoint`. Full details in [Model Zoo](docs/MODEL_ZOO.md).

---

## ▶️ Quick start

```bash
# 1. Clone
git clone https://github.com/<your-user>/PanoWorld.git
cd PanoWorld

# 2. Install (creates the cosmos conda env, installs cu128 torch + flash-attn + deps)
bash install.sh
source activate.sh

# 3. Download the pre-trained checkpoint tarball (~26 GB) from Google Drive
#    Folder: https://drive.google.com/drive/folders/1Db7O2enPfuugamwd9mdE0IR6facOwVG0
#    Either click "Download" in the web UI on `panoworld_main.tar`, or use rclone:
mkdir -p checkpoints
rclone copy gdrive:panoworld_main.tar . \
    --drive-root-folder-id=1Db7O2enPfuugamwd9mdE0IR6facOwVG0 -P
tar -xf panoworld_main.tar -C checkpoints/ && rm panoworld_main.tar
# → checkpoints/panoworld_main/{model.pt, model_ema_bf16.pt, model_ema_fp32.pt}
# (only model_ema_bf16.pt is required for inference; the other two are bundled
#  for fp32-precision and resume-from-state experiments.)

# 4. Generate a 5 s 360° panoramic video from a perspective image
python generate_pano.py \
  --input_path assets/panotest/lobby.png \
  --prompt      "$(cat assets/panotest/lobby.txt)" \
  --pano_prompt "A 360-degree equirectangular view of a modern apartment lobby with warm ambient lighting, wooden panels, checkered tile floor, and a central glass entrance door." \
  --output_dir  output/lobby_demo \
  --finetune_checkpoint checkpoints/panoworld_main/model_ema_bf16.pt \
  --resolution 512,1024 --num_frames 93 --guidance 7 --num_steps 35 --seed 42 \
  --equirect_rope --num_input_frames 1 \
  --fov_x 90.0 --yaw 0.0 --pitch 0.0 --roll 0.0 --i2v_resolution 480,640 \
  --use_clip --v3 --disable_guardrails --offload_diffusion_model
```

Result: `output/lobby_demo/pano_v3_512x1024_equirect_s42.mp4` (5 s ERP video) and a `*_concat.mp4` side-by-side input+output preview.

> **Note.** Requires a single H100 or A100 80 GB GPU. Inference is ~7 min per 5 s clip with `--offload_diffusion_model`.

Full flag reference: [`docs/INFERENCE.md`](docs/INFERENCE.md). Paper-quality chained inference: [`docs/PANOWORLD_INFERENCE_GUIDE.md`](docs/PANOWORLD_INFERENCE_GUIDE.md).

---

## ⚙️ Setup

`bash install.sh` does the following (everything is idempotent, re-run to top up):

1. installs system deps (`git-lfs`, `ffmpeg`, `curl`, `wget`)
2. creates a conda env named `cosmos` (Python 3.10 — required by flash-attn)
3. installs `uv`, then `uv pip install -e ".[cu128]"` (PyTorch 2.7 + CUDA 12.8 + flash-attn 2.7.3)
4. installs the DAP geometry-annotation extras (CoTracker3, opencv-headless)
5. configures the Hugging Face token (see [`docs/TOKENS.md`](docs/TOKENS.md))
6. installs rclone (used to download the test set + checkpoints from Google Drive)

After `install.sh` finishes, separately fetch:

- the **pre-trained checkpoint** (~4.6 GB) — see [`docs/MODEL_ZOO.md`](docs/MODEL_ZOO.md)
- the **evaluation test set** (150 clips, ~14 GB) — see [`docs/DATASET.md`](docs/DATASET.md)

### 📁 Repository layout

```
PanoWorld/
├── README.md                       ← you are here
├── install.sh / activate.sh        ← env bootstrap
├── pyproject.toml + uv.lock        ← deps (uv-managed)
├── Dockerfile                      ← container build
│
├── cosmos_predict2/                ← core model code (kept namespaced for ckpt-compat)
├── packages/                       ← cosmos-cuda / cosmos-oss sub-packages
├── DAP/                            ← Depth Anything Panorama (depth backbone)
│
├── generate_pano.py                ← MAIN PanoWorld inference entry-point
├── generate_annotations.py         ← geometry annotation (depth + tracks)
├── prepare_pano_data.py            ← dataset builder (public + self-collected)
├── eval_extension.py               ← evaluation utilities
│
├── scripts/
│   ├── pano_caption/               ← Gemini caption pipeline (cut + caption + register)
│   └── build_eval_set/             ← evaluation harness + habitat synthetic renderer
│
├── assets/                         ← demo prompts + small input images/videos
└── docs/
    ├── setup.md                    ← env setup
    ├── INFERENCE.md                ← inference flags reference
    ├── PANOWORLD_INFERENCE_GUIDE.md← chained-inference (paper eval) guide
    ├── MODEL_ZOO.md                ← pre-trained model downloads
    ├── DATASET.md                  ← data preparation
    └── TOKENS.md                   ← API token setup
```

---

## 📂 Input format

PanoWorld takes three inputs per clip:

| Input            | Format                                                                                                                 |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `--input_path`   | Perspective image `.png/.jpg` (preferred) or a perspective video `.mp4` with `--num_input_frames > 1`                  |
| `--prompt`       | Short, perspective-style description (used to bias the SigLIP2 conditioning branch)                                    |
| `--pano_prompt`  | Full 360° equirectangular scene description. **Do NOT mention people** — the person is supplied via the input image    |

```json
{
  "prompt":      "A modern apartment lobby with warm lighting, wooden panels, checkered tile floor, a person walking across.",
  "pano_prompt": "A 360-degree equirectangular view of a modern apartment lobby with warm ambient lighting, symmetrical wooden panels, and a central glass entrance door."
}
```

Bundled prompt files for the demo scenes live in `assets/panotest/<scene>.txt`. Five hard rules for prompt construction are listed in [`docs/INFERENCE.md`](docs/INFERENCE.md#2-five-hard-rules-do-not-skip).

---

## 🚀 Pipeline

PanoWorld supports two inference modes:

### ✨ Single-shot — `generate_pano.py`

Generates 5 s ERP video in one forward pass. Best for casual generation and demos. Command above (Quick start §4).

### 🔁 Two-round chained inference — `infer_panoworld_chained.py`

Used in the paper to generate evaluation outputs. Runs the model twice per clip:

```
pers_crop ─Round 1 (FOV-locked ERP-init)─▶  ERP video ─frame[0]─▶  Round 2 (free V2W) ─▶ EVAL video
```

Round 1 keeps the input FOV locked so the perspective input has a stable anchor, then Round 2 takes only the *first* frame of Round 1 and synthesizes full 360° motion freely. This avoids the "frozen-FOV" failure of naive single-shot ERP generation. Full walkthrough: [`docs/PANOWORLD_INFERENCE_GUIDE.md`](docs/PANOWORLD_INFERENCE_GUIDE.md).

### 🧪 Evaluation harness — `scripts/build_eval_set/`

End-to-end pipeline to reproduce the paper's metrics on our 150-clip benchmark
(50 self_iid + 50 argus_ood + 50 habitat_ood):

#### Step 1 — Download the test set (~14 GB)

```bash
# Configure rclone once (rclone config → create remote named "gdrive")
export PANOWORLD_TEST_ROOT=$HOME/panoworld_test_set
mkdir -p "$PANOWORLD_TEST_ROOT" && cd "$PANOWORLD_TEST_ROOT"

# Pull the single tarball from the release folder
rclone copy gdrive:panoworld_testset_150.tar . \
    --drive-root-folder-id=1Db7O2enPfuugamwd9mdE0IR6facOwVG0 -P

# Unpack (creates self_iid/, argus_ood/, habitat_ood/, master.csv, README.md)
tar -xf panoworld_testset_150.tar && rm panoworld_testset_150.tar

# Inside, two per-split annotation bundles still need to be expanded:
tar -xf self_iid/annotations.tar           -C self_iid/
tar -xf argus_ood/annotations.tar          -C argus_ood/
tar -xf argus_ood/camera_trajectories.tar  -C argus_ood/
# habitat_ood/annotations/ is already an unpacked directory tree
```

After this `$PANOWORLD_TEST_ROOT/master.csv` indexes all 150 clips with **relative**
paths to `videos/`, `annotations/`, `captions/`. Full schema + coverage stats are in
[`docs/DATASET.md`](docs/DATASET.md).

#### Step 2 — Generate PanoWorld outputs on all 150 clips (~20 h on H100)

```bash
cd /path/to/PanoWorld
python scripts/build_eval_set/eval/runners/infer_panoworld_chained.py \
    --master  $PANOWORLD_TEST_ROOT/master.csv \
    --results eval_results \
    --finetune_checkpoint checkpoints/panoworld_main/model_ema_bf16.pt \
    --method_id panoworld_main \
    --round1_reuse_dir "" \
    --scene_first
```

This writes one mp4 per clip to `eval_results/panoworld_main/<clip_id>.mp4`. The
two-round chained pipeline (Round 1 FOV-locked → Round 2 free V2W) is documented
in [`docs/PANOWORLD_INFERENCE_GUIDE.md`](docs/PANOWORLD_INFERENCE_GUIDE.md).

#### Step 3 — Score (visual + geometry metrics)

```bash
python scripts/build_eval_set/eval/run_eval.py \
    --results_root eval_results/ \
    --master       $PANOWORLD_TEST_ROOT/master.csv \
    --methods      panoworld_main \
    --output       eval_results/scores.csv
```

`scores.csv` reports per-clip FVD, FID, CLIP-T, LPIPS, depth AbsRel / δ<1.25,
camera ATE / RPE, and 3D-track consistency. Aggregate to a paper-style table
with:

```bash
python scripts/build_eval_set/eval/aggregate.py \
    --results_long eval_results/scores.csv \
    --output_dir   eval_results/paper_tables/
```

#### (Optional) Step 4 — Compare against baselines

Inference runners for **Argus**, **360DVD**, **Follow-Your-Canvas**,
**Imagine360**, and **OmniRoam** are bundled under
`scripts/build_eval_set/eval/runners/`. Run any of them on the same 150 clips,
then pass `--methods panoworld_main argus dvd_360 follow_your_canvas_merged imagine360 omniroam_pers`
to `run_eval.py` to score them side-by-side. Baseline checkpoint locations are
in [`scripts/build_eval_set/eval/runners/BASELINE_SETUP.md`](scripts/build_eval_set/eval/runners/BASELINE_SETUP.md).

---

## 🎬 Build your own panoramic dataset

PanoWorld ships with the same captioning + annotation pipeline used to build **PanoGeo** (our paper's training corpus). Use it to caption and annotate your own 360° clips:

```bash
# 1. Configure your own Gemini API key (see docs/TOKENS.md)
export GEMINI_API_KEY="<your-key>"

# 2. End-to-end: cut long videos → 5 s clips → 4-view caption → register
bash scripts/pano_caption/run_all.sh

# 3. Build a training package from the captioned clips
python prepare_pano_data.py \
  --self_collected_dir /path/to/your/clips \
  --output_dir         /path/to/output_train_pkg \
  --datasets self

# 4. Generate depth + 2D/3D tracks + static masks
python generate_annotations.py --resume
```

Cost estimate: ~$5 USD of Gemini 2.5 Pro per 1 000 captioned clips. Full walkthrough in [`docs/DATASET.md`](docs/DATASET.md).

---

## 🙏 Acknowledgements

PanoWorld builds on top of several open-source projects. We thank the authors and contributors of:

- [NVIDIA Cosmos-Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) — base world-foundation-model architecture.
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) + our equirectangular adaptation (DAP) — depth backbone used to supervise the geometry head.
- [CoTracker3](https://github.com/facebookresearch/co-tracker) — point trajectory supervision.
- [SigLIP2](https://huggingface.co/google/siglip2-so400m-patch14-384) — perspective image conditioning encoder.

See [ATTRIBUTIONS.md](ATTRIBUTIONS.md) for the full third-party software list.

---

## 📄 Citation

If you find PanoWorld useful in your research, please consider citing:

```bibtex
@article{jiang2026panoworld,
  title   = {PanoWorld: Geometry-Consistent Panoramic Video World Modeling},
  author  = {Jiang, Le and Bai, Xiangyu and Galoaa, Bishoy and Moezzi, Shayda and
             Lee, Caleb James and Imtiaz, Tooba and Yeh, Edmund and Dy, Jennifer and
             Wang, Yanzhi and Ostadabbas, Sarah},
  journal = {arXiv preprint arXiv:2605.15391},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.15391}
}
```

Please also cite the underlying Cosmos-Predict2.5 paper:

```bibtex
@article{cosmos_predict25_2025,
  title   = {Cosmos-Predict2.5},
  author  = {NVIDIA Cosmos Team},
  journal = {arXiv preprint arXiv:2511.00062},
  year    = {2025}
}
```

---

## 📜 License

Code released under the [Apache 2.0 License](LICENSE) (inherited from NVIDIA Cosmos-Predict2.5). Pre-trained PanoWorld checkpoints follow the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license) of the underlying base model.
