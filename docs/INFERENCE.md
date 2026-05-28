# PanoWorld Inference — Command Reference

This file is the **flag-by-flag reference** for `generate_pano.py`, the single-shot panoramic generator. For the **two-round chained eval pipeline** (used in the paper for benchmark comparisons) see [`PANOWORLD_INFERENCE_GUIDE.md`](PANOWORLD_INFERENCE_GUIDE.md).

---

## 1. Minimal command

```bash
python generate_pano.py \
  --input_path  <path/to/input_pers.png|.mp4> \
  --prompt      "<short perspective-view description>" \
  --pano_prompt "<full 360° scene description (NO PEOPLE — see Rules)>" \
  --output_dir  output/<run_name> \
  --finetune_checkpoint checkpoints/panoworld_main/model_ema_bf16.pt \
  --resolution 512,1024 --num_frames 93 --guidance 7 --num_steps 35 --seed 42 \
  --equirect_rope \
  --num_input_frames 1 --fov_x 90.0 --yaw 0.0 --pitch 0.0 --roll 0.0 \
  --i2v_resolution 480,640 \
  --use_clip --v3 --disable_guardrails --offload_diffusion_model
```

Output:
- `output/<run_name>/pano_v3_512x1024_equirect_s42.mp4` — the panoramic video.
- `output/<run_name>/pano_v3_512x1024_equirect_s42_concat.mp4` — side-by-side input + output for quick inspection.
- `output/<run_name>/run.json` — full configuration log.

---

## 2. Five hard rules (do not skip)

1. **`pano_prompt` MUST NOT mention people.** The person is supplied via SigLIP2 conditioning on the input perspective image. Mentioning a person in the prompt causes duplicates (e.g. two people walking).
2. **`--num_frames 93` for ~5 s, `--num_frames 33` for ~2 s.** State-token length is derived from this; other values break the temporal positional embedding.
3. **`--equirect_rope` is required.** Without it, the ceiling and floor poles distort heavily.
4. **`--i2v_resolution 480,640` matches training.** Changing it degrades quality.
5. **Use `model_ema_bf16.pt`, NOT `model.pt`.** The EMA weights are what we report numbers on.

---

## 3. Flag reference

### Inputs
| Flag | Default | Meaning |
|---|---|---|
| `--input_path` | (req) | Perspective image `.png/.jpg` or a perspective video `.mp4`. For images, `--num_input_frames 1`. |
| `--prompt` | (req) | Short perspective-view description used to bias the diffusion. |
| `--pano_prompt` | (req) | Full 360° equirectangular scene description. **No people.** |
| `--negative_prompt` | a long default | Standard quality negative. Override only if you need to. |
| `--num_input_frames` | `1` | How many frames from `--input_path` to feed as ERP conditioning. For static images this is always 1. |

### Diffusion settings
| Flag | Default | Meaning |
|---|---|---|
| `--resolution` | `512,1024` | ERP `H,W`. Match training. `384,768` works at lower quality if OOM. |
| `--num_frames` | `93` | 93 → ~5.8 s at 16 fps. 33 → ~2 s. |
| `--guidance` | `7` | CFG scale. Higher → more prompt-adherent, less diversity. |
| `--num_steps` | `35` | Diffusion steps. Don't reduce below 30 (visible quality drop). |
| `--seed` | `42` | RNG seed. |

### Geometry / projection
| Flag | Default | Meaning |
|---|---|---|
| `--equirect_rope` | **OFF** — turn ON | Use equirectangular positional embedding. **Required.** |
| `--fov_x` | `90.0` | Horizontal FOV of the input perspective frame. |
| `--yaw` | `0.0` | Yaw angle at which the input pers is placed in the panorama. |
| `--pitch` | `0.0` | Pitch. |
| `--roll` | `0.0` | Roll. |
| `--i2v_resolution` | `480,640` | Pers-input resolution before being projected to ERP. **Don't change.** |

### Model & runtime
| Flag | Default | Meaning |
|---|---|---|
| `--finetune_checkpoint` | (req) | Path to the trained EMA weights. See [Model Zoo](MODEL_ZOO.md). |
| `--use_clip` | **OFF** — turn ON | Use CLIP text encoder branch. **Required.** |
| `--v3` | **OFF** — turn ON | Use v3 pers→pano projection (FOV-locked, fixed). |
| `--disable_guardrails` | OFF | Skip NVIDIA's prompt guardrails. Recommended for research use. |
| `--offload_diffusion_model` | OFF | Offload UNet weights to CPU when not in use. Needed on <80 GB GPUs. |

---

## 4. Common scenarios

### 4a. 5 s lobby demo (the headline command)

```bash
python generate_pano.py \
  --input_path assets/panotest/lobby.png \
  --prompt "A modern apartment lobby with warm lighting, wooden panels, checkered tile floor, a person walking across." \
  --pano_prompt "A 360-degree equirectangular view of a modern apartment lobby with warm ambient lighting, wooden panels, checkered tile floor, and a central glass entrance door." \
  --output_dir output/lobby_5s \
  --finetune_checkpoint checkpoints/panoworld_main/model_ema_bf16.pt \
  --resolution 512,1024 --num_frames 93 --guidance 7 --num_steps 35 --seed 42 \
  --equirect_rope --num_input_frames 1 \
  --fov_x 90.0 --yaw 0.0 --pitch 0.0 --roll 0.0 \
  --i2v_resolution 480,640 \
  --use_clip --v3 --disable_guardrails --offload_diffusion_model
```

### 4b. 2 s quick smoke test

Swap `--num_frames 93` → `--num_frames 33`; run-time drops from ~7 min to ~3 min.

### 4c. Different scene

Replace `--input_path`, `--prompt`, `--pano_prompt`, and `--output_dir`. Prompt files for the bundled demos live under `assets/panotest/<scene>.txt`.

### 4d. Long generation (>5 s) via autoregressive rounds

The single-shot path supports at most ~93 frames (~5.8 s) due to the positional-embedding window. For longer videos, use the chained driver in [`PANOWORLD_INFERENCE_GUIDE.md`](PANOWORLD_INFERENCE_GUIDE.md) which iteratively re-seeds from the last generated frame.

---

## 5. Checkpoint conversion (if you trained your own)

The training pipeline writes distributed-format checkpoints. Convert to a single `.pt` with:

```bash
python scripts/convert_distcp_to_pt.py \
    <path/to/distcp_model_dir> \
    checkpoints/<run_name>/iter_<N>/
```

This produces `model.pt`, `model_ema_bf16.pt` and `model_ema_fp32.pt` in the output dir. Always use `model_ema_bf16.pt` for inference.

---

## 6. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Floor / ceiling distorted | forgot `--equirect_rope` | add the flag |
| Two people in output | mentioned the person in `--pano_prompt` | remove person from `pano_prompt` (keep in `--prompt`) |
| Output too short / wrong length | wrong `--num_frames` | use 93 (5 s) or 33 (2 s) |
| OOM during decode | resolution too high | drop to `--resolution 384,768` or add `--offload_diffusion_model` |
| Output looks generic / not aligned with input | wrong checkpoint or wrong `--use_clip / --v3` flags | re-check flags 4–5 of Section 2 |
| Seam visible at longitude wrap | `--equirect_rope` missing OR `--latent_padding_size` < 2 | use `--equirect_rope` and default `--latent_padding_size 2` |

Full pipeline-level pitfalls: see Section 5 of [`PANOWORLD_INFERENCE_GUIDE.md`](PANOWORLD_INFERENCE_GUIDE.md).
