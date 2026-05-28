# Baseline runner setup (run on A100 once before evaluation)

Five baseline runners ship with PanoWorld's evaluation pipeline:

| method_id            | runner                | conda env    | input format                             | native output |
| -------------------- | --------------------- | ------------ | ---------------------------------------- | ------------- |
| `dvd_360`            | `infer_360dvd.py`     | `360dvd`     | text caption only                        | 16f@8fps, 512x1024 |
| `imagine360`         | `infer_imagine360.py` | `imagine360` | static-repeat pers video                 | 64f@8fps, 512x1024 |
| `argus`              | `infer_argus.py`      | `argus`      | static-repeat pers video                 | 25f@8fps, 512x1024 |
| `follow_your_canvas` | `infer_fyc.py`        | `fyc`        | static-repeat pers video                 | 64f@8fps, 512x1024 |
| `omniroam_pers`      | `infer_omniroam.py --stage pers` | `omniroam` | single pers image + fixed forward traj | Preview 81f@16fps, 480x960 → mp4 container fps=30 |
| `omniroam_erp`       | `infer_omniroam.py --stage erp`  | `omniroam` | single ERP image + fixed forward traj  | Preview 81f@16fps, 480x960 → mp4 container fps=30 |

All native outputs are temporally resampled to the common eval grid
(80 frames @ 16 fps = 5.0 s) AND spatially resized to GT resolution
(1024x512) by `metrics/_common.align_pair_to_eval_grid`, so wrappers should
NOT downsample/uplift on their own.

> Note on OmniRoam's `fps=30` container metadata: that's just `imageio.mimsave`
> writing a default container frame rate; the **semantic** fps of the underlying
> Wan-2.1 / Self-Forcing backbone is **16** (`Self-Forcing/wan/configs/shared_config.py:18`
> `sample_fps = 16`). The eval pipeline declares the semantic fps via
> `method_fps_overrides` in `run_eval.py`, so the container metadata is ignored
> downstream.

---

## 1. Conda envs (do this on A100, not on dev box)

```bash
# 360DVD
cd $HOME/Le/360DVD && bash install.sh         # → env "360dvd"

# Imagine360
cd $HOME/Le/Imagine360 && bash install.sh     # → env "imagine360"

# Argus
cd $HOME/Le/argus-code && bash install.sh     # → env "argus"

# Follow-Your-Canvas (no install.sh; manually create)
conda create -n fyc python=3.10 -y && conda activate fyc
cd $HOME/Le/FollowYourCanvas && pip install -r requirements.txt

# OmniRoam
conda create -n omniroam python=3.10 -y && conda activate omniroam
cd $HOME/Le/OmniRoam && pip install -r requirements.txt
```

## 2. Missing checkpoints

### OmniRoam Preview/Refine (REQUIRED before runner can run)

```bash
cd $HOME/Le/OmniRoam
python download_omniroam_models.py
# writes models/OmniRoam/{Preview,Refine}/{preview,refine}.ckpt
```

Currently only `models/Wan-AI/Wan2.1-T2V-1.3B/...` is on disk; without
preview.ckpt + refine.ckpt the runner will fail in stage 1.

### Qwen-VL-Chat — **NO LONGER NEEDED** (skip this download)

Imagine360 and Follow-Your-Canvas use Qwen-VL-Chat (~17 GB) only to
generate per-clip captions from a single frame. Since we already have
high-quality `gpt-5-mini` captions for ALL 150 clips in `master.csv`, we
bypass Qwen entirely. This frees ~17 GB of GPU memory and lets both
methods comfortably fit on a 40 GB A100.

How the bypass works:

1. The `infer_imagine360.py` runner writes the master.csv caption to
   `<video_path>.txt` next to the static-repeated input mp4. Imagine360's
   `inference_dual_p2e.py` already supports this: when a `.txt` file is
   present alongside `<video>.mp4`, it reads the prompt from there and
   sets `prompt_gen_flag = False`. (Lines 528-536 of the upstream file.)

2. However, the upstream code STILL loads Qwen at startup regardless
   (lines 579-580). Patch it once on A100 so Qwen is only loaded if
   actually needed:

```bash
# Imagine360 patch — wrap LMM loading in `if prompt_gen_flag:`
cd $HOME/Le/Imagine360
python - <<'PY'
import re, pathlib
p = pathlib.Path("inference_dual_p2e.py")
s = p.read_text()
old = ("        lmm_tokenizer = AutoTokenizer.from_pretrained(lmm_path, trust_remote_code=True)\n"
       "        lmm_model = AutoModelForCausalLM.from_pretrained(lmm_path, device_map=\"cuda\", trust_remote_code=True).eval()\n"
       "        lmm_model.requires_grad_(False)\n"
       "        # Generate input prompt\n"
       "        if prompt_gen_flag:\n"
       "            prompt = get_prompt(orig_pixel_values[4, :,:,:], lmm_tokenizer, lmm_model)\n"
       "            print(f\"[INFO] Prompt generate from llm: {prompt}\")\n"
       "\n"
       "\n"
       "        del lmm_tokenizer\n"
       "        del lmm_model\n")
new = ("        # PanoWorld-eval patch: only load LMM when really needed\n"
       "        if prompt_gen_flag:\n"
       "            lmm_tokenizer = AutoTokenizer.from_pretrained(lmm_path, trust_remote_code=True)\n"
       "            lmm_model = AutoModelForCausalLM.from_pretrained(lmm_path, device_map=\"cuda\", trust_remote_code=True).eval()\n"
       "            lmm_model.requires_grad_(False)\n"
       "            prompt = get_prompt(orig_pixel_values[4, :,:,:], lmm_tokenizer, lmm_model)\n"
       "            print(f\"[INFO] Prompt generate from llm: {prompt}\")\n"
       "            del lmm_tokenizer\n"
       "            del lmm_model\n")
assert old in s, "Imagine360 patch anchor not found; upstream changed"
p.write_text(s.replace(old, new))
print("Imagine360 LMM bypass patch applied.")
PY
```

```bash
# FYC patch — wrap LMM loading in `if not prompts_input:`
cd $HOME/Le/FollowYourCanvas
python - <<'PY'
import pathlib
p = pathlib.Path("inference_outpainting-dir-with-prompt.py")
s = p.read_text()
old = ("    lmm_tokenizer = AutoTokenizer.from_pretrained(lmm_path, trust_remote_code=True)\n"
       "    lmm_model = AutoModelForCausalLM.from_pretrained(lmm_path, device_map=\"cuda\", trust_remote_code=True).eval()\n")
new = ("    # PanoWorld-eval patch: only load Qwen-VL-Chat when needed.\n"
       "    if prompts_input is None or len(prompts_input) == 0:\n"
       "        lmm_tokenizer = AutoTokenizer.from_pretrained(lmm_path, trust_remote_code=True)\n"
       "        lmm_model = AutoModelForCausalLM.from_pretrained(lmm_path, device_map=\"cuda\", trust_remote_code=True).eval()\n"
       "    else:\n"
       "        lmm_tokenizer = None\n"
       "        lmm_model = None\n")
assert old in s, "FYC patch anchor not found; upstream changed"
p.write_text(s.replace(old, new))
print("FYC LMM bypass patch applied.")
PY
```

After patching, both runners will use the master.csv caption directly
and skip Qwen, fitting comfortably on A100 40 GB.

### Imagine360 cache symlinks (REQUIRED — its yaml hardcodes paths)

```bash
mkdir -p ~/.cache/imagine360
ln -sf $HOME/Le/Imagine360/_ckpt/imagine360_checkpoints \
       ~/.cache/imagine360/imagine360_checkpoints
ln -sf $HOME/Le/_baseline_shared/sam/sam_vit_b_01ec64.pth \
       ~/.cache/imagine360/sam_vit_b_01ec64.pth
```

### Argus auto-downloads on first run

* `naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric` (~5 GB) — pulled by HF API
* `stabilityai/stable-video-diffusion-img2vid` (~10 GB)

If the A100 has restricted network, pre-pull both to HF cache.

### 360DVD's StableDiffusion-1.5 base

The 360DVD installer drops a `ckpts/StableDiffusion/` symlink/folder; verify
it points to a SD-1.5 snapshot. If empty, run:

```bash
huggingface-cli download runwayml/stable-diffusion-v1-5 \
  --local-dir $HOME/Le/360DVD/ckpts/StableDiffusion
```

### FYC's SD-2.1 base

FYC needs SD-2.1 (not 2.1-base). The runner currently points to
`models--Manojb--stable-diffusion-2-1-base/...` which is what we
already have on disk. If output quality is poor, swap to the official
`stabilityai/stable-diffusion-2-1` mirror via `--sd21_path`.

## 3. One-clip GPU smoke test (recommended order)

```bash
# Run from cosmos-predict2.5/ root, env "cosmos" for the wrapper itself
RES=$PANO_DATA_ROOT/eval_results

# 360DVD: text-only, smallest dependency surface
python -m test_set_pkg.eval.runners.infer_360dvd \
    --results $RES --conda_env 360dvd \
    --splits self_iid --limit 1

# Argus: pers→pano, lightest pano model
python -m test_set_pkg.eval.runners.infer_argus \
    --results $RES --conda_env argus \
    --splits self_iid --limit 1

# Imagine360: pers→pano, two-UNet pipeline
python -m test_set_pkg.eval.runners.infer_imagine360 \
    --results $RES --conda_env imagine360 \
    --splits self_iid --limit 1

# Follow-Your-Canvas: outpaint into ERP canvas
python -m test_set_pkg.eval.runners.infer_fyc \
    --results $RES --conda_env fyc \
    --splits self_iid --limit 1

# OmniRoam (only after Preview/Refine ckpts download)
python -m test_set_pkg.eval.runners.infer_omniroam --stage pers \
    --results $RES --conda_env omniroam \
    --splits self_iid --limit 1
python -m test_set_pkg.eval.runners.infer_omniroam --stage erp \
    --results $RES --conda_env omniroam \
    --splits self_iid --limit 1
```

After each smoke test, verify
`results/<method_id>/<split>__<clip_id>/video.mp4` exists and plays.

## 4. Known fragility / TODOs

* **OmniRoam refine** assumes 81 frames input; if Preview emits fewer the
  refine stage interpolates internally — should be fine.
* **FYC** is not panorama-aware. Outpainting to a 2:1 canvas approximates
  ERP but does NOT enforce horizontal seam continuity. Expect artifacts at
  the +/-180° wrap; this is an honest baseline limitation, not a bug.
* **Imagine360** sometimes writes outputs into a `samples-XXXX` timestamped
  subfolder. The runner's `find_first_mp4` walks recursively but excludes
  filenames containing `input/pers/grid` — verify on first run.
* **Argus**'s output filename mirrors the input filename (`input.mp4`).
  The runner just picks the first non-input mp4; if Argus actually
  overwrites the same name, switch to its `*_pano.mp4` convention by
  inspecting the log.
