# PanoWorld Chained-Inference Guide

This guide documents the **two-round chained inference pipeline** used in the PanoWorld paper for benchmark evaluation. It explains *why* the pipeline is two rounds, not a single shot, and how to run it end-to-end against the released test set.

For the simpler single-shot path (good enough for casual demos), see [`INFERENCE.md`](INFERENCE.md). This document is what you want for reproducing the paper's numbers.

---

## 0. The pipeline you are running

The chained driver runs the model **twice** per clip:

```
pers_crop ─Round 1 (V3 pers→pano, FOV-locked)─▶  ERP video
                                                    │
                                                 frame[0]  (only the FIRST frame!)
                                                    ▼
              ─Round 2 (image-init V2W, no spatial mask)─▶  EVAL video.mp4
```

Three design choices are baked into the chained driver — do not bypass them:

1. **Round 2 takes only `frame[0]` of Round 1, not the last frame.**
   The last frame inherited Round 1's FOV-locked spatial mask conditioning and is effectively static around the input FOV. Chaining from it produces "frozen-FOV" outputs. Using `frame[0]` lets Round 2 freely synthesize motion across all 360°.

2. **Round 2 is plain image-conditioned V2W (no spatial-mask blending).**
   We removed the per-step blend `g_mask * g_img + (1 - g_mask) * latents` that Round 1 uses, and we do not pass `reference_latent` / `img_context_emb` in Round 2. The input is just `frame[0]` of Round 1 as a single ERP image; diffusion then synthesizes all ~5 s of motion freely.

3. **FOV-orientation fix (already in code).**
   `equi2pers` uses a z=down convention while `pers2equi_simple` uses z=up. The driver vertically flips the perspective tensor before projecting back to ERP in Round 1. The SigLIP image path is *not* flipped (SigLIP was trained on the unflipped version). You do not need to do anything as long as you use `infer_panoworld_chained.py` — it's handled.

---

## 1. Prerequisites

### 1a. A checkpoint

Download the main PanoWorld weights from the [Model Zoo](MODEL_ZOO.md):

```bash
mkdir -p checkpoints/panoworld_main
rclone copy gdrive:PanoWorld/checkpoints/panoworld_main/ \
    checkpoints/panoworld_main/ \
    --transfers 4 --tpslimit 8 -P
# Result: checkpoints/panoworld_main/model_ema_bf16.pt
```

### 1b. The evaluation test set

Set a data root (override at will):

```bash
export PANO_DATA_ROOT=${PANO_DATA_ROOT:-$HOME/pano_video_data}
```

Then sync the test set from Google Drive:

```bash
mkdir -p $PANO_DATA_ROOT/test
rclone copy gdrive:PanoWorld/test/ $PANO_DATA_ROOT/test/ \
    --transfers 4 --tpslimit 8 -P

# Untar the annotation archives that ship as .tar
cd $PANO_DATA_ROOT/test/
[ -f self_iid_annotations.tar  ] && tar -xf self_iid_annotations.tar  && rm self_iid_annotations.tar
[ -f argus_ood_annotations.tar ] && tar -xf argus_ood_annotations.tar && rm argus_ood_annotations.tar
```

Sanity check:

```bash
wc -l $PANOWORLD_TEST_ROOT/master.csv               # 151 (1 header + 150 rows)
ls $PANO_DATA_ROOT/test/self_iid/annotations/   | wc -l   # 50
ls $PANO_DATA_ROOT/test/argus_ood/annotations/  | wc -l   # 71
ls $PANO_DATA_ROOT/test/habitat_ood/annotations/| wc -l   # 51
```

### 1c. Fix paths in `master.csv` if needed

`master.csv` was generated on the authors' host and contains paths like `/home/ubuntu/Le/...`. If your data root differs:

```bash
sed -i "s|/home/ubuntu/Le/pano_video_data|$PANO_DATA_ROOT|g" \
    $PANO_DATA_ROOT/test/master.csv
```

---

## 2. Launch the chained inference run

### 2a. Recommended invocation

```bash
mkdir -p logs eval_results

LOG=logs/panoworld_main_$(date +%Y%m%d_%H%M%S).log
nohup bash -c "
    source activate.sh
    python scripts/build_eval_set/eval/runners/infer_panoworld_chained.py \
        --master  $PANO_DATA_ROOT/test/master.csv \
        --results eval_results \
        --finetune_checkpoint checkpoints/panoworld_main/model_ema_bf16.pt \
        --method_id panoworld_main \
        --round1_reuse_dir '' \
        --scene_first
" > "$LOG" 2>&1 &
echo $! > /tmp/panoworld_main.pid
```

### 2b. Flag-by-flag rationale

| Flag | Value | Why |
|---|---|---|
| `--master` | path to `master.csv` | Indexes the 150 evaluation clips. |
| `--results` | `eval_results/` | Output root. Driver writes `<results>/<method_id>/<clip>/video.mp4`. |
| `--finetune_checkpoint` | `checkpoints/.../model_ema_bf16.pt` | EMA weights. **Do not** use `model.pt`. |
| `--method_id` | `panoworld_main` | Output directory tag. Set to anything unique if you want to run multiple configurations side-by-side. |
| `--round1_reuse_dir ""` | empty string | Disables Round-1 caching so Round 1 is always regenerated with the same checkpoint as Round 2 (reusing a Round 1 from a different model would invalidate the result). |
| `--scene_first` | flag | Re-orders the 150 clips so each scene's first clip is generated before any second clip. Gives you per-scene early signal (~38 scenes seen in the first ~5 hours instead of waiting 20 h). |
| `--num_frames 93` | default | 93 frames @ 16 fps ≈ 5.8 s. Matches training distribution; don't change. |
| `--guidance 7` | default | CFG scale; empirically best for this domain. |
| `--num_steps 35` | default | Diffusion steps. Reducing visibly degrades quality. |
| `--resolution 512,1024` | default | ERP H×W matching training. |
| `--equirect_rope` | default ON | Equirectangular positional embedding. Required to avoid seam artifacts. |
| `--latent_padding_size 2` | default | Circular latent decode at the longitude wrap. Required. |

### 2c. Output layout

```
eval_results/
  └─ panoworld_main/                              ← <method_id> dir
       └─ <clip_id_dir>/
            ├─ video.mp4                          ← Round-2 EVAL video (the one for metrics)
            ├─ run.json                           ← timing + config
            └─ _work/
                 ├─ pers_input.mp4                ← Round-1 input (static-repeat pers crop)
                 ├─ round1.mp4                    ← Round-1 output (FOV-locked ERP)
                 └─ erp_first_frame.png           ← frame[0] of round1.mp4 fed into Round 2
  └─ _logs/panoworld_main/<clip>.log
```

### 2d. Expected runtime

| Phase | Time |
|---|---|
| Setup (model load + SigLIP) | ~100 s |
| Per clip end-to-end (Round 1 + Round 2) | 7–8 min on a single H100/A100 80 GB |
| **Full 150-clip benchmark** | **~20–22 hours** |

---

## 3. Monitor and resume

```bash
# tail latest log
tail -f logs/panoworld_main_*.log | tail

# count finished clips
ls eval_results/panoworld_main/*/video.mp4 2>/dev/null | wc -l

# is the process still alive?
ps -p $(cat /tmp/panoworld_main.pid) -o pid,etime,stat,cmd | head -2
```

### Resume after a crash / kill

The driver has built-in resume: `--skip_existing` is ON by default and skips any clip whose `video.mp4` already exists. Just re-run the same `nohup ...` command and it picks up where it left off.

### Selectively retry a single clip

```bash
python scripts/build_eval_set/eval/runners/infer_panoworld_chained.py \
    ... usual flags ... \
    --only_clips <clip_id_1> <clip_id_2> \
    --no_skip_existing
```

---

## 4. Score the results

Once `eval_results/panoworld_main/` is fully populated, run the metric pipeline:

```bash
python scripts/build_eval_set/eval/run_eval.py \
    --results_root eval_results/ \
    --master $PANO_DATA_ROOT/test/master.csv \
    --methods panoworld_main \
    --output eval_results/scores_panoworld_main.csv
```

See [`scripts/build_eval_set/eval/README.md`](../scripts/build_eval_set/eval/README.md) for the metric definitions (depth consistency, trajectory consistency, distributional metrics, etc).

---

## 5. Common pitfalls

1. **"Frozen-FOV" outputs.** You ran the legacy single-round driver. Make sure you're using `infer_panoworld_chained.py`.

2. **Round-2 output looks worse than `round1.mp4`.** Round 2's spatial blending was reintroduced. Make sure you're using the current `infer_panoworld_chained.py` from this repo.

3. **FOV center vertically flipped in Round 1.** The orientation fix was reverted or you used a stale `pers2equi_simple`. Confirm the line `pers_for_proj = pers_frames_m1p1.to(device).flip(dims=[2])` exists in `run_round1`.

4. **Round 1 reused from a different model's cache.** You forgot `--round1_reuse_dir ""`. Delete the method dir, set `--round1_reuse_dir ""`, rerun.

5. **`master.csv` paths point at `/home/ubuntu/...` but your data root is different.** Apply the `sed -i` from §1c.

6. **OOM during Round 2.** Drop `--num_steps` to 30 (slight quality cost) or run with `--resolution 384,768`. **Do not** change `--num_frames` — that breaks the temporal positional embedding.
