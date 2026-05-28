# PanoWorld Model Zoo

Pre-trained PanoWorld weights are distributed via Google Drive as a single tarball. All weights are released under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license) of the underlying Cosmos-Predict2.5 base model.

> **TL;DR**: download `panoworld_main.tar`, untar it into `checkpoints/`, and point inference at `checkpoints/panoworld_main/model_ema_bf16.pt` (~4.6 GB). See [Quick start](../README.md#quick-start) in the top-level README.

## Released checkpoint

| Name | Description | Tarball contents | Total size | Google Drive |
|---|---|---|---|---|
| **`panoworld_main`** | Main PanoWorld model — full geometry-consistency (depth + trajectory) fine-tuning on top of Cosmos-Predict2.5-2B | `panoworld_main/model_ema_bf16.pt` (4.6 GB) <br/> `panoworld_main/model_ema_fp32.pt` (9.2 GB) <br/> `panoworld_main/model.pt` (13.8 GB) | ~26 GB tar | [`panoworld_main.tar`](https://drive.google.com/drive/folders/1Db7O2enPfuugamwd9mdE0IR6facOwVG0) |

For inference use **`model_ema_bf16.pt`** (the EMA weights). The full `model.pt` and `model_ema_fp32.pt` are only needed for numerical-precision experiments and are bundled inside the same tarball as a convenience.

## Download via rclone

Once you have a Google Drive remote configured as `gdrive:` (see [`TOKENS.md`](TOKENS.md#rclone-setup)):

```bash
# Download the tarball next to your repo root
rclone copy gdrive:panoworld_main.tar . \
    --drive-root-folder-id=1Db7O2enPfuugamwd9mdE0IR6facOwVG0 -P

# Untar — creates checkpoints/panoworld_main/{model.pt, model_ema_bf16.pt, model_ema_fp32.pt}
mkdir -p checkpoints
tar -xf panoworld_main.tar -C checkpoints/ && rm panoworld_main.tar
```

## Manual web download

Open the [release folder](https://drive.google.com/drive/folders/1Db7O2enPfuugamwd9mdE0IR6facOwVG0), right-click `panoworld_main.tar` → Download (~26 GB), then `tar -xf panoworld_main.tar -C checkpoints/`.

## Verifying the checkpoint loaded correctly

```python
import torch
ckpt = torch.load(
    "checkpoints/panoworld_main/model_ema_bf16.pt",
    map_location="cpu",
)
print(list(ckpt.keys())[:10])         # Should list module names
print(sum(p.numel() for p in ckpt.values() if hasattr(p, 'numel')) / 1e9, "B params")
# Expect ~2.1 B params for the 2B base model.
```

## Loading in inference

Just point `--finetune_checkpoint` at the EMA file:

```bash
python generate_pano.py --finetune_checkpoint \
    checkpoints/panoworld_main/model_ema_bf16.pt \
    ... other flags ...
```

## Supporting weights you also need

PanoWorld inference requires the Cosmos-Predict2.5 **base 2B checkpoint** (the diffusion backbone before our fine-tuning) and the **SigLIP2 image encoder**. Both are auto-downloaded from Hugging Face on first run by `install.sh`. Manual fetch:

- Base WFM: `nvidia/Cosmos-Predict2.5-2B` → `huggingface-cli download nvidia/Cosmos-Predict2.5-2B`
- SigLIP2:  `google/siglip2-so400m-patch14-384`
- Cosmos VAE tokenizer: bundled inside the base WFM repo

You'll need a Hugging Face account + accept the Cosmos license on the HF model page. See [`TOKENS.md`](TOKENS.md) for `HF_TOKEN` setup.

The **DAP (Depth Anything Panorama)** weights are only needed if you regenerate annotations for your own training data; they are NOT required for plain inference. Download path is auto-handled by `generate_annotations.py`.
