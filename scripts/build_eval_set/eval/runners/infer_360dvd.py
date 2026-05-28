#!/usr/bin/env python3
"""360DVD inference runner (text-only baseline).

360DVD takes ONLY a caption (no input frame, no input video) and emits a
panoramic ERP video using a pre-trained AnimateDiff motion module + a
360Adapter. Native output: 16 frames @ 8 fps = 2.0s, resolution 512x1024.

Setup (run on A100 once):
    cd $HOME/Le/360DVD
    bash install.sh                 # creates conda env `360dvd`
    # ckpts already downloaded under ckpts/{Motion_Module,360Adapter,
    # DreamBooth_LoRA,StableDiffusion} per the original README

Per-clip flow:
    For each clip in master.csv:
      caption  --[360DVD: animate.py]-->  pano video.mp4
    The runner writes a per-clip yaml override that injects the caption,
    invokes scripts.animate as a subprocess inside `360dvd` conda env,
    and moves the produced mp4 into the standard results layout.

Output:
    results_dir/dvd_360/<clip_id_dir>/video.mp4   (16f @ 8fps, 512x1024)

Usage:
    python -m test_set_pkg.eval.runners.infer_360dvd \\
        --results $PANO_DATA_ROOT/eval_results \\
        --conda_env 360dvd \\
        --splits self_iid argus_ood habitat_ood --limit 1
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _common import (  # noqa: E402
    MASTER_DEFAULT,
    have_output,
    load_caption_with_style,
    normalize_output_video,
    per_clip_outdir,
    read_master,
    run_subprocess,
    select_clips,
    wrap_with_conda,
    write_run_json,
)

DVD360_REPO = Path("$HOME/Le/360DVD")
DEFAULT_FLOW = DVD360_REPO / "__assets__/example_flows/100541.mp4"

METHOD_ID = "dvd_360"


def _find_latest_output(root: Path, suffixes: tuple = (".mp4", ".gif"),
                        exclude_substrings: tuple = ()) -> Path | None:
    """Recursively walk `root`, return the most recently-modified file whose
    suffix is in `suffixes` and whose name contains none of
    `exclude_substrings`. Used because 360DVD's `animate.py` writes a fresh
    timestamp-named subfolder under `samples/` per run, and emits .gif (not
    .mp4) so the generic find_first_mp4 helper alone is insufficient.
    """
    if not root.is_dir():
        return None
    cands: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in suffixes:
            continue
        nm = p.name.lower()
        if any(s.lower() in nm for s in exclude_substrings):
            continue
        if p.stat().st_size <= 0:
            continue
        cands.append(p)
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _gif_to_mp4(src_gif: Path, dst_mp4: Path, fps: int = 8) -> Path:
    """Transcode .gif -> .mp4 (libx264, yuv420p, even dims) via ffmpeg.
    360DVD writes its samples as animated gifs; downstream metrics expect
    mp4, so we convert here."""
    import subprocess as _sp
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    if dst_mp4.exists():
        dst_mp4.unlink()
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src_gif),
        "-r", str(int(fps)),
        "-vf", "scale='trunc(iw/2)*2':'trunc(ih/2)*2'",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(dst_mp4),
    ]
    _sp.run(cmd, check=True)
    return dst_mp4


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=str(MASTER_DEFAULT))
    ap.add_argument("--results", required=True)
    ap.add_argument("--logs_dir", default=None)
    ap.add_argument("--conda_env", default="360dvd")
    ap.add_argument("--repo", default=str(DVD360_REPO))
    ap.add_argument("--default_flow", default=str(DEFAULT_FLOW),
                    help="Reference flow mp4 baked into 360DVD's prompt yaml")
    ap.add_argument("--num_steps", type=int, default=25)
    ap.add_argument("--guidance_scale", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--n_prompt", default=("blur, haze, deformed iris, deformed pupils, "
                                           "semi-realistic, cgi, 3d, render, sketch, cartoon, "
                                           "drawing, anime, mutated hands and fingers, deformed, "
                                           "distorted, disfigured, poorly drawn, bad anatomy, "
                                           "wrong anatomy, extra limb, missing limb, floating "
                                           "limbs, disconnected limbs, mutation, mutated, ugly, "
                                           "disgusting, amputation"))
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--only_clips", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip_existing", action="store_true", default=True)
    ap.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    ap.add_argument("--dry_run", action="store_true", default=False)
    return ap.parse_args()


def write_per_clip_config(args, *, prompt: str, out_yaml: Path):
    """360DVD's animate.py reads a yaml whose top key is the prompt-set name.
    We write a single-prompt yaml for one clip."""
    text = (
        "PanoWorldEval:\n"
        "  motion_module:\n"
        "    - \"ckpts/Motion_Module/mm_sd_v14.ckpt\"\n"
        "  motion_adapter: \"ckpts/360Adapter/360Adapter_flow_v1.ckpt\"\n"
        "  dreambooth_path: \"ckpts/DreamBooth_LoRA/realisticVisionV51_v20Novae.safetensors\"\n"
        f"  seed: [{int(args.seed)}]\n"
        f"  steps: {int(args.num_steps)}\n"
        f"  guidance_scale: {float(args.guidance_scale)}\n"
        f"  prompt:\n"
        f"    - {_yaml_str(prompt)}\n"
        f"  n_prompt:\n"
        f"    - {_yaml_str(args.n_prompt)}\n"
        f"  flow:\n"
        f"    - \"{args.default_flow}\"\n"
    )
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(text)


def _yaml_str(s: str) -> str:
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    return '"' + s + '"'


def process_clip(args, row: dict) -> dict:
    clip_id = row["clip_id"]
    out_dir = per_clip_outdir(args.results, METHOD_ID, clip_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_mp4 = out_dir / "video.mp4"
    if args.skip_existing and have_output(out_dir):
        return {"clip_id": clip_id, "status": "skipped"}

    caption, caption_style = load_caption_with_style(
        row.get("caption_path", ""), style="long")
    if not caption:
        return {"clip_id": clip_id, "status": "error", "error": "no caption"}

    work_dir = out_dir / "_work"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    cfg = work_dir / "prompt.yaml"
    write_per_clip_config(args, prompt=caption, out_yaml=cfg)

    # 360DVD writes outputs to a folder under `samples/` inside the repo,
    # named after the prompt yaml's top-level key + timestamp. We point
    # samples/<run> at our work_dir via the --output_dir-equivalent. Since
    # animate.py hardcodes samples/, we instead invoke it with cwd=work_dir
    # so its `samples/` materializes there, then we discover the mp4.
    cmd_inner = [
        "python", "-m", "scripts.animate",
        "--config", str(cfg),
    ]
    cmd = wrap_with_conda(cmd_inner, args.conda_env)

    write_run_json(out_dir, {
        "method_id": METHOD_ID, "clip_id": clip_id, "status": "running",
        "caption": caption, "caption_style": caption_style,
        "cmd": cmd, "ts_start": int(time.time()),
    })

    if args.dry_run:
        return {"clip_id": clip_id, "status": "dry_run", "cmd": cmd}

    log_dir = Path(args.logs_dir) if args.logs_dir else (
        Path(args.results) / "_logs" / METHOD_ID
    )
    log_path = log_dir / f"{clip_id.replace('::', '__')}.log"

    # Run inside repo so relative ckpts paths resolve, and copy/symlink
    # samples/ outputs into work_dir afterwards.
    t0 = time.time()
    rc = run_subprocess(cmd, log_path=log_path, cwd=str(args.repo))
    dt = time.time() - t0
    if rc != 0:
        write_run_json(out_dir, {
            "method_id": METHOD_ID, "clip_id": clip_id, "status": "failed",
            "rc": rc, "elapsed_s": dt, "log": str(log_path), "cmd": cmd,
        })
        return {"clip_id": clip_id, "status": "error",
                "error": f"subprocess rc={rc}", "log": str(log_path)}

    # 360DVD's animate.py writes outputs as .gif (NOT .mp4) into
    # {repo}/samples/<run_dir>/, so we accept both extensions and pick the
    # most recently modified file (one run per process, but the dir grows
    # cumulatively across clips).
    samples_dir = Path(args.repo) / "samples"
    src = _find_latest_output(samples_dir, suffixes=(".mp4", ".gif"),
                              exclude_substrings=("grid",))
    if src is None:
        src = _find_latest_output(work_dir, suffixes=(".mp4", ".gif"))
    if src is None:
        write_run_json(out_dir, {
            "method_id": METHOD_ID, "clip_id": clip_id, "status": "failed",
            "elapsed_s": dt, "error": "no mp4/gif produced",
            "log": str(log_path), "cmd": cmd,
        })
        return {"clip_id": clip_id, "status": "error",
                "error": "no mp4/gif; see " + str(log_path)}

    if src.suffix.lower() == ".gif":
        _gif_to_mp4(src, final_mp4, fps=8)
    else:
        normalize_output_video(src, final_mp4)
    try:
        shutil.rmtree(src.parent.parent)
    except Exception:
        pass

    write_run_json(out_dir, {
        "method_id": METHOD_ID, "clip_id": clip_id, "status": "ok",
        "caption": caption, "caption_style": caption_style,
        "elapsed_s": dt,
        "video": str(final_mp4), "log": str(log_path), "cmd": cmd,
    })
    try:
        shutil.rmtree(work_dir)
    except Exception:
        pass
    return {"clip_id": clip_id, "status": "ok", "elapsed_s": dt,
            "video": str(final_mp4)}


def main():
    args = parse_args()
    rows = select_clips(read_master(args.master),
                        splits=args.splits, only_clips=args.only_clips,
                        limit=args.limit)
    print(f"[360dvd_runner] method={METHOD_ID} {len(rows)} clip(s)")
    if args.dry_run:
        print("[360dvd_runner] DRY RUN")
    n_ok = n_skip = n_err = 0
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        cid = row["clip_id"]
        try:
            res = process_clip(args, row)
        except Exception as e:
            import traceback; traceback.print_exc()
            res = {"clip_id": cid, "status": "error", "error": str(e)}
        st = res.get("status")
        if st == "ok":
            n_ok += 1
            extra = f" ({res.get('elapsed_s', 0):.1f}s)"
        elif st == "skipped":
            n_skip += 1; extra = " skip"
        elif st == "dry_run":
            extra = " dry"
        else:
            n_err += 1; extra = f" FAIL: {res.get('error', '')}"
        print(f"[{i}/{len(rows)}] {cid}{extra}", flush=True)
    print(f"\n=== Done ===  ok={n_ok}  skip={n_skip}  err={n_err}  "
          f"elapsed={time.time()-t0:.1f}s")
    if n_err > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
