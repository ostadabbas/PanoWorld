#!/usr/bin/env python3
"""OmniRoam inference runner (Preview + Refine, fixed forward trajectory).

OmniRoam (Adobe, 2026) is a two-stage panoramic-video generator built on
Wan-2.1: a Preview stage outputs a low-resolution ERP video conditioned
on a single image + a per-frame trajectory, then a Refine stage upsamples
it to 720p. Per the input-parity policy in eval_config.yaml, we feed it
a FIXED FORWARD trajectory (NOT the GT trajectory), which is what
`--traj_mode fixed --traj_preset forward` enables.

Two stages of OUR evaluation:
    --stage pers  -> input = perspective crop of GT first ERP frame
                     (one PNG in <work>/preview_input/)
    --stage erp   -> input = the GT ERP first frame itself
                     (one PNG; OmniRoam treats it as a single still and
                     unrolls 81 frames over the fixed-forward trajectory)

Setup (run once on A100):
    cd $HOME/Le/OmniRoam
    pip install -r requirements.txt           # into a fresh `omniroam` env
    python download_wan2.1.py                  # base model (already done)
    python download_omniroam_models.py         # Preview + Refine ckpts
        # this writes models/OmniRoam/Preview/preview.ckpt and
        # models/OmniRoam/Refine/refine.ckpt   (these are not yet on disk)

Per-clip flow:
    pers/erp PNG  ─Preview (480x960, 81f, fixed forward)─►  preview/<id>.mp4
    preview      ─Refine  (720x1440, 81f, crossfade)─────►  refine/<id>.mp4
    refine.mp4 → results_dir/<method_id>/<clip_id_dir>/video.mp4

method_id resolves to:
    --stage pers  ->  omniroam_pers
    --stage erp   ->  omniroam_erp

Usage:
    python -m test_set_pkg.eval.runners.infer_omniroam \\
        --stage pers \\
        --results $PANO_DATA_ROOT/eval_results \\
        --conda_env omniroam \\
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
    erp_to_perspective_image,
    find_first_mp4,
    have_output,
    per_clip_outdir,
    read_first_frame_rgb_uint8,
    read_master,
    run_subprocess,
    save_image_png,
    select_clips,
    wrap_with_conda,
    write_run_json,
)

OMNI_REPO = Path("$HOME/Le/OmniRoam")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pers", "erp"], required=True,
                    help="pers = pers crop input; erp = full ERP image input. "
                         "Both feed a fixed-forward trajectory (parity policy).")
    ap.add_argument("--master", default=str(MASTER_DEFAULT))
    ap.add_argument("--results", required=True)
    ap.add_argument("--logs_dir", default=None)
    ap.add_argument("--conda_env", default="omniroam")
    ap.add_argument("--repo", default=str(OMNI_REPO))
    ap.add_argument("--preview_ckpt",
                    default="models/OmniRoam/Preview/preview.ckpt")
    ap.add_argument("--refine_ckpt",
                    default="models/OmniRoam/Refine/refine.ckpt")
    ap.add_argument("--preview_h", type=int, default=480)
    ap.add_argument("--preview_w", type=int, default=960)
    ap.add_argument("--refine_h", type=int, default=720)
    ap.add_argument("--refine_w", type=int, default=1440)
    ap.add_argument("--num_frames", type=int, default=81)
    ap.add_argument("--cfg_scale", type=float, default=5.0)
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--devices", default="cuda:0",
                    help="Comma-separated CUDA devices to spread Wan inference across")
    ap.add_argument("--fov_x", type=float, default=90.0)
    ap.add_argument("--pers_h", type=int, default=480)
    ap.add_argument("--pers_w", type=int, default=640)
    ap.add_argument("--skip_refine", action="store_true", default=False,
                    help="Stop after Preview; useful for fast smoke testing")
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--only_clips", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip_existing", action="store_true", default=True)
    ap.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    ap.add_argument("--dry_run", action="store_true", default=False)
    return ap.parse_args()


def build_preview_cmd(args, *, images_dir: str, output_dir: str) -> list[str]:
    return [
        "python", "infer_omniroam.py",
        "--local_images_dir", images_dir,
        "--height", str(args.preview_h),
        "--width",  str(args.preview_w),
        "--num_frames", str(args.num_frames),
        "--ckpt_path", args.preview_ckpt,
        "--enable_speed_control", "--speed_fixed", "1.0",
        "--use_cam_traj",
        "--traj_mode", "fixed",
        "--traj_preset", "forward",
        "--re_scale_pose", "fixed:1.0",
        "--traj_s_curve_amp_m", "1.4",
        "--traj_loop_radius_m", "1.5",
        "--cfg_scale", str(args.cfg_scale),
        "--num_inference_steps", str(args.num_inference_steps),
        "--output_dir", output_dir,
        "--devices", args.devices,
    ]


def build_refine_cmd(args, *, refine_local_dir: str, output_dir: str) -> list[str]:
    return [
        "python", "infer_omniroam.py",
        "--enable_refine",
        "--refine_local_dir", refine_local_dir,
        "--refine_num_segments", "8",
        "--refine_degrade_down_h", str(args.preview_h),
        "--refine_degrade_down_w", str(args.preview_w),
        "--refine_use_crossfade",
        "--refine_crossfade_alpha", "0.5",
        "--height", str(args.refine_h),
        "--width",  str(args.refine_w),
        "--num_frames", str(args.num_frames),
        "--ckpt_path", args.refine_ckpt,
        "--output_dir", output_dir,
        "--devices", args.devices,
    ]


def process_clip(args, row: dict, method_id: str) -> dict:
    clip_id = row["clip_id"]
    out_dir = per_clip_outdir(args.results, method_id, clip_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_mp4 = out_dir / "video.mp4"
    if args.skip_existing and have_output(out_dir):
        return {"clip_id": clip_id, "status": "skipped"}

    gt_video = row.get("video_path", "")
    if not gt_video or not Path(gt_video).is_file():
        return {"clip_id": clip_id, "status": "error",
                "error": f"GT video missing: {gt_video}"}

    work_dir = out_dir / "_work"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    erp0 = read_first_frame_rgb_uint8(gt_video)
    images_dir = work_dir / "preview_input"
    images_dir.mkdir()
    if args.stage == "pers":
        pers_arr = erp_to_perspective_image(
            erp0, fov_x_deg=args.fov_x, yaw_rad=0.0, pitch_rad=0.0,
            out_h=args.pers_h, out_w=args.pers_w,
        )
        save_image_png(pers_arr, images_dir / f"{clip_id.replace('::', '__')}.png")
    else:
        save_image_png(erp0, images_dir / f"{clip_id.replace('::', '__')}.png")

    log_dir = Path(args.logs_dir) if args.logs_dir else (
        Path(args.results) / "_logs" / method_id
    )
    log_path = log_dir / f"{clip_id.replace('::', '__')}.log"

    # ── Stage 1: Preview ───────────────────────────────────────────────
    preview_out = work_dir / "preview"
    preview_out.mkdir()
    pcmd_inner = build_preview_cmd(args,
                                   images_dir=str(images_dir),
                                   output_dir=str(preview_out))
    pcmd = wrap_with_conda(pcmd_inner, args.conda_env)

    write_run_json(out_dir, {
        "method_id": method_id, "clip_id": clip_id, "status": "running_preview",
        "stage": args.stage, "preview_cmd": pcmd, "gt_video": gt_video,
        "ts_start": int(time.time()),
    })
    if args.dry_run:
        return {"clip_id": clip_id, "status": "dry_run",
                "preview_cmd": pcmd,
                "refine_cmd": "(omitted; dry run)"}

    t0 = time.time()
    rc = run_subprocess(pcmd, log_path=log_path, cwd=str(args.repo))
    if rc != 0:
        write_run_json(out_dir, {
            "method_id": method_id, "clip_id": clip_id, "status": "failed_preview",
            "rc": rc, "elapsed_s": time.time() - t0,
            "log": str(log_path), "preview_cmd": pcmd,
        })
        return {"clip_id": clip_id, "status": "error",
                "error": f"preview rc={rc}", "log": str(log_path)}

    preview_mp4 = find_first_mp4(preview_out, exclude_substrings=("a_input",))
    if preview_mp4 is None:
        write_run_json(out_dir, {
            "method_id": method_id, "clip_id": clip_id, "status": "failed_preview",
            "elapsed_s": time.time() - t0, "error": "no preview mp4",
            "log": str(log_path), "preview_cmd": pcmd,
        })
        return {"clip_id": clip_id, "status": "error",
                "error": "no preview mp4; see " + str(log_path)}

    if args.skip_refine:
        shutil.move(str(preview_mp4), str(final_mp4))
        write_run_json(out_dir, {
            "method_id": method_id, "clip_id": clip_id, "status": "ok_preview_only",
            "elapsed_s": time.time() - t0, "video": str(final_mp4),
            "log": str(log_path), "preview_cmd": pcmd,
        })
        try:
            shutil.rmtree(work_dir)
        except Exception:
            pass
        return {"clip_id": clip_id, "status": "ok",
                "elapsed_s": time.time() - t0, "video": str(final_mp4)}

    # ── Stage 2: Refine ────────────────────────────────────────────────
    # OmniRoam's refine takes a DIRECTORY of preview mp4s. Place ours alone.
    refine_in = work_dir / "refine_input"
    refine_in.mkdir()
    shutil.copy2(str(preview_mp4), str(refine_in / preview_mp4.name))
    refine_out = work_dir / "refine"
    refine_out.mkdir()
    rcmd_inner = build_refine_cmd(args,
                                  refine_local_dir=str(refine_in),
                                  output_dir=str(refine_out))
    rcmd = wrap_with_conda(rcmd_inner, args.conda_env)
    log_path2 = log_path.with_suffix(".refine.log")
    rc2 = run_subprocess(rcmd, log_path=log_path2, cwd=str(args.repo))
    dt = time.time() - t0
    if rc2 != 0:
        write_run_json(out_dir, {
            "method_id": method_id, "clip_id": clip_id, "status": "failed_refine",
            "rc": rc2, "elapsed_s": dt,
            "log": str(log_path), "log_refine": str(log_path2),
            "preview_cmd": pcmd, "refine_cmd": rcmd,
        })
        return {"clip_id": clip_id, "status": "error",
                "error": f"refine rc={rc2}", "log": str(log_path2)}

    refine_mp4 = find_first_mp4(refine_out, exclude_substrings=("a_input",))
    if refine_mp4 is None:
        write_run_json(out_dir, {
            "method_id": method_id, "clip_id": clip_id, "status": "failed_refine",
            "elapsed_s": dt, "error": "no refine mp4",
            "log": str(log_path), "log_refine": str(log_path2),
            "preview_cmd": pcmd, "refine_cmd": rcmd,
        })
        return {"clip_id": clip_id, "status": "error",
                "error": "no refine mp4; see " + str(log_path2)}

    shutil.move(str(refine_mp4), str(final_mp4))
    write_run_json(out_dir, {
        "method_id": method_id, "clip_id": clip_id, "status": "ok",
        "elapsed_s": dt, "video": str(final_mp4),
        "log": str(log_path), "log_refine": str(log_path2),
        "preview_cmd": pcmd, "refine_cmd": rcmd, "gt_video": gt_video,
        "stage": args.stage,
    })
    try:
        shutil.rmtree(work_dir)
    except Exception:
        pass
    return {"clip_id": clip_id, "status": "ok",
            "elapsed_s": dt, "video": str(final_mp4)}


def main():
    args = parse_args()
    method_id = "omniroam_pers" if args.stage == "pers" else "omniroam_erp"
    rows = select_clips(read_master(args.master),
                        splits=args.splits, only_clips=args.only_clips,
                        limit=args.limit)
    print(f"[omniroam_runner] stage={args.stage} method={method_id} "
          f"{len(rows)} clip(s)"
          f"{' DRY RUN' if args.dry_run else ''}")
    n_ok = n_skip = n_err = 0
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        cid = row["clip_id"]
        try:
            res = process_clip(args, row, method_id=method_id)
        except Exception as e:
            import traceback; traceback.print_exc()
            res = {"clip_id": cid, "status": "error", "error": str(e)}
        st = res.get("status")
        if st == "ok":
            n_ok += 1; extra = f" ({res.get('elapsed_s', 0):.1f}s)"
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
