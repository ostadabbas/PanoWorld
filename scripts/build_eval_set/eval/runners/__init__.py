"""Inference runners that produce per-clip outputs consumable by run_eval.py.

Layout convention:
    results_dir/<method_id>/<clip_id_dir>/video.mp4
    results_dir/<method_id>/<clip_id_dir>/run.json
    results_dir/<method_id>/<clip_id_dir>/depth/0000.npy ...   (optional)

Where <clip_id_dir> = clip_id with "::" replaced by "__" for filesystem safety.
"""
