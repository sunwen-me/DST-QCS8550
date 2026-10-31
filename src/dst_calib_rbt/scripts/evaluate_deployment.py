#!/usr/bin/env python3
"""Frozen-checkpoint, multi-frame CPU/QNN saved-data evaluation; no TF/service.

Truth generates prespecified priors and reports accuracy, never selects updates.
Do not use the test split for choosing checkpoints or tuning thresholds.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from dst_calib.augmentation import sample_perturbation
from dst_calib.calibrate import _apply_ckpt_virtual_camera
from dst_calib.config import load_config, _to_ns
from dst_calib.deployment_finetune import limit_points, pose_error, sha256
from dst_calib.eva_infer import eva_calibrate_frames, load_eva
from dst_calib.geometry import np_se3_from_rt
from dst_calib.self_supervised import Frame


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--backend", choices=("torch", "qnn"), default="torch")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--qnn-context")
    p.add_argument("--qnn-manifest")
    p.add_argument("--qnn-runtime", default="runtime/qairt250")
    p.add_argument("--qnn-library", default="build/dst-calib/lib/libdst_qnn.so")
    p.add_argument("--seed", type=int, default=20261111)
    p.add_argument("--window-size", type=int, default=4)
    p.add_argument("--window-limit", type=int, default=0)
    p.add_argument("--frame-offset", type=int, default=0)
    p.add_argument("--caps", type=int, nargs="+", default=[12000, 0])
    p.add_argument("--variants", nargs="+", choices=("raw", "guarded"), default=["raw", "guarded"])
    p.add_argument("--scenarios", nargs="+", choices=("identity", "small", "large"), default=["identity", "small", "large"])
    a = p.parse_args()
    if min(a.window_size, a.window_limit+1, a.frame_offset+1) <= 0 or any(c < 0 for c in a.caps):
        p.error("invalid window or point-cap settings")
    if Path(a.output).exists():
        raise FileExistsError("refuse to overwrite evaluation evidence")
    torch.set_num_threads(2)
    data = Path(a.data_dir).resolve()
    cfg = load_config(a.config)
    cfg.inference = getattr(cfg, "inference", _to_ns({}))
    truth = np.asarray(yaml.safe_load((data / "extrinsic.yaml").read_text())["T_cam_lidar"])
    frames = []
    files = sorted(data.glob("frame_*.npz"))[a.frame_offset:]
    if a.window_limit:
        files = files[:a.window_limit*a.window_size]
    for file in files:
        with np.load(file, allow_pickle=False) as z:
            frames.append(Frame(np.array(z["points_lidar"], dtype=np.float32),
                                np.array(z["depth"], dtype=np.float32), np.array(z["K"]),
                                None, float(np.asarray(z["stamp"]).reshape(-1)[0])))
    if len(frames) < a.window_size:
        raise ValueError("not enough frames for one complete window")
    cfg.inference.stability_guard = _to_ns(dict(enabled=False, max_points=2048, voxel=.1,
                                                trunc=.3, min_relative_improvement=.01))
    if a.backend == "qnn":
        from dst_calib.qnn_eva import QnnEvaluator
        if not a.qnn_context or not a.qnn_manifest:
            p.error("QNN requires context and manifest")
        manifest = json.loads(Path(a.qnn_manifest).read_text())
        if manifest["checkpoint_sha256"] != sha256(a.checkpoint):
            raise ValueError("QNN checkpoint/manifest digest mismatch")
        model = QnnEvaluator(a.qnn_context, a.qnn_runtime, a.qnn_library, a.qnn_manifest)
        meta = model._dst_eva_meta
        device = "cpu"
    else:
        model, meta = load_eva(a.checkpoint, a.device)
        device = a.device
    _apply_ckpt_virtual_camera(cfg, meta)
    rng = np.random.default_rng(a.seed)
    rows = []
    ranges = dict(identity=(0., 0.), small=(.5,.05), large=(5.,.5))
    start = time.perf_counter()
    try:
        for offset in range(0, len(frames)-a.window_size+1, a.window_size):
            window = frames[offset:offset+a.window_size]
            for scenario in a.scenarios:
                xi = sample_perturbation(*ranges[scenario], rng=rng)
                prior = np_se3_from_rt(xi[:3], xi[3:]) @ truth
                for cap in a.caps:
                    capped = [Frame(limit_points(fr.points_lidar, cap), fr.depth, fr.K, None, fr.stamp) for fr in window]
                    for variant in a.variants:
                        cfg.inference.stability_guard.enabled = variant == "guarded"
                        tick = time.perf_counter()
                        result = eva_calibrate_frames(capped, prior, model, cfg, device, n_iters=3)
                        T = result["T_cam_lidar"]
                        rows.append(dict(window=offset//a.window_size, scenario=scenario, cap=cap, variant=variant,
                                         frame_files=[str(f) for f in files[offset:offset+a.window_size]],
                                         initial_error=pose_error(prior, truth), truth_error=pose_error(T, truth),
                                         T_cam_lidar=T.tolist(), scores=result["scores"].tolist(),
                                         stability=result.get("stability", {}), seconds=time.perf_counter()-tick))
            print(f"window {offset//a.window_size} finished", flush=True)
    finally:
        if a.backend == "qnn":
            model.close()
    summary = []
    for scenario in a.scenarios:
        for cap in a.caps:
            for variant in a.variants:
                selected=[r for r in rows if (r["scenario"],r["cap"],r["variant"]) == (scenario,cap,variant)]
                item=dict(scenario=scenario,cap=cap,variant=variant,n=len(selected))
                for key in ("rotation_deg", "translation_m"):
                    values=np.array([r["truth_error"][key] for r in selected])
                    initial=np.array([r["initial_error"][key] for r in selected])
                    item[key+"_mean"]=float(values.mean())
                    item[key+"_median"]=float(np.median(values))
                    item[key+"_p95"]=float(np.percentile(values,95))
                    item[key+"_initial_mean"]=float(initial.mean())
                    item[key+"_worsened_fraction"]=float(np.mean(values>initial+1e-7))
                item["prior_retained_fraction"]=float(np.mean([not r["stability"].get("geometric_update_accepted", True) for r in selected]))
                summary.append(item)
    report=dict(arguments=vars(a),scope="saved-data window evaluation, not live sensor acceptance; known historical KITTI split",
                checkpoint_sha256=sha256(a.checkpoint),seconds=time.perf_counter()-start,
                frames=len(files),windows=len(rows)//(len(a.scenarios)*len(a.caps)*len(a.variants)),
                auto_backend_approved=False,live_sensor_accuracy_verified=False,summary=summary,rows=rows)
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
