#!/usr/bin/env python3
"""Diagnose density shift, fixed-point drift and iterative accuracy on a named split.

Ground truth is used for reporting only, never candidate selection. This script
does not modify checkpoints, deployment defaults, or publish extrinsics/TF.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from dst_calib.config import load_config
from dst_calib.calibrate import _apply_ckpt_virtual_camera
from dst_calib.eva_infer import load_eva
from dst_calib.difference_map import build_difference_map
from dst_calib.geometry import np_se3_from_rt
from dst_calib.projection import generate_cdp, generate_ldp, virtual_camera_intrinsics
from dst_calib.self_supervised import Frame, camera_cloud
from dst_calib.augmentation import sample_perturbation


def pose_error(T, truth):
    c = np.clip((np.trace(T[:3, :3] @ truth[:3, :3].T) - 1) / 2, -1, 1)
    return {"rotation_deg": float(np.degrees(np.arccos(c))),
            "translation_m": float(np.linalg.norm(T[:3, 3] - truth[:3, 3]))}


def limit_points(points, cap):
    if cap <= 0 or len(points) <= cap:
        return points
    # The RBT snapshot uses a deterministic stride; keep the same ordering.
    stride = (len(points) + cap - 1) // cap
    return points[::stride]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    p.add_argument("--frame-limit", type=int, default=8)
    p.add_argument("--frame-offset", type=int, default=0)
    p.add_argument("--points-caps", type=int, nargs="+", default=[12000, 0])
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--seed", type=int, default=20261031)
    p.add_argument("--with-guard", action="store_true", help="compare raw iteration with the opt-in geometry guard")
    a = p.parse_args()
    if any(cap < 0 for cap in a.points_caps):
        p.error("point caps must be nonnegative (0 means uncapped)")
    if a.frame_limit <= 0 or a.frame_offset < 0 or a.steps < 0:
        p.error("frame-limit must be positive; offset/steps must be nonnegative")
    torch.set_num_threads(2)
    data = Path(a.data_dir)
    cfg = load_config(a.config)
    truth_data = yaml.safe_load((data / "extrinsic.yaml").read_text())
    truth = np.asarray(truth_data["T_cam_lidar"], dtype=np.float64)
    model, meta = load_eva(a.checkpoint, a.device)
    _apply_ckpt_virtual_camera(cfg, meta)
    if meta["arch"] != "sb":
        raise ValueError("This audit requires the deployed SB model")
    size = (cfg.virtual_camera.height, cfg.virtual_camera.width)
    K = virtual_camera_intrinsics(cfg.virtual_camera.focal, size)
    files = sorted(data.glob("frame_*.npz"))[a.frame_offset:a.frame_offset + a.frame_limit]
    if not files:
        raise ValueError("No frames selected")
    rows = []
    rng = np.random.default_rng(a.seed)
    for file in files:
        with np.load(file) as z:
            frame = Frame(np.array(z["points_lidar"], dtype=np.float32),
                          np.array(z["depth"], dtype=np.float32),
                          np.array(z["K"], dtype=np.float64), None, float(z["stamp"]))
        Q = camera_cloud(frame, cfg)
        cdp = generate_cdp(Q, np.eye(4), K, size)
        perturbations = [("identity", np.zeros(6)),
                         ("small", sample_perturbation(.5, .05, rng=rng)),
                         ("training_range", sample_perturbation(5., .5, rng=rng))]
        for scenario, xi in perturbations:
            T0 = np_se3_from_rt(xi[:3], xi[3:]) @ truth
            for cap in a.points_caps:
                P = limit_points(frame.points_lidar, cap)
                variants = ("raw", "guarded") if a.with_guard else ("raw",)
                for variant in variants:
                    run_trajectory(frame, P, cdp, T0, truth, model, a, cfg, K, size, file, scenario, cap, variant, rows)
        print(f"audited {file.name}, points={len(frame.points_lidar)}", flush=True)
    summary = []
    variants = ("raw", "guarded") if a.with_guard else ("raw",)
    for variant in variants:
        for scenario, _ in perturbations:
            for cap in a.points_caps:
                baseline = [r for r in rows if r["variant"] == variant and r["scenario"] == scenario and r["cap"] == cap and r["step"] == 0]
                for step in range(a.steps + 1):
                    selected = [r for r in rows if r["variant"] == variant and r["scenario"] == scenario and r["cap"] == cap and r["step"] == step]
                    r = dict(variant=variant, scenario=scenario, cap=cap, step=step, n=len(selected))
                    for key in ("rotation_deg", "translation_m"):
                        values = np.asarray([v[key] for v in selected])
                        r[key + "_mean"] = float(values.mean())
                        r[key + "_median"] = float(np.median(values))
                        r[key + "_p95"] = float(np.percentile(values, 95))
                        r[key + "_worsened_fraction"] = float(np.mean(values > np.asarray([v[key] for v in baseline]) + 1e-7))
                    summary.append(r)
    report = dict(scope="diagnostic, named development split; not new independent held-out accuracy",
                  data_dir=str(data.resolve()), checkpoint=str(Path(a.checkpoint).resolve()),
                  checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
                  device=a.device, seed=a.seed, config=a.config,
                  sampling="points[::ceil(N/cap)] matches corrected native RBT cap",
                  frame_files=[str(f) for f in files], summary=summary, rows=rows)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    for row in summary:
        print(row["variant"], row["scenario"], row["cap"], row["step"],
              f"rotation={row['rotation_deg_mean']:.4f} deg, translation={row['translation_m_mean']:.4f} m")


def run_trajectory(frame, P, cdp, T0, truth, model, a, cfg, K, size, file, scenario, cap, variant, rows):
    from dst_calib.stability import FixedSupportGeometry, improves
    guard_frame = Frame(P, frame.depth, frame.K, None, frame.stamp)
    guard = FixedSupportGeometry([guard_frame], T0, cfg) if variant == "guarded" else None
    T = T0.copy()
    baseline = guard(T) if guard else None
    stopped = False
    for step in range(a.steps + 1):
        row = dict(frame=file.name, scenario=scenario, cap=cap,
                   points=len(P), variant=variant, step=step, stopped=stopped, **pose_error(T, truth))
        if step < a.steps and not stopped:
            ldp = generate_ldp(P, T, K, size)
            diff = build_difference_map(ldp, cdp, cfg.difference_map.e_tar)
            with torch.inference_mode():
                output = model(torch.from_numpy(diff).unsqueeze(0).to(a.device)).cpu().numpy().reshape(6)
            if not np.isfinite(output).all():
                raise RuntimeError("Nonfinite model output")
            row.update(ldp_pixels=int(np.count_nonzero(ldp)), cdp_pixels=int(np.count_nonzero(cdp)),
                       overlap_pixels=int(np.count_nonzero((ldp > 0) & (cdp > 0))),
                       predicted_delta=output.tolist())
            if guard is None:
                T = np_se3_from_rt(output[:3], output[3:]) @ T
            else:
                stopped = True
                for scale in (1., .5, .25):
                    candidate = np_se3_from_rt(output[:3] * scale, output[3:] * scale) @ T
                    score = guard(candidate)
                    if improves(score, baseline):
                        T, baseline, stopped = candidate, score, False
                        row["accepted_scale"] = scale
                        break
                row["accepted"] = not stopped
        rows.append(row)

if __name__ == "__main__":
    main()
