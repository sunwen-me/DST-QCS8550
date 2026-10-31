"""Experimental deployment-distribution fine-tuning, NOT the paper baseline.

The upstream trainer/losses/checkpoints remain unchanged. This branch trains
the left correction truth @ inverse(current), adds identity/small residuals,
matches native ceil-stride sampling, and freezes BN statistics. Only training
frames may generate replay examples; validation is reporting/selection only.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import yaml

from .augmentation import sample_perturbation
from .difference_map import build_difference_map
from .geometry import np_rt_from_se3, np_se3_from_rt, np_se3_inverse, np_transform
from .projection import generate_cdp, generate_ldp, virtual_camera_intrinsics
from .self_supervised import Frame, camera_cloud

SCENARIOS = {"identity": (0., 0.), "small": (.5, .05),
             "medium": (2., .2), "large": (5., .5)}


def limit_points(points, cap):
    if cap < 0:
        raise ValueError("point cap must be nonnegative")
    if cap == 0 or len(points) <= cap:
        return points
    return points[::(len(points) + cap - 1) // cap]


def correction_target(truth, current):
    """Exact left SE(3) correction, including rotated translation composition."""
    correction = np.asarray(truth) @ np_se3_inverse(np.asarray(current))
    r, t = np_rt_from_se3(correction)
    return np.concatenate([r, t]).astype(np.float32)


def pose_error(pose, truth):
    c = np.clip((np.trace(pose[:3, :3] @ truth[:3, :3].T) - 1) / 2, -1, 1)
    return dict(rotation_deg=float(np.degrees(np.arccos(c))),
                translation_m=float(np.linalg.norm(pose[:3, 3] - truth[:3, 3])))


def load_records(directories, cfg, *, per_session=0):
    records = []
    size = int(cfg.virtual_camera.height), int(cfg.virtual_camera.width)
    K = virtual_camera_intrinsics(float(cfg.virtual_camera.focal), size)
    for directory in directories:
        directory = Path(directory).resolve()
        truth_path = directory / "extrinsic.yaml"
        data = yaml.safe_load(truth_path.read_text())
        truth = np.asarray(data["T_cam_lidar"], dtype=np.float64)
        if (truth.shape != (4, 4) or not np.isfinite(truth).all()
                or not np.allclose(truth[3], [0, 0, 0, 1])
                or not np.allclose(truth[:3, :3] @ truth[:3, :3].T, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(truth[:3, :3]), 1)):
            raise ValueError(f"invalid ground-truth transform in {truth_path}")
        files = sorted(directory.glob("frame_*.npz"))
        if per_session and len(files) > per_session:
            files = [files[i] for i in np.linspace(0, len(files)-1, per_session, dtype=int)]
        for path in files:
            with np.load(path, allow_pickle=False) as z:
                frame = Frame(np.array(z["points_lidar"], dtype=np.float32),
                              np.array(z["depth"], dtype=np.float32),
                              np.array(z["K"], dtype=np.float64), None,
                              float(np.asarray(z["stamp"]).reshape(-1)[0]))
            Q = camera_cloud(frame, cfg)
            if len(Q) < 500 or len(frame.points_lidar) < 30:
                raise ValueError(f"insufficient geometry: {path}")
            records.append(dict(path=str(path), points=frame.points_lidar, truth=truth.copy(),
                                cdp=generate_cdp(Q, np.eye(4), K, size)))
        print(f"loaded {directory}: {len(files)} frames", flush=True)
    if not records:
        raise ValueError("no frames selected")
    return records


class DeploymentDataset(Dataset):
    def __init__(self, records, cfg, *, seed=20261031, samples_per_frame=4,
                 caps=(12000, 24000, 60000, 0), replay=None):
        if samples_per_frame <= 0 or not caps or any(c < 0 for c in caps):
            raise ValueError("invalid deployment dataset settings")
        self.records, self.cfg, self.seed = records, cfg, seed
        self.samples_per_frame, self.caps = samples_per_frame, tuple(caps)
        self.epoch = 0
        self.replay = replay or []
        self.size = int(cfg.virtual_camera.height), int(cfg.virtual_camera.width)
        self.K = virtual_camera_intrinsics(float(cfg.virtual_camera.focal), self.size)

    def __len__(self):
        return len(self.records) * self.samples_per_frame + len(self.replay)

    def __getitem__(self, index):
        regular = len(self.records) * self.samples_per_frame
        if index < regular:
            frame_index = index % len(self.records)
            record = self.records[frame_index]
            rng = np.random.default_rng(self.seed + self.epoch * 1000003 + index)
            scenario = tuple(SCENARIOS)[(index // len(self.records) + self.epoch) % 4]
            cap = int(rng.choice(self.caps))
            r, t = SCENARIOS[scenario]
            xi = sample_perturbation(r, t, rng=rng)
            current = np_se3_from_rt(xi[:3], xi[3:]) @ record["truth"]
        else:
            entry = self.replay[index - regular]
            record, current, cap = self.records[entry["frame_index"]], entry["current"], entry["cap"]
        points = limit_points(record["points"], cap)
        ldp = generate_ldp(points, current, self.K, self.size)
        diff = build_difference_map(ldp, record["cdp"], self.cfg.difference_map.e_tar)
        target = correction_target(record["truth"], current)
        return torch.from_numpy(diff), torch.from_numpy(target)


def normalized_residual_loss(prediction, target):
    """Engineering objective: rad(1 degree)/5cm scales, not paper equation16."""
    if prediction.shape != target.shape or prediction.shape[-1] != 6:
        raise ValueError("prediction/target must have matching (...,6) shapes")
    scales = prediction.new_tensor([np.deg2rad(1.)] * 3 + [.05] * 3)
    return torch.nn.functional.smooth_l1_loss(prediction / scales, target / scales)


def freeze_bn(model):
    """Keep inference/training normalization aligned while retaining gradients."""
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def mine_replay(model, records, cfg, device, *, max_frames=96, seed=20261101):
    """Training-only model trajectories, targets recomputed from their true pose."""
    size = int(cfg.virtual_camera.height), int(cfg.virtual_camera.width)
    K = virtual_camera_intrinsics(float(cfg.virtual_camera.focal), size)
    indices = np.linspace(0, len(records)-1, min(max_frames, len(records)), dtype=int)
    rng = np.random.default_rng(seed)
    replay = []
    model.eval()
    with torch.no_grad():
        for i in indices:
            record = records[int(i)]
            for cap in (12000, 0):
                xi = sample_perturbation(5., .5, rng=rng)
                current = np_se3_from_rt(xi[:3], xi[3:]) @ record["truth"]
                points = limit_points(record["points"], cap)
                for _ in range(2):
                    ldp = generate_ldp(points, current, K, size)
                    diff = build_difference_map(ldp, record["cdp"], cfg.difference_map.e_tar)
                    value = model(torch.from_numpy(diff).unsqueeze(0).to(device)).cpu().numpy().reshape(-1)
                    if value.shape != (6,) or not np.isfinite(value).all():
                        raise RuntimeError("invalid replay model output")
                    current = np_se3_from_rt(value[:3], value[3:]) @ current
                    replay.append(dict(frame_index=int(i), cap=cap, current=current.copy()))
    print(f"mined {len(replay)} training-only residual examples", flush=True)
    return replay


def evaluate_trajectories(model, records, cfg, device, *, seed=20261102, steps=3):
    """Frozen validation cases; no geometry guard hiding model regressions."""
    size = int(cfg.virtual_camera.height), int(cfg.virtual_camera.width)
    K = virtual_camera_intrinsics(float(cfg.virtual_camera.focal), size)
    rows, rng = [], np.random.default_rng(seed)
    model.eval()
    with torch.no_grad():
        for record in records:
            for scenario in ("identity", "small", "large"):
                r, t = SCENARIOS[scenario]
                xi = sample_perturbation(r, t, rng=rng)
                initial = np_se3_from_rt(xi[:3], xi[3:]) @ record["truth"]
                for cap in (12000, 0):
                    points = limit_points(record["points"], cap)
                    current = initial.copy()
                    for step in range(steps + 1):
                        rows.append(dict(frame=record["path"], scenario=scenario, cap=cap,
                                         step=step, **pose_error(current, record["truth"])))
                        if step == steps:
                            continue
                        ldp = generate_ldp(points, current, K, size)
                        diff = build_difference_map(ldp, record["cdp"], cfg.difference_map.e_tar)
                        value = model(torch.from_numpy(diff).unsqueeze(0).to(device)).cpu().numpy().reshape(-1)
                        if value.shape != (6,) or not np.isfinite(value).all():
                            raise RuntimeError("invalid validation model output")
                        current = np_se3_from_rt(value[:3], value[3:]) @ current
    summary = []
    for scenario in ("identity", "small", "large"):
        for cap in (12000, 0):
            for step in (0, 1, steps):
                selected = [r for r in rows if (r["scenario"], r["cap"], r["step"]) == (scenario, cap, step)]
                initial = [r for r in rows if (r["scenario"], r["cap"], r["step"]) == (scenario, cap, 0)]
                item = dict(scenario=scenario, cap=cap, step=step, n=len(selected))
                for key in ("rotation_deg", "translation_m"):
                    values = np.array([r[key] for r in selected])
                    item[key + "_mean"] = float(values.mean())
                    item[key + "_median"] = float(np.median(values))
                    item[key + "_p95"] = float(np.percentile(values, 95))
                    item[key + "_worsened_fraction"] = float(np.mean(values > np.array([r[key] for r in initial]) + 1e-7))
                summary.append(item)
    # Prespecified equal-weight final error over all scenarios and both caps.
    final = [r for r in summary if r["step"] == steps]
    selection_score = float(np.mean([r["rotation_deg_mean"] + r["translation_m_mean"] / .05 for r in final]))
    return dict(scope="development validation only; raw three-step trajectories; no accuracy certificate",
                seed=seed, steps=steps, selection_score=selection_score, summary=summary, rows=rows)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
