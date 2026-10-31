"""Opt-in, fixed-support geometry guard. This is not a pose-accuracy certificate.

All candidates are scored against exactly the same sampled clouds, independent
of the evaluation network and without ground truth. Inconclusive updates keep
the prior. Never enable automatically solely because Chamfer decreases.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from .geometry import np_transform
from .self_supervised import (MIN_STAGE_POINTS, _cam_depth_range, camera_cloud,
                              fov_crop_points, voxel_downsample)


class InsufficientGeometryError(ValueError):
    """Retain the prior when an opt-in update lacks geometric support."""


class FixedSupportGeometry:
    def __init__(self, frames, initial, cfg, *, max_points=2048, voxel=.1, trunc=.3):
        if max_points < MIN_STAGE_POINTS or voxel <= 0 or trunc <= 0:
            raise ValueError("invalid fixed-support geometry parameters")
        self.trunc = float(trunc)
        self.clouds = []
        _, depth_max = _cam_depth_range(cfg)
        for index, frame in enumerate(frames):
            rng = np.random.default_rng(31000 + index)
            P = fov_crop_points(frame.points_lidar, initial, frame.K, frame.depth.shape, 8., depth_max)
            P = voxel_downsample(P, voxel, max_points, rng).astype(np.float64)
            Q = voxel_downsample(camera_cloud(frame, cfg), voxel, max_points, rng).astype(np.float64)
            if min(len(P), len(Q)) < MIN_STAGE_POINTS:
                raise InsufficientGeometryError("insufficient common-view geometry for a safe update")
            self.clouds.append((P, Q, cKDTree(Q)))
        if not self.clouds:
            raise InsufficientGeometryError("no frames for fixed-support geometry")

    def __call__(self, pose):
        if np.shape(pose) != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("invalid candidate pose")
        forward, reverse = [], []
        for P, Q, tree in self.clouds:
            transformed = np_transform(pose, P)
            dp = tree.query(transformed, workers=1)[0]
            dq = cKDTree(transformed).query(Q, workers=1)[0]
            forward.append(float(np.mean(np.minimum(dp, self.trunc) ** 2)))
            reverse.append(float(np.mean(np.minimum(dq, self.trunc) ** 2)))
        f, r = float(np.mean(forward)), float(np.mean(reverse))
        return dict(forward=f, reverse=r, total=(f + r) * .5)


def improves(candidate, baseline, min_relative_improvement=.01):
    if not 0 <= min_relative_improvement < 1:
        raise ValueError("min_relative_improvement must be in [0,1)")
    # Both directions must not worsen: avoid accepting overlap collapse.
    return (all(np.isfinite([candidate[k], baseline[k]]).all()
                for k in ("forward", "reverse", "total"))
            and candidate["forward"] <= baseline["forward"]
            and candidate["reverse"] <= baseline["reverse"]
            and candidate["total"] < baseline["total"] * (1 - min_relative_improvement))
