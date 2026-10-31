"""KITTI geometry, independent stereo scale, and deterministic validation."""
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

from dst_calib.config import load_config
from dst_calib.train import DoubleSidedDataset, build_argparser, train

spec = importlib.util.spec_from_file_location(
    "prepare_kitti", Path(__file__).resolve().parents[3] / "tools/upstream/prepare_kitti_odometry.py")
kitti = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kitti)


def test_calibration_camera_zero_and_baseline():
    K, T, baseline = kitti.parse_calibration(
        "P0: 100 0 32 0 0 100 16 0 0 0 1 0\n"
        "P1: 100 0 32 -54 0 100 16 0 0 0 1 0\n"
        "Tr: 0 -1 0 -0.01 0 0 -1 -0.05 1 0 0 -0.29\n")
    assert baseline == pytest.approx(0.54)
    assert K[0, 0] == 100
    np.testing.assert_allclose(T @ [4, 1, 2, 1], [-1.01, -2.05, 3.71, 1])


def test_nested_selection_keeps_baseline_frames():
    ids = [f"{i:06d}" for i in range(100)]
    baseline = kitti.select_frame_ids(ids, 8)
    medium = kitti.select_frame_ids(ids, 16, baseline)
    large = kitti.select_frame_ids(ids, 32, medium)
    assert len(medium) == 16 and len(large) == 32
    assert set(baseline) <= set(medium) <= set(large)
    # Expansion should fill temporal gaps, not select adjacent source frames.
    extras = set(medium) - set(baseline)
    assert min(abs(int(x) - int(y)) for x in extras for y in baseline) >= 3
    grid = [f"{i:06d}" for i in range(65)]
    assert kitti.select_frame_ids(grid, 9, grid[::16]) == grid[::8]
    with pytest.raises(ValueError):
        kitti.select_frame_ids(ids, 4, baseline)


def test_stereo_depth_metric_scale_without_lidar():
    rng = np.random.default_rng(5)
    left = rng.integers(0, 256, (80, 512), dtype=np.uint8)
    left = cv2.GaussianBlur(left, (3, 3), 0.5)
    right = np.zeros_like(left)
    right[:, :-16] = left[:, 16:]
    K = np.array([[200., 0., 256.], [0., 200., 40.], [0., 0., 1.]])
    depth = kitti.stereo_depth(left, right, K, 0.54)
    assert np.count_nonzero(depth) > 1000
    assert np.median(depth[depth > 0]) == pytest.approx(200 * 0.54 / 16, rel=0.015)


def make_session(path):
    path.mkdir()
    K = np.array([[50., 0., 32.], [0., 50., 16.], [0., 0., 1.]])
    depth = np.full((32, 64), 3, dtype=np.float32)
    rng = np.random.default_rng(4)
    points = rng.uniform([-1, -0.7, 2], [1, 0.7, 4], (100, 3)).astype(np.float32)
    np.savez(path / "frame_000000.npz", points_lidar=points, depth=depth, K=K)
    (path / "extrinsic.yaml").write_text(yaml.safe_dump({"T_cam_lidar": np.eye(4).tolist()}))


def test_validation_is_reproducible_per_index(tmp_path):
    session = tmp_path / "session"
    make_session(session)
    cfg = load_config()
    cfg.virtual_camera.height, cfg.virtual_camera.width, cfg.virtual_camera.focal = 32, 64, 50
    dataset = DoubleSidedDataset(str(session), cfg, np.eye(4), samples_per_frame=2,
                                cloud_points=16, deterministic=True)
    first, second = dataset[0], dataset[0]
    assert all(torch.equal(first[k], second[k]) for k in first)
    assert not torch.equal(dataset[0]["xi_gt"], dataset[1]["xi_gt"])
    compact = DoubleSidedDataset(str(session), cfg, np.eye(4), samples_per_frame=2,
                                cloud_points=16, deterministic=True, compact=True)
    assert compact.frames[0].depth.size == 0
    assert all(torch.equal(first[k], compact[0][k]) for k in first)


def test_training_writes_validation_selected_checkpoint(tmp_path, monkeypatch):
    import importlib
    trainer = importlib.import_module("dst_calib.train")
    constructed = []
    dataset_class = trainer.DoubleSidedDataset
    def capture_dataset(*args, **kwargs):
        constructed.append(kwargs["seed"])
        return dataset_class(*args, **kwargs)
    monkeypatch.setattr(trainer, "DoubleSidedDataset", capture_dataset)
    torch.set_num_threads(2)
    train_dir, val_dir = tmp_path / "train", tmp_path / "val"
    make_session(train_dir)
    make_session(val_dir)
    cfg = load_config()._raw
    cfg["virtual_camera"] = {"height": 32, "width": 64, "focal": 50.}
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "output"
    args = build_argparser().parse_args([
        "--data_dir", str(train_dir), "--val_data_dir", str(val_dir),
        "--epochs", "2", "--samples_per_frame", "2",
        "--val_samples_per_frame", "2", "--cloud_points", "16",
        "--batch_size", "2", "--device", "cpu",
        "--seed", "7", "--val_seed", "1000003",
        "--config", str(config), "--out", str(out)])
    train(args)
    assert constructed == [7, 1000003]
    checkpoint = torch.load(out / "best.pt", weights_only=False, map_location="cpu")
    assert checkpoint["selection"] == "validation"
    assert np.isfinite(checkpoint["validation"]["total"])
    assert checkpoint["validation"]["samples"] == 2
    assert len((out / "metrics.jsonl").read_text().splitlines()) == 2
    assert (out / "checkpoint.pt").exists()
