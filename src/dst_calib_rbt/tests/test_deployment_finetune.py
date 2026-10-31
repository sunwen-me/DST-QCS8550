import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from dst_calib.deployment_finetune import (DeploymentDataset, correction_target,
    limit_points, freeze_bn, normalized_residual_loss)
from dst_calib.geometry import np_se3_from_rt


def test_exact_left_correction_including_lever_arm():
    truth = np_se3_from_rt(np.array([.7, -.5, .9]), np.array([.2, -.1, .4]))
    delta = np_se3_from_rt(np.array([.02, -.04, .01]), np.array([.05, .02, -.06]))
    current = delta @ truth
    xi = correction_target(truth, current)
    np.testing.assert_allclose(np_se3_from_rt(xi[:3], xi[3:]) @ current, truth, atol=1e-7)
    np.testing.assert_allclose(correction_target(truth, truth), 0, atol=1e-7)


@pytest.mark.parametrize("count,cap", [(0, 12000), (12000,12000), (12001,12000), (121184,12000), (2,0)])
def test_point_cap_matches_native_ceil_stride(count, cap):
    points = np.arange(count * 3).reshape(-1, 3)
    chosen = limit_points(points, cap)
    stride = max(1, (count + cap-1)//cap) if cap else 1
    np.testing.assert_array_equal(chosen, points[::stride])
    assert not cap or len(chosen) <= cap


def test_invalid_cap_and_loss_shapes_rejected():
    with pytest.raises(ValueError):
        limit_points(np.zeros((3, 3)), -1)
    with pytest.raises(ValueError):
        normalized_residual_loss(torch.zeros(1, 5), torch.zeros(1, 5))


def test_zero_loss_has_finite_gradient():
    pred = torch.zeros(2, 6, requires_grad=True)
    loss = normalized_residual_loss(pred, torch.zeros(2, 6))
    loss.backward()
    assert loss.item() == 0 and torch.isfinite(pred.grad).all()


def test_bn_stats_frozen_while_gradients_flow():
    model = torch.nn.Sequential(torch.nn.Linear(6, 6), torch.nn.BatchNorm1d(6))
    freeze_bn(model)
    before = model[1].running_mean.clone()
    model(torch.ones(4, 6)).sum().backward()
    assert model.training and not model[1].training
    assert model[0].weight.grad is not None
    torch.testing.assert_close(before, model[1].running_mean)


def test_dataset_deterministic_per_epoch_and_exact_identity():
    from dst_calib.projection import generate_ldp, virtual_camera_intrinsics
    rng = np.random.default_rng(17)
    points = rng.uniform([-.5,-.5,3], [.5,.5,5], (200,3)).astype(np.float32)
    config = NS(virtual_camera=NS(height=32, width=64, focal=40), difference_map=NS(e_tar=.1))
    K = virtual_camera_intrinsics(40, (32,64))
    cdp = generate_ldp(points, np.eye(4), K, (32,64))
    records = [dict(path="train/frame_0.npz", points=points, truth=np.eye(4), cdp=cdp)]
    ds = DeploymentDataset(records, config, caps=(12000,), samples_per_frame=4)
    image1, label1 = ds[0]
    image2, label2 = ds[0]
    torch.testing.assert_close(image1, image2)
    torch.testing.assert_close(label1, label2)
    torch.testing.assert_close(label1, torch.zeros(6))
    assert image1[1:].abs().sum() == 0
    ds.epoch=1
    _, nonzero = ds[0]
    assert nonzero.abs().sum() > 0
