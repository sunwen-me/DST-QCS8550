"""Experimental guard: invariants, fail-closed behavior and raw compatibility."""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from dst_calib import eva_infer as eva
from dst_calib import stability as st


def cfg(enabled=True):
    return NS(inference=NS(stability_guard=NS(enabled=enabled)))


def score(x):
    return dict(forward=x, reverse=x, total=x)


def refine(monkeypatch, increment, guard=None, n_iters=3):
    monkeypatch.setattr(eva, "_eva_step", lambda *args: np.array(increment, dtype=float))
    trace = []
    result = eva._refine_from_cdp(
        np.zeros((1, 3)), np.zeros((1, 1)), np.eye(4), None, "sb",
        np.eye(3), (1, 1), .1, torch.device("cpu"), n_iters,
        geometry_guard=guard, trace=trace)
    return *result, trace


def test_raw_path_still_composes_each_step(monkeypatch):
    pose, delta, trace = refine(monkeypatch, [0, 0, 0, .1, 0, 0])
    assert pose[0, 3] == pytest.approx(.3)
    assert delta[3] == pytest.approx(.1)
    assert trace == []


def test_rejected_increment_keeps_prior(monkeypatch):
    pose, _, trace = refine(monkeypatch, [0, 0, 0, .1, 0, 0],
                            lambda T: score(T[0, 3] ** 2))
    np.testing.assert_array_equal(pose, np.eye(4))
    assert len(trace) == 1 and trace[0]["accepted"] is False


def test_backtracking_accepts_half_step(monkeypatch):
    pose, delta, trace = refine(monkeypatch, [0, 0, 0, .8, 0, 0],
                                lambda T: score((T[0, 3] - .3) ** 2), n_iters=1)
    assert trace[0]["scale"] == .5
    assert pose[0, 3] == pytest.approx(.4)
    assert delta[3] == pytest.approx(.4)


@pytest.mark.parametrize("increment", [[0] * 5, [float("nan")] * 6, [float("inf")] * 6])
def test_nonfinite_or_wrong_shape_increment_rejected(monkeypatch, increment):
    with pytest.raises(RuntimeError, match="invalid six-dimensional"):
        refine(monkeypatch, increment)


def test_single_direction_or_nonfinite_score_not_accepted():
    assert not st.improves(dict(forward=.8, reverse=1.1, total=.95), score(1))
    assert not st.improves(score(.5), score(float("nan")))
    assert not st.improves(score(float("nan")), score(1))
    assert not st.improves(score(1), score(1))
    assert st.improves(score(.5), score(1))
    with pytest.raises(ValueError):
        st.improves(score(.5), score(1), 1)


def test_fixed_support_is_deterministic_and_not_recropped(monkeypatch):
    rng = np.random.default_rng(42)
    points = rng.uniform(-1, 1, (300, 3))
    camera = points + [.2, 0, 0]
    calls = []
    monkeypatch.setattr(st, "fov_crop_points", lambda *args: calls.append(1) or points.copy())
    monkeypatch.setattr(st, "camera_cloud", lambda *args: camera.copy())
    frame = NS(points_lidar=points, K=np.eye(3), depth=np.ones((32, 32)))
    guard = st.FixedSupportGeometry([frame], np.eye(4), NS(), max_points=100, voxel=.01, trunc=1)
    again = st.FixedSupportGeometry([frame], np.eye(4), NS(), max_points=100, voxel=.01, trunc=1)
    np.testing.assert_array_equal(guard.clouds[0][0], again.clouds[0][0])
    np.testing.assert_array_equal(guard.clouds[0][1], again.clouds[0][1])
    before = guard.clouds[0][0].copy()
    moved = np.eye(4); moved[0, 3] = .2
    guard(np.eye(4)); guard(moved)
    assert len(calls) == 2
    np.testing.assert_array_equal(before, guard.clouds[0][0])
    with pytest.raises(ValueError, match="invalid candidate"):
        guard(np.full((4, 4), np.nan))


def test_window_guard_disabled_preserves_raw_candidate():
    candidate = np.eye(4); candidate[0, 3] = 2
    result, diagnostics = eva.guard_window_result([], np.eye(4), candidate, cfg(False))
    assert result is candidate and not diagnostics


def test_empty_guard_window_retains_prior():
    prior = np.eye(4); prior[1, 3] = .7
    result, diagnostics = eva.guard_window_result([], prior, np.eye(4), cfg())
    np.testing.assert_array_equal(result, prior)
    assert diagnostics["geometric_update_accepted"] is False
    assert diagnostics["accuracy_verified"] is False


@pytest.mark.parametrize("accepted", [False, True])
def test_window_guard_rechecks_fused_candidate(monkeypatch, accepted):
    prior = np.eye(4)
    candidate = np.eye(4); candidate[0, 3] = .2
    def fixed_guard(T):
        return score(.5 if accepted and T[0, 3] else 1)
    monkeypatch.setattr(eva, "_geometry_guard", lambda *args: fixed_guard)
    result, diagnostics = eva.guard_window_result([object()], prior, candidate, cfg())
    np.testing.assert_array_equal(result, candidate if accepted else prior)
    assert diagnostics["geometric_update_accepted"] is accepted


@pytest.mark.parametrize("enabled", [False, True])
def test_all_invalid_frames_return_prior(monkeypatch, enabled):
    monkeypatch.setattr(eva, "camera_cloud", lambda *args: np.zeros((0, 3)))
    prior = np.eye(4); prior[1, 3] = .7
    result = eva.eva_calibrate_frames([object(), object()], prior,
                                      NS(eval=lambda: None), cfg(enabled), "cpu")
    np.testing.assert_array_equal(result["T_cam_lidar"], prior)
    assert not np.asarray(result["scores"]).any()
    if enabled:
        assert result["stability"]["geometric_update_accepted"] is False


def test_insufficient_support_does_not_call_network(monkeypatch):
    monkeypatch.setattr(eva, "camera_cloud", lambda *args: np.zeros((500, 3)))
    monkeypatch.setattr(eva, "generate_cdp", lambda *args: np.zeros((1, 1)))
    def insufficient(*args):
        raise st.InsufficientGeometryError("test sparse LiDAR")
    monkeypatch.setattr(eva, "_geometry_guard", insufficient)
    monkeypatch.setattr(eva, "_eva_step", lambda *args: pytest.fail("unsafe forward"))
    prior = np.eye(4)
    result = eva.eva_calibrate_frames([object()], prior, NS(eval=lambda: None), cfg(), "cpu")
    np.testing.assert_array_equal(result["T_cam_lidar"], prior)
    refined, _ = eva.eva_refine(object(), prior, NS(eval=lambda: None), cfg(), "cpu")
    np.testing.assert_array_equal(refined, prior)
