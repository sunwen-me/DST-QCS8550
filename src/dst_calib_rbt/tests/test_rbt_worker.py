"""RBT adapter regressions, separate from the copied algorithm tests."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dst_rbt_worker", ROOT / "scripts/dst_worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
worker._load_dst_package()


def args_for(tmp_path, **overrides):
    args = dict(input=str(tmp_path / "window.bin"),
                output=str(tmp_path / "extrinsic.yaml"), config="unused",
                mode="selfsup", backend="auto", init_yaml="",
                eva_onnx=str(tmp_path / "absent.onnx"),
                eva_ckpt=str(tmp_path / "absent.pt"))
    args.update(overrides)
    return SimpleNamespace(**args)


@pytest.mark.parametrize("with_prior", [False, True])
def test_selfsup_preserves_coarse_search_choice(monkeypatch, tmp_path, with_prior):
    import dst_calib.config as config
    import dst_calib.self_supervised as ss

    cfg = SimpleNamespace()
    frames = [object()]
    monkeypatch.setattr(worker, "read_window", lambda path: frames)
    monkeypatch.setattr(config, "load_config", lambda path: cfg)

    def calibrate(actual_frames, actual_cfg):
        assert actual_frames is frames
        if with_prior:
            np.testing.assert_allclose(actual_cfg.init_T, np.eye(4))
        else:
            assert getattr(actual_cfg, "init_T", None) is None
        return dict(T_cam_lidar=np.eye(4), final_cd=0.01)

    monkeypatch.setattr(ss, "calibrate_self_supervised", calibrate)
    args = args_for(tmp_path)
    if with_prior:
        prior = tmp_path / "prior.yaml"
        prior.write_text(yaml.safe_dump(dict(T_cam_lidar=np.eye(4).tolist())))
        args.init_yaml = str(prior)
    worker._run(args)
    assert yaml.safe_load(Path(args.output).read_text())["final_cd"] == 0.01


@pytest.mark.parametrize("mode", ["fast", "both"])
def test_torch_reuses_full_original_pipeline(monkeypatch, tmp_path, mode):
    import dst_calib.calibrate as calibrate

    checkpoint = tmp_path / "evaluation.pt"
    checkpoint.touch()  # Dispatch test only; no claim that this is a trained model.
    cfg = SimpleNamespace(inference=SimpleNamespace(eva_iters=7))
    frames = [object()]

    def run(actual_frames, actual_cfg, actual_mode, actual_path, iters):
        assert actual_frames is frames and actual_cfg is cfg
        assert actual_mode == mode and actual_path == str(checkpoint) and iters == 7
        return dict(T_cam_lidar=np.eye(4), final_cd=0.02)

    monkeypatch.setattr(calibrate, "_calibrate_with_eva", run)
    T, cd, backend = worker._fast_torch(frames, cfg, checkpoint, mode)
    np.testing.assert_allclose(T, np.eye(4))
    assert cd == 0.02 and backend in ("cpu", "cuda")


@pytest.mark.parametrize("mode", ["selfsup"])
def test_explicit_unsupported_qnn_is_not_silently_cpu(tmp_path, mode):
    with pytest.raises(ValueError, match="QNN"):
        worker._run(args_for(tmp_path, mode=mode, backend="qnn"))


def test_evaluator_missing_checkpoint_explains_selfsup(tmp_path):
    with pytest.raises(FileNotFoundError, match="--mode selfsup"):
        worker._fast_torch([], SimpleNamespace(), tmp_path / "missing.pt")


def test_invalid_result_is_not_written(monkeypatch, tmp_path):
    import dst_calib.config as config
    import dst_calib.self_supervised as ss

    monkeypatch.setattr(worker, "read_window", lambda path: [object()])
    monkeypatch.setattr(config, "load_config", lambda path: SimpleNamespace())
    monkeypatch.setattr(ss, "calibrate_self_supervised",
                        lambda *args: dict(T_cam_lidar=np.eye(4), final_cd=float("nan")))
    args = args_for(tmp_path)
    with pytest.raises(RuntimeError, match="invalid SE"):
        worker._run(args)
    assert not Path(args.output).exists()


def test_guard_status_written_without_accuracy_claim(monkeypatch, tmp_path):
    import dst_calib.config as config
    cfg = SimpleNamespace()
    status = dict(enabled=True, geometric_update_accepted=False,
                  accuracy_verified=False, reason="prior_retained")
    monkeypatch.setattr(worker, "read_window", lambda _: [object()])
    monkeypatch.setattr(config, "load_config", lambda _: cfg)
    def run(*args):
        cfg._dst_stability_result = status
        return np.eye(4), .01, "cpu"
    monkeypatch.setattr(worker, "_fast_torch", run)
    args = args_for(tmp_path, mode="fast", backend="cpu")
    worker._run(args)
    result = yaml.safe_load(Path(args.output).read_text())
    assert result["stability"] == status


@pytest.mark.parametrize("mode", ["fast", "both"])
def test_qnn_keeps_original_pipeline(monkeypatch, tmp_path, mode):
    import dst_calib.calibrate as calibrate
    import dst_calib.qnn_eva as qnn
    import torch

    class FakeEvaluator:
        _dst_eva_meta = {"arch": "sb", "virtual_camera": {"height": 256, "width": 512, "focal": 600}}
        times_ms = [5.0, 6.0]
        init_ms = 100.0
        layout = "NCHW"
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass

    monkeypatch.setattr(qnn, "QnnEvaluator", FakeEvaluator)
    def run(frames, cfg, actual_mode, path, iters, *, evaluator):
        assert actual_mode == mode and iters == 3
        assert isinstance(evaluator[0], FakeEvaluator)
        assert evaluator[2] == torch.device("cpu")
        return dict(T_cam_lidar=np.eye(4), final_cd=0.03)
    monkeypatch.setattr(calibrate, "_calibrate_with_eva", run)
    (tmp_path / "export_manifest.json").write_text("{}")
    args = args_for(tmp_path, mode=mode, backend="qnn", qnn_context=str(tmp_path / "model.bin"),
                    qnn_library="fake", qnn_manifest="", qnn_profile="", qnn_runtime="fake")
    T, cd, backend = worker._fast_qnn([], SimpleNamespace(), args)
    assert backend == "qnn" and cd == 0.03
    assert args._evaluator_metrics["calls"] == 2


def test_auto_qnn_requires_pipeline_approval(tmp_path):
    context = tmp_path / "model.bin"
    library = tmp_path / "model.so"
    manifest = tmp_path / "export_manifest.json"
    context.touch(); library.touch()
    args = args_for(tmp_path, qnn_context=str(context), qnn_library=str(library), qnn_manifest=str(manifest))
    manifest.write_text(json.dumps({"pipeline_validation": {"auto_backend_approved": False}}))
    assert not worker._qnn_auto_approved(args)
    manifest.write_text(json.dumps({"pipeline_validation": {"auto_backend_approved": True}}))
    assert worker._qnn_auto_approved(args)


@pytest.mark.parametrize("mode", ["fast", "both"])
def test_qnn_error_never_falls_back(monkeypatch, tmp_path, mode):
    import dst_calib.config as config
    monkeypatch.setattr(worker, "read_window", lambda _: [object()])
    monkeypatch.setattr(config, "load_config", lambda _: SimpleNamespace())
    def fail(*args): raise RuntimeError("HTP failure")
    monkeypatch.setattr(worker, "_fast_qnn", fail)
    def forbidden(*args): pytest.fail("explicit QNN must not silently use Torch")
    monkeypatch.setattr(worker, "_fast_torch", forbidden)
    with pytest.raises(RuntimeError, match="HTP failure"):
        worker._run(args_for(tmp_path, mode=mode, backend="qnn"))


def test_qnn_context_hash_is_checked_before_loading_library(tmp_path):
    from dst_calib.qnn_eva import QnnEvaluator
    context = tmp_path / "model.bin"
    context.write_bytes(b"not the expected context")
    manifest = tmp_path / "export_manifest.json"
    manifest.write_text(json.dumps({"qnn": {"context_sha256": "bad"}}))
    with pytest.raises(ValueError, match="SHA256"):
        QnnEvaluator(context, "unused", "unused", manifest)
