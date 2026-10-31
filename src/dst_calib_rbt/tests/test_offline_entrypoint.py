import importlib.util
from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("dst_offline", ROOT / "scripts/run_dst_offline.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def invoke(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["run_dst_offline.py", *map(str,args)])
    return entry.main()


def test_selfsup_cannot_claim_npu(monkeypatch):
    with pytest.raises(SystemExit) as exc:
        invoke(monkeypatch, "--input", "unused", "--backend", "qnn")
    assert exc.value.code == 2


def test_fast_requires_explicit_prior(monkeypatch):
    with pytest.raises(SystemExit) as exc:
        invoke(monkeypatch, "--input", "unused", "--mode", "fast")
    assert exc.value.code == 2


def test_existing_result_never_overwritten(monkeypatch, tmp_path):
    window = tmp_path / "window.bin"; window.touch()
    cfg = tmp_path / "cfg.yaml"; cfg.touch()
    output = tmp_path / "result.yaml"; output.write_text("KEEP")
    with pytest.raises(SystemExit) as exc:
        invoke(monkeypatch, "--input", window, "--config", cfg, "--output", output)
    assert exc.value.code == 2 and output.read_text() == "KEEP"


def test_saved_window_only_dispatch(monkeypatch, tmp_path):
    window = tmp_path / "window.bin"; window.touch()
    cfg = tmp_path / "cfg.yaml"; cfg.touch()
    output = tmp_path / "result.yaml"
    calls = []
    class Done:
        returncode = 0
    monkeypatch.setattr(entry.subprocess, "run", lambda cmd, cwd: calls.append((cmd,cwd)) or Done())
    assert invoke(monkeypatch, "--input", window, "--config", cfg, "--output", output) == 0
    assert len(calls) == 1
    cmd, cwd = calls[0]
    assert cmd[2].endswith("dst_calib_rbt/scripts/dst_worker.py")
    assert "--backend" in cmd and "cpu" in cmd and cwd == ROOT
    assert not any("composer" in s or "run_dst_calib.sh" in s for s in cmd)


@pytest.mark.parametrize("allow_unguarded", [False, True])
def test_fast_guard_default_and_explicit_ablation(monkeypatch, tmp_path, allow_unguarded):
    window = tmp_path / "window.bin"; window.touch()
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("sensors: {camera: {depth_min: 1.0, depth_max: 60.0}}\ninference: {stability_guard: {voxel: 0.2}}\n")
    prior = tmp_path / "prior.yaml"; prior.touch()
    model = tmp_path / "model"; model.mkdir(); (model / "best.pt").touch()
    output = tmp_path / "result.yaml"
    calls = []
    class Done:
        returncode = 0
    monkeypatch.setattr(entry.subprocess, "run", lambda cmd, cwd: calls.append(cmd) or Done())
    extra = ["--allow-unguarded"] if allow_unguarded else []
    assert invoke(monkeypatch, "--input", window, "--config", cfg, "--init-yaml", prior,
                  "--model-dir", model, "--output", output, "--mode", "fast", *extra) == 0
    snapshot = yaml.safe_load(output.with_suffix(".config.yaml").read_text())
    assert snapshot["inference"]["stability_guard"]["enabled"] is (not allow_unguarded)
    assert snapshot["inference"]["stability_guard"]["voxel"] == .2
    assert snapshot["sensors"]["camera"]["depth_max"] == 60.0
    assert "enabled" not in yaml.safe_load(cfg.read_text())["inference"]["stability_guard"]
    assert calls[0][calls[0].index("--config")+1] == str(output.with_suffix(".config.yaml"))
    with pytest.raises(SystemExit) as exc:
        invoke(monkeypatch, "--input", window, "--config", cfg, "--init-yaml", prior,
               "--model-dir", model, "--output", output, "--mode", "fast")
    assert exc.value.code == 2


def test_v2_profile_is_guarded_save_only_and_not_auto_approved():
    import json
    profile = yaml.safe_load((ROOT / "config/dst_calib_rbt/deployment-v2-experimental.yaml").read_text())
    cfg = yaml.safe_load((ROOT / profile["worker"]["config"]).read_text())
    assert profile["output"]["publish_results"] is False
    assert profile["worker"]["backend"] == "qnn"
    assert cfg["inference"]["stability_guard"]["enabled"] is True
    assert cfg["sensors"]["camera"]["depth_min"] == .25
    assert cfg["sensors"]["camera"]["depth_max"] == 6.0
    manifest = json.loads((ROOT / profile["worker"]["qnn_manifest"]).read_text())
    assert manifest["pipeline_validation"]["auto_backend_approved"] is False
    assert manifest["pipeline_validation"]["live_sensor_accuracy_verified"] is False
