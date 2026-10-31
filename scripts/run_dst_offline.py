#!/usr/bin/env python3
"""Practical saved-window entrypoint: no RBT, sensor services or TF publisher."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path, help="saved DSTWIN01 window.bin")
    p.add_argument("--config", type=Path, default=Path("src/dst_calib_rbt/python/default.yaml"),
                   help="default Gemini depth config; KITTI needs its own config")
    p.add_argument("--init-yaml", type=Path, help="independently checked prior; required for fast/both")
    p.add_argument("--mode", choices=("selfsup", "fast", "both"), default="selfsup")
    p.add_argument("--backend", choices=("cpu", "qnn"), default="cpu")
    p.add_argument("--model-dir", type=Path, default=Path("models/dst-calib/deployment-v2"))
    p.add_argument("--allow-unguarded", action="store_true",
                   help="explicit fast/both ablation only; disables fixed-support geometry protection")
    p.add_argument("--output", type=Path, help="default unique runs/dst-offline.*/extrinsic.yaml")
    p.add_argument("--python", default=os.environ.get("DST_CALIB_PYTHON", ""))
    p.add_argument("--qnn-runtime", type=Path, default=Path(os.environ.get("DST_QNN_RUNTIME", "runtime/qairt250")))
    p.add_argument("--qnn-library", type=Path, default=Path("build/dst-calib/lib/libdst_qnn.so"))
    p.add_argument("--qnn-profile", type=Path)
    a = p.parse_args()
    # All relative paths consistently resolve against the workspace, not shell cwd.
    def resolve(path):
        return path if path.is_absolute() else root / path
    for name in ("input", "config", "init_yaml", "model_dir", "output", "qnn_runtime", "qnn_library", "qnn_profile"):
        value = getattr(a, name)
        if value is not None:
            setattr(a, name, resolve(value))
    if a.mode == "selfsup" and a.backend == "qnn":
        p.error("selfsup uses CPU autograd; QNN accelerates only the fast/both evaluator")
    if a.mode != "selfsup" and a.init_yaml is None:
        p.error("experimental fast/both needs an explicit independently checked --init-yaml")
    required = [a.input, a.config]
    if a.init_yaml is not None:
        required.append(a.init_yaml)
    if a.mode != "selfsup":
        required.append(a.model_dir / "best.pt")
    if a.backend == "qnn":
        required.extend([a.model_dir / "eva_sb_qairt250.bin", a.model_dir / "export_manifest.json", a.qnn_library])
    for file in required:
        if not file.is_file():
            p.error(f"required file not found: {file}")
    if a.output is not None and a.output.exists():
        p.error(f"refuse to overwrite an existing result: {a.output}")
    if a.mode != "selfsup" and a.output is not None and a.output.with_suffix(".config.yaml").exists():
        p.error(f"refuse to overwrite an existing config snapshot: {a.output.with_suffix('.config.yaml')}")
    if a.output is None:
        (root / "runs").mkdir(exist_ok=True)
        a.output = Path(tempfile.mkdtemp(prefix="dst-offline.", dir=root / "runs")) / "extrinsic.yaml"
    if a.mode != "selfsup":
        import yaml
        raw = yaml.safe_load(a.config.read_text())
        if not isinstance(raw, dict):
            p.error("config must contain a YAML mapping")
        inference = raw.setdefault("inference", {})
        guard = inference.setdefault("stability_guard", {})
        guard["enabled"] = not a.allow_unguarded
        a.output.parent.mkdir(parents=True, exist_ok=True)
        snapshot = a.output.with_suffix(".config.yaml")
        with snapshot.open("x") as handle:
            yaml.safe_dump(raw, handle, sort_keys=False)
        a.config = snapshot
        print(f"Geometry protection={'OFF (explicit ablation)' if a.allow_unguarded else 'ON'}; "
              "retaining a prior is not proof of calibration accuracy.", flush=True)
    python = a.python or sys.executable
    cmd = [python, "-u", str(root / "src/dst_calib_rbt/scripts/dst_worker.py"),
           "--input", str(a.input), "--config", str(a.config), "--mode", a.mode,
           "--backend", a.backend, "--output", str(a.output)]
    if a.init_yaml is not None:
        cmd.extend(["--init-yaml", str(a.init_yaml)])
    if a.mode != "selfsup":
        cmd.extend(["--eva-ckpt", str(a.model_dir / "best.pt")])
    if a.backend == "qnn":
        cmd.extend(["--qnn-context", str(a.model_dir / "eva_sb_qairt250.bin"),
                    "--qnn-manifest", str(a.model_dir / "export_manifest.json"),
                    "--qnn-runtime", str(a.qnn_runtime), "--qnn-library", str(a.qnn_library)])
        if a.qnn_profile is not None:
            cmd.extend(["--qnn-profile", str(a.qnn_profile)])
    print(f"Offline candidate ONLY; no RBT/TF. mode={a.mode} backend={a.backend} output={a.output}", flush=True)
    return subprocess.run(cmd, cwd=root).returncode


if __name__ == "__main__":
    raise SystemExit(main())
