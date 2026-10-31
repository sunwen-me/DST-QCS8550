#!/usr/bin/env python3
"""Run the DST algorithm on one RBT-owned calibration window.

The native component owns transport and synchronization. This worker only
receives a compact binary snapshot, runs the upstream DST geometry, and
returns one YAML result. The learned evaluator uses persistent native QNN HTP;
geometry and self-supervised optimization remain on CPU.
"""

from __future__ import annotations

import argparse
import json
import time
import os
import struct
import sys
from pathlib import Path

import numpy as np
import yaml


def _load_dst_package() -> None:
    package_root = Path(__file__).resolve().parents[1] / "python"
    sys.path.insert(0, str(package_root))


def _read_exact(handle, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise ValueError("DST worker input is truncated")
    return data


def _read_scalar(handle, fmt: str):
    return struct.unpack(fmt, _read_exact(handle, struct.calcsize(fmt)))[0]


def read_window(path: Path):
    from dst_calib.self_supervised import Frame

    with path.open("rb") as handle:
        if _read_exact(handle, 8) != b"DSTWIN01":
            raise ValueError("DST worker input magic is invalid")
        version = _read_scalar(handle, "<I")
        if version != 1:
            raise ValueError(f"unsupported DST worker input version {version}")
        count = _read_scalar(handle, "<I")
        width = _read_scalar(handle, "<I")
        height = _read_scalar(handle, "<I")
        fx, fy, cx, cy = (_read_scalar(handle, "<d") for _ in range(4))
        if count == 0 or width == 0 or height == 0 or min(fx, fy) <= 0:
            raise ValueError("DST worker input camera metadata is invalid")
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        frames = []
        for _ in range(count):
            stamp = _read_scalar(handle, "<Q")
            point_count = _read_scalar(handle, "<I")
            point_bytes = _read_exact(handle, point_count * 3 * 4)
            points = np.frombuffer(point_bytes, dtype="<f4").reshape(point_count, 3).copy()
            depth_count = _read_scalar(handle, "<I")
            if depth_count != width * height:
                raise ValueError("DST worker depth geometry does not match camera metadata")
            depth_bytes = _read_exact(handle, depth_count * 4)
            depth = np.frombuffer(depth_bytes, dtype="<f4").reshape(height, width).copy()
            frames.append(Frame(points_lidar=points, depth=depth, K=K.copy(), rgb=None, stamp=float(stamp) * 1e-9))
    return frames


def _nominal_transform() -> np.ndarray:
    # Livox x-forward/y-left/z-up to camera optical x-right/y-down/z-forward.
    return np.array([[0.0, -1.0, 0.0, 0.0],
                     [0.0, 0.0, -1.0, 0.0],
                     [1.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)


def _load_init_yaml(path: str) -> np.ndarray:
    if not path:
        return _nominal_transform()
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    matrix = data.get("T_cam_lidar", data) if isinstance(data, dict) else data
    T = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    if not np.all(np.isfinite(T)) or not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("init YAML contains an invalid T_cam_lidar")
    R = T[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-2) or np.linalg.det(R) <= 0.0:
        raise ValueError("init YAML rotation is not a proper SE(3) rotation")
    return T


def _fast_qnn(frames, cfg, args):
    import torch
    from dst_calib.calibrate import _calibrate_with_eva
    from dst_calib.qnn_eva import QnnEvaluator

    context = Path(args.qnn_context)
    manifest = Path(args.qnn_manifest) if args.qnn_manifest else context.parent / "export_manifest.json"
    validation = json.loads(manifest.read_text()).get("pipeline_validation", {})
    approved = validation.get("auto_backend_approved") is True
    if not approved:
        print("[dst_worker] experimental HTP evaluator: full-pipeline numerical/accuracy acceptance has NOT passed", file=sys.stderr)
    with QnnEvaluator(context, args.qnn_runtime, args.qnn_library, manifest,
                      profile=bool(args.qnn_profile)) as model:
        result = _calibrate_with_eva(
            frames, cfg, args.mode, str(context),
            int(getattr(getattr(cfg, "inference", None), "eva_iters", 3)),
            evaluator=(model, model._dst_eva_meta, torch.device("cpu")))
        if args.qnn_profile:
            profile = Path(args.qnn_profile)
            profile.parent.mkdir(parents=True, exist_ok=True)
            model.write_profile(profile)
        args._evaluator_metrics = {
            "backend": "qnn_htp", "layout": model.layout, "init_ms": model.init_ms,
            "calls": len(model.times_ms), "forward_total_ms": float(sum(model.times_ms)),
            "forward_p50_ms": float(np.median(model.times_ms)) if model.times_ms else None,
            "geometry_backend": "cpu", "self_supervised_backend": "cpu",
            "pipeline_approved": approved, "context": str(context),
            "score_method": result.get("score_method", "unknown"),
        }
    cfg._dst_stability_result = result.get("stability", {})
    return np.asarray(result["T_cam_lidar"], dtype=np.float64), float(result["final_cd"]), "qnn"


def _qnn_auto_approved(args):
    """A compiled graph alone is not sufficient for automatic deployment."""
    context = Path(getattr(args, "qnn_context", "__absent__"))
    library = Path(getattr(args, "qnn_library", "__absent__"))
    if not context.is_file() or not library.is_file():
        return False
    manifest = Path(args.qnn_manifest) if getattr(args, "qnn_manifest", "") else context.parent / "export_manifest.json"
    try:
        return json.loads(manifest.read_text()).get("pipeline_validation", {}).get("auto_backend_approved") is True
    except (OSError, ValueError, TypeError):
        return False


def _fast_torch(frames, cfg, checkpoint: Path, mode: str = "fast"):
    from dst_calib.calibrate import _calibrate_with_eva
    from dst_calib.self_supervised import _device

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"mode={mode} requires a trained evaluation checkpoint: {checkpoint}; "
            "use --mode selfsup for calibration without pretrained weights")
    # Reuse the reproduction's complete fast/both pipeline, including checkpoint
    # camera metadata, evaluator priors, final refinement and the fast guard.
    result = _calibrate_with_eva(
        frames, cfg, mode, str(checkpoint),
        int(getattr(getattr(cfg, "inference", None), "eva_iters", 3)))
    cfg._dst_stability_result = result.get("stability", {})
    return (np.asarray(result["T_cam_lidar"], dtype=np.float64),
            float(result["final_cd"]), _device().type)


def _run(args):
    if args.mode == "selfsup" and args.backend == "qnn":
        raise ValueError(
            "selfsup uses Torch autograd/Chamfer optimization, not the QNN "
            "inference backend; use --backend cpu or auto")
    start = time.perf_counter()
    if args.backend in ("cpu", "qnn"):
        # Set before importing Torch through the DST package.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    _load_dst_package()
    from dst_calib.config import load_config
    from dst_calib.geometry import euler_zyx_from_R
    from dst_calib.self_supervised import calibrate_self_supervised

    frames = read_window(Path(args.input))
    cfg = load_config(args.config)
    T_init = _load_init_yaml(args.init_yaml)
    # No supplied prior means the original selfsup coarse search must run.
    # The nominal optical transform is only the fast evaluator's fallback.
    if args.init_yaml:
        cfg.init_T = T_init
    selected_backend = "cpu"
    if args.mode in ("fast", "both"):
        use_qnn = args.backend == "qnn" or (args.backend == "auto" and _qnn_auto_approved(args))
        if use_qnn:
            try:
                T, final_cd, selected_backend = _fast_qnn(frames, cfg, args)
            except Exception as error:
                if args.backend == "qnn":
                    raise
                print(f"[dst_worker] HTP unavailable; auto fallback to Torch: {error}", file=sys.stderr)
                T, final_cd, selected_backend = _fast_torch(frames, cfg, Path(args.eva_ckpt), args.mode)
        else:
            T, final_cd, selected_backend = _fast_torch(frames, cfg, Path(args.eva_ckpt), args.mode)
    else:
        result = calibrate_self_supervised(frames, cfg)
        T = np.asarray(result["T_cam_lidar"], dtype=np.float64)
        final_cd = float(result["final_cd"])
        from dst_calib.self_supervised import _device
        selected_backend = _device().type

    R = T[:3, :3]
    if (T.shape != (4, 4) or not np.isfinite(T).all()
            or not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
            or not np.allclose(R @ R.T, np.eye(3), atol=1e-3)
            or not np.isclose(np.linalg.det(R), 1.0, atol=1e-3)
            or not np.isfinite(final_cd)):
        raise RuntimeError("DST calibration returned an invalid SE(3) transform or score")

    output = {
        "T_cam_lidar": [[float(value) for value in row] for row in T],
        "rotation_deg": [float(value) for value in np.degrees(euler_zyx_from_R(T[:3, :3]))],
        "translation_m": [float(value) for value in T[:3, 3]],
        "final_cd": final_cd,
        "mode": args.mode,
        "backend": selected_backend,
        "frames": len(frames),
        "worker_seconds": time.perf_counter() - start,
    }
    if getattr(cfg, "_dst_stability_result", None):
        output["stability"] = cfg._dst_stability_result
    if hasattr(args, "_evaluator_metrics"):
        output["evaluator"] = args._evaluator_metrics
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(output, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("selfsup", "fast", "both"), default="selfsup")
    parser.add_argument("--backend", choices=("auto", "qnn", "cpu"), default="auto")
    parser.add_argument("--eva-onnx", default="models/dst-calib/eva_sb.onnx",
                        help="legacy option; execution uses Torch or native --qnn-context, not ORT EP")
    parser.add_argument("--eva-ckpt", default="models/dst-calib/kitti-validation-best/best.pt")
    parser.add_argument("--init-yaml", default="")
    parser.add_argument("--qnn-context", default="models/dst-calib/kitti-validation-best/eva_sb_qairt250.bin")
    parser.add_argument("--qnn-runtime", default=os.environ.get("DST_QNN_RUNTIME", "runtime/qairt250"))
    parser.add_argument("--qnn-library", default="build/dst-calib/lib/libdst_qnn.so")
    parser.add_argument("--qnn-manifest", default="")
    parser.add_argument("--qnn-profile", default="")
    args = parser.parse_args()
    # Dynamic linker reads LD_LIBRARY_PATH at process creation; re-exec once.
    if args.backend == "qnn" or (args.backend == "auto" and args.mode != "selfsup"
                                   and _qnn_auto_approved(args)):
        runtime = Path(args.qnn_runtime).resolve()
        marker = str(runtime)
        if os.environ.get("DST_QNN_ENV_READY") != marker:
            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = str(runtime / "lib/aarch64-oe-linux-gcc11.2") + ":" + env.get("LD_LIBRARY_PATH", "")
            env["ADSP_LIBRARY_PATH"] = str(runtime / "lib/hexagon-v73/unsigned") + ";" + env.get("ADSP_LIBRARY_PATH", "/usr/lib/rfsa/adsp;/dsp")
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["DST_QNN_ENV_READY"] = marker
            os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env)
    _run(args)


if __name__ == "__main__":
    main()
