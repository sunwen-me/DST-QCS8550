#!/usr/bin/env python3
"""Export the DST single-branch evaluator for ONNX/native QNN compilation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import numpy as np


class ExactAdaptivePool(torch.nn.Module):
    """Constant separable bin averages, including non-divisible output sizes."""
    def __init__(self, input_hw, output_hw):
        super().__init__()
        def matrix(n, m):
            weights = torch.zeros(m, n)
            for i in range(m):
                start = math.floor(i * n / m)
                stop = math.ceil((i + 1) * n / m)
                weights[i, start:stop] = 1.0 / (stop - start)
            return weights
        self.register_buffer("height_weights", matrix(input_hw[0], output_hw[0]))
        self.register_buffer("width_weights", matrix(input_hw[1], output_hw[1]).T)

    def forward(self, x):
        return torch.matmul(torch.matmul(self.height_weights, x), self.width_weights)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-window")
    parser.add_argument("--config")
    parser.add_argument("--init-yaml")
    args = parser.parse_args()
    package_root = Path(__file__).resolve().parents[1] / "python"
    sys.path.insert(0, str(package_root))
    from dst_calib.eva_infer import load_eva

    model, meta = load_eva(args.checkpoint, torch.device("cpu"))
    if meta.get("arch") != "sb":
        raise SystemExit("QNN/ONNX export currently supports only the DST single-branch evaluator (arch=sb)")
    model.eval()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    vc = meta.get("virtual_camera", {})
    height, width = int(vc.get("height", 256)), int(vc.get("width", 512))
    samples = [np.random.default_rng(0).normal(0, 1, (1, 3, height, width)).astype(np.float32)]
    if args.sample_window:
        if not args.config or not args.init_yaml:
            parser.error("--sample-window requires --config and --init-yaml")
        from dst_worker import read_window, _load_init_yaml
        from dst_calib.config import load_config
        from dst_calib.projection import generate_cdp, generate_ldp, virtual_camera_intrinsics
        from dst_calib.difference_map import build_difference_map
        from dst_calib.self_supervised import camera_cloud
        cfg = load_config(args.config)
        K = virtual_camera_intrinsics(float(vc.get("focal", 600.0)), (height, width))
        T = _load_init_yaml(args.init_yaml)
        for frame in read_window(Path(args.sample_window)):
            cdp = generate_cdp(camera_cloud(frame, cfg), np.eye(4), K, (height, width))
            ldp = generate_ldp(frame.points_lidar, T, K, (height, width))
            samples.append(build_difference_map(ldp, cdp, e_tar=cfg.difference_map.e_tar)[None].astype(np.float32))
    exported = copy.deepcopy(model)
    with torch.inference_mode():
        features = model.backbone(torch.zeros(1, 3, height, width))
    exported.head.pool = ExactAdaptivePool(tuple(features.shape[-2:]), (model.head.n * 4,) * 2)
    torch_refs = []
    for data in samples:
        with torch.inference_mode():
            ref = model(torch.from_numpy(data)).numpy()
            candidate = exported(torch.from_numpy(data)).numpy()
        np.testing.assert_allclose(candidate, ref, rtol=2e-5, atol=2e-6)
        torch_refs.append(ref)
    dummy = torch.from_numpy(samples[0])
    torch.onnx.export(
        exported,
        dummy,
        str(output),
        input_names=["diff_map"],
        output_names=["xi"],
        dynamic_axes=None,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(str(output))
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    errors = []
    for i, (data, ref) in enumerate(zip(samples, torch_refs)):
        actual = session.run(None, {"diff_map": data})[0]
        np.testing.assert_allclose(actual, ref, rtol=2e-4, atol=2e-5)
        errors.append(float(np.max(np.abs(actual - ref))))
        data.tofile(output.parent / f"sample_{i}.f32")
        ref.tofile(output.parent / f"reference_{i}.f32")
    manifest = dict(checkpoint=str(Path(args.checkpoint).resolve()),
        checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        onnx_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        virtual_camera=dict(height=height, width=width, focal=float(vc.get("focal", 600.0))),
        input=dict(name="diff_map", shape=[1, 3, height, width], layout="NCHW", dtype="float32"),
        output=dict(name="xi", shape=[1, 6], units="axis-angle radians, translation meters"),
        export_change="exact separable adaptive-average bins; export itself does not train; see checkpoint provenance",
        pytorch_onnx_max_abs_errors=errors, sample_count=len(samples))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("experiment"):
        manifest["training_experiment"] = checkpoint["experiment"]
        manifest["selection"] = checkpoint.get("selection", "unknown")
        manifest["selected_epoch"] = checkpoint.get("epoch")
    manifest["pipeline_validation"] = dict(auto_backend_approved=False,
        htp_execution_verified=False, live_sensor_accuracy_verified=False,
        full_fast_status="export only; full pipeline accuracy/HTP execution not yet validated")
    (output.parent / "export_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"exported {output}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
