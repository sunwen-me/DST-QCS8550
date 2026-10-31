#!/usr/bin/env python3
"""Isolated evaluator parity/timing probe; no sensor subscriptions or TF writes."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from dst_calib.qnn_eva import QnnEvaluator


def stats(times):
    return {"n": len(times), "p50_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)), "mean_ms": float(np.mean(times))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--runtime", required=True)
    p.add_argument("--library", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--cpu-threads", type=int, default=0, help="0 preserves Torch's default thread count")
    a = p.parse_args()
    d = Path(a.model_dir)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    samples = [np.fromfile(d / f"sample_{i}.f32", dtype=np.float32).reshape(1, 3, 256, 512) for i in range(3)]
    references = [np.fromfile(d / f"reference_{i}.f32", dtype=np.float32).reshape(1, 6) for i in range(3)]
    report = {"scope": "single evaluator forward, identical saved inputs on one board; not calibration accuracy"}
    kwargs = dict(context=d / "eva_sb_qairt250.bin", runtime=a.runtime,
                  library=a.library, manifest=d / "export_manifest.json")
    with QnnEvaluator(**kwargs, profile=True) as model:
        report["init_profiled_ms"] = model.init_ms
        report["layout"] = model.layout
        report["samples"] = []
        for i, (x, reference) in enumerate(zip(samples, references)):
            result = model.infer(x)
            result.tofile(out / f"qnn_{i}.f32")
            delta = (result - reference).reshape(6)
            report["samples"].append({"sample": i, "max_abs_error": float(np.max(np.abs(delta))),
                 "rotation_vector_delta_deg": float(np.degrees(np.linalg.norm(delta[:3]))),
                 "translation_delta_m": float(np.linalg.norm(delta[3:])),
                 "qnn": result.reshape(6).tolist(), "torch_reference": reference.reshape(6).tolist()})
        model.write_profile(out / "htp_profile.jsonl")
    with QnnEvaluator(**kwargs) as model:
        report["init_unprofiled_ms"] = model.init_ms
        for i in range(10):
            model.infer(samples[i % 3])
        model.times_ms.clear()
        for i in range(a.repeats):
            model.infer(samples[i % 3])
        report["qnn_warm"] = stats(model.times_ms)
    (out / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    # CPU uses the same learned weights and original (not rewritten) Torch model.
    import torch
    if a.cpu_threads > 0:
        torch.set_num_threads(a.cpu_threads)
    from dst_calib.eva_infer import load_eva
    report["cpu_threads"] = torch.get_num_threads()
    start = time.perf_counter()
    model, _ = load_eva(str(d / "best.pt"), "cpu")
    report["cpu_model_load_ms"] = (time.perf_counter() - start) * 1000
    tensors = [torch.from_numpy(x) for x in samples]
    with torch.no_grad():
        for i in range(10):
            model(tensors[i % 3])
        times = []
        for i in range(a.repeats):
            start = time.perf_counter()
            model(tensors[i % 3])
            times.append((time.perf_counter() - start) * 1000)
        report["cpu_warm"] = stats(times)
        report["board_cpu_reference_max_abs_error"] = [float(np.max(np.abs(model(x).numpy() - ref))) for x, ref in zip(tensors, references)]
    report["forward_p50_speedup"] = report["cpu_warm"]["p50_ms"] / report["qnn_warm"]["p50_ms"]
    (out / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
