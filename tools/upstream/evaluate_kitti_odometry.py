#!/usr/bin/env python3
"""Evaluate a KITTI-trained checkpoint on fixed, held-out perturbations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import ConcatDataset, DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/dst_calib_rbt/python"))
from dst_calib.config import load_config
from dst_calib.eva_infer import load_eva
from dst_calib.train import DoubleSidedDataset, _resolve_session_T, evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", nargs="+", required=True)
    p.add_argument("--config", default=str(ROOT / "config/kitti_odometry.yaml"))
    p.add_argument("--out", required=True)
    p.add_argument("--samples-per-frame", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=2000003)
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args()
    torch.set_num_threads(2)
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_eva(args.checkpoint, device)
    subsets = []
    for i, directory in enumerate(args.data_dir):
        T, _ = _resolve_session_T(directory, None)
        subsets.append(DoubleSidedDataset(
            directory, cfg, T, samples_per_frame=args.samples_per_frame,
            cloud_points=2048, seed=args.seed + 7919 * i, deterministic=True))
    dataset = subsets[0] if len(subsets) == 1 else ConcatDataset(subsets)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    metrics = evaluate(model, loader, meta["arch"], device)
    report = {"checkpoint": str(Path(args.checkpoint).resolve()), "metadata": meta,
              "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
              "data_dirs": args.data_dir, "seed": args.seed, "metrics": metrics,
              "protocol": "fixed double-sided perturbations, one network correction; not odometry leaderboard scores"}
    if len(args.data_dir) == 1:
        manifest = Path(args.data_dir[0]).resolve().parents[2] / "manifest.json"
        if manifest.is_file():
            report["dataset_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
