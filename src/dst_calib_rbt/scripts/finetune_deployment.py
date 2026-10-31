#!/usr/bin/env python3
"""Isolated engineering fine-tune; never deploys, overwrites baseline, or publishes TF."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from dst_calib.config import load_config
from dst_calib.calibrate import _apply_ckpt_virtual_camera
from dst_calib.eva_infer import load_eva
from dst_calib.deployment_finetune import (DeploymentDataset, evaluate_trajectories,
    freeze_bn, load_records, mine_replay, normalized_residual_loss, sha256)


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True, help="prepared directory containing train/ and val/")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--samples-per-frame", type=int, default=4)
    p.add_argument("--train-per-session", type=int, default=0)
    p.add_argument("--val-per-session", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=20261031)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--eval-every", type=int, default=3)
    p.add_argument("--replay-frames", type=int, default=96)
    p.add_argument("--head-only", action="store_true")
    a = p.parse_args()
    if min(a.epochs, a.eval_every, a.batch_size, a.samples_per_frame) <= 0 or a.lr <= 0:
        p.error("epochs/eval-every/batch-size/samples-per-frame/lr must be positive")
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"refuse to overwrite nonempty experiment directory: {out}")
    torch.set_num_threads(2)
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    if a.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA unavailable")
    cfg = load_config(a.config)
    model, meta = load_eva(a.checkpoint, a.device)
    if meta["arch"] != "sb":
        raise ValueError("deployment experiment supports SB only")
    _apply_ckpt_virtual_camera(cfg, meta)
    root = Path(a.data_root).resolve()
    train_dirs = sorted((root / "train").glob("*/"))
    val_dirs = sorted((root / "val").glob("*/"))
    train_paths = {p.resolve() for d in train_dirs for p in d.glob("frame_*.npz")}
    val_paths = {p.resolve() for d in val_dirs for p in d.glob("frame_*.npz")}
    if not train_dirs or not val_dirs or train_paths & val_paths:
        raise ValueError("separate nonempty train/validation sessions required")
    provenance = dict(arguments=vars(a), baseline_sha256=sha256(a.checkpoint),
                      methodology="engineering normalized residual loss; not original paper equation16",
                      sampling="ceil-stride caps 12000/24000/60000/full; identity/small/medium/large equally sampled",
                      bn="running statistics frozen", test_data_used=False,
                      auto_backend_approved=False, live_sensor_accuracy_verified=False,
                      train_sessions=[str(d) for d in train_dirs], validation_sessions=[str(d) for d in val_dirs],
                      code_sha256=sha256(Path(__file__).resolve().parents[1] / "python/dst_calib/deployment_finetune.py"))
    write_json(out / "experiment.json", provenance)
    train = load_records(train_dirs, cfg, per_session=a.train_per_session)
    val = load_records(val_dirs, cfg, per_session=a.val_per_session)
    write_json(out / "split_files.json", dict(train=[r["path"] for r in train], validation=[r["path"] for r in val]))
    print(f"baseline validation: {len(val)} frames", flush=True)
    baseline = evaluate_trajectories(model, val, cfg, a.device)
    write_json(out / "baseline-validation.json", baseline)
    best_score, best_epoch = baseline["selection_score"], 0
    print(f"baseline selection score={best_score:.6f}", flush=True)
    if a.head_only:
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=1e-4)
    dataset = DeploymentDataset(train, cfg, seed=a.seed, samples_per_frame=a.samples_per_frame)
    history = []
    for epoch in range(1, a.epochs+1):
        start = time.perf_counter()
        if epoch in (1, 1+a.epochs//2) and a.replay_frames:
            dataset.replay = mine_replay(model, train, cfg, a.device, max_frames=a.replay_frames, seed=a.seed+epoch)
        dataset.epoch = epoch - 1
        generator = torch.Generator().manual_seed(a.seed+epoch)
        loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=True, num_workers=a.workers,
                            pin_memory=a.device=="cuda", generator=generator)
        freeze_bn(model)
        learning_rate = a.lr * (.2 + .8 * (1+math.cos(math.pi*(epoch-1)/a.epochs))/2)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        total = 0.
        for iteration, (diff, target) in enumerate(loader):
            diff, target = diff.to(a.device), target.to(a.device)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(diff)
            loss = normalized_residual_loss(predicted, target)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise RuntimeError("nonfinite gradient")
            optimizer.step()
            total += float(loss.detach()) * len(diff)
            if iteration % 100 == 0:
                print(f"epoch={epoch}/{a.epochs} batch={iteration}/{len(loader)} loss={float(loss.detach()):.6f}", flush=True)
        item = dict(epoch=epoch, loss=total/len(dataset), lr=learning_rate, seconds=time.perf_counter()-start)
        if epoch % a.eval_every == 0 or epoch == a.epochs:
            validation = evaluate_trajectories(model, val, cfg, a.device)
            write_json(out / f"validation-epoch{epoch:03d}.json", validation)
            item["validation_score"] = validation["selection_score"]
            checkpoint = dict(model=model.state_dict(), arch="sb", virtual_camera=meta.get("virtual_camera"),
                              epoch=epoch, loss=item["loss"], selection="raw_development_trajectory_score",
                              validation_score=validation["selection_score"], experiment=provenance,
                              pipeline_validation=dict(auto_backend_approved=False, accuracy_verified=False))
            torch.save(checkpoint, out / f"candidate-epoch{epoch:03d}.pt")
            if validation["selection_score"] < best_score:
                best_score, best_epoch = validation["selection_score"], epoch
                torch.save(checkpoint, out / "best_candidate.pt")
        history.append(item)
        write_json(out / "progress.json", dict(status="training", history=history, best_epoch=best_epoch,
                                               baseline_score=baseline["selection_score"], best_score=best_score))
        print(json.dumps(item), flush=True)
    write_json(out / "progress.json", dict(status="completed", history=history, best_epoch=best_epoch,
                                           baseline_score=baseline["selection_score"], best_score=best_score,
                                           auto_backend_approved=False, accuracy_verified=False))
    print(f"completed: best_epoch={best_epoch}, score={best_score:.6f}; not automatically deployed", flush=True)


if __name__ == "__main__":
    main()
