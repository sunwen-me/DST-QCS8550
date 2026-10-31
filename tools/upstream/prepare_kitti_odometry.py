#!/usr/bin/env python3
"""Prepare real KITTI odometry stereo/LiDAR frames for the existing trainer.

Read selected ZIP members from KITTI's public S3 objects with HTTP Range; never
download the complete 108 GB archives. Camera depth is computed from image_0/1
stereo ONLY, independently of Velodyne, to avoid a circular calibration target.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import struct
import time
import urllib.request
import zipfile
import zlib

import cv2
import numpy as np
import yaml

BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti/"
SOURCE = "https://www.cvlibs.net/datasets/kitti/eval_odometry.php"


def http_read(url: str, start: int | None = None, end: int | None = None) -> bytes:
    for attempt in range(6):
        try:
            headers = {} if start is None else {"Range": f"bytes={start}-{end}"}
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                        timeout=75) as r:
                if start is not None and r.status != 206:
                    raise OSError(f"Server ignored Range: {r.status}")
                data = r.read()
                if start is not None and len(data) != end - start + 1:
                    raise OSError("Incomplete HTTP Range response")
                return data
        except Exception:
            if attempt == 5:
                raise
            time.sleep(min(2 ** attempt, 16))
    raise AssertionError("unreachable")


class RemoteZip:
    """Seekable ZIP index with per-member, CRC-checked ranged downloads."""
    def __init__(self, url: str, cache: Path):
        self.url = url
        cache.mkdir(parents=True, exist_ok=True)
        self.cache = cache
        key = Path(url).name
        index = cache / (key + ".index.json")
        if index.exists():
            meta = json.loads(index.read_text())
            self.size = meta["size"]
            self.members = meta["members"]
            return
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"),
                                    timeout=40) as r:
            self.size = int(r.headers["Content-Length"])
        tail_start = max(0, self.size - 65557)
        tail = http_read(url, tail_start, self.size - 1)
        pos = tail.rfind(b"PK\x05\x06")
        if pos < 0:
            raise ValueError("Missing ZIP end record")
        end = struct.unpack_from("<4s4H2LH", tail, pos)
        cd_size, cd_offset = end[5:7]
        if cd_offset == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
            loc_pos = tail.rfind(b"PK\x06\x07", 0, pos)
            if loc_pos < 0:
                raise ValueError("Missing ZIP64 locator")
            zip64_offset = struct.unpack_from("<4sLQL", tail, loc_pos)[2]
            end64 = http_read(url, zip64_offset, zip64_offset + 55)
            fields = struct.unpack("<4sQ2H2L4Q", end64)
            cd_size, cd_offset = fields[-2:]
        central = http_read(url, cd_offset, cd_offset + cd_size - 1)
        self.members = {}
        offset = 0
        while offset < len(central):
            fields = struct.unpack_from("<4s6H3L5H2L", central, offset)
            if fields[0] != b"PK\x01\x02":
                raise ValueError(f"Invalid central directory at {offset}")
            name_len, extra_len, comment_len = fields[10:13]
            name = central[offset + 46:offset + 46 + name_len].decode("utf-8")
            extra = central[offset + 46 + name_len:offset + 46 + name_len + extra_len]
            usize, csize, local_offset = fields[9], fields[8], fields[16]
            ep = 0
            while ep + 4 <= len(extra):
                tag, length = struct.unpack_from("<HH", extra, ep)
                if tag == 1:
                    qp = ep + 4
                    vals = [usize, csize, local_offset]
                    for i in range(3):
                        if vals[i] == 0xFFFFFFFF:
                            vals[i] = struct.unpack_from("<Q", extra, qp)[0]
                            qp += 8
                    usize, csize, local_offset = vals
                ep += 4 + length
            self.members[name] = {"offset": local_offset, "compressed": csize,
                                  "size": usize, "method": fields[4],
                                  "crc": fields[7]}
            offset += 46 + name_len + extra_len + comment_len
        index.write_text(json.dumps({"url": url, "size": self.size,
                                     "members": self.members}))
        print(f"[index] {key}: {len(self.members)} entries", flush=True)

    def read(self, name: str) -> bytes:
        destination = self.cache / Path(self.url).stem / name
        entry = self.members[name]
        if destination.exists():
            data = destination.read_bytes()
            if len(data) == entry["size"] and zlib.crc32(data) == entry["crc"]:
                return data
        start = entry["offset"]
        # Local ZIP extra fields can differ from central extra fields. Read a
        # modest prefix with the compressed payload, fetch the remainder if needed.
        stop = min(self.size - 1, start + 30 + len(name.encode()) +
                   256 + entry["compressed"] - 1)
        raw = http_read(self.url, start, stop)
        header = struct.unpack_from("<4s5H3L2H", raw)
        if header[0] != b"PK\x03\x04":
            raise ValueError(f"Invalid local ZIP header: {name}")
        data_start = 30 + header[-2] + header[-1]
        need = data_start + entry["compressed"]
        if need > len(raw):
            raw += http_read(self.url, stop + 1, start + need - 1)
        payload = raw[data_start:need]
        if entry["method"] == zipfile.ZIP_DEFLATED:
            data = zlib.decompress(payload, -15)
        elif entry["method"] == zipfile.ZIP_STORED:
            data = payload
        else:
            raise ValueError(f"Unsupported ZIP compression {entry['method']}")
        if len(data) != entry["size"] or zlib.crc32(data) != entry["crc"]:
            raise ValueError(f"ZIP CRC/length mismatch: {name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_suffix(destination.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(destination)
        return data


def parse_calibration(text: str) -> tuple[np.ndarray, np.ndarray, float]:
    fields = {}
    for line in text.splitlines():
        if ":" in line:
            name, values = line.split(":", 1)
            fields[name] = np.fromstring(values, sep=" ").reshape(3, 4)
    P0, P1 = fields["P0"], fields["P1"]
    K = P0[:, :3].copy()
    # Odometry Tr maps Velodyne directly into rectified left grayscale cam0.
    T = np.eye(4)
    T[:3] = fields["Tr"]
    baseline = float(P0[0, 3] / P0[0, 0] - P1[0, 3] / P1[0, 0])
    if baseline <= 0 or not np.isfinite(T).all():
        raise ValueError("Invalid KITTI stereo calibration")
    if (not np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(T[:3, :3]), 1, atol=1e-4)):
        raise ValueError("Invalid KITTI extrinsic rotation")
    return K, T, baseline


def stereo_depth(left: np.ndarray, right: np.ndarray, K: np.ndarray,
                 baseline: float, depth_min: float = 1.0,
                 depth_max: float = 60.0) -> np.ndarray:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("Expected same-size rectified grayscale stereo images")
    matcher = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=192, blockSize=5, P1=8 * 25,
        P2=32 * 25, disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=100, speckleRange=2, preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disparity = matcher.compute(left, right).astype(np.float32) / 16.0
    depth = np.zeros_like(disparity)
    valid = disparity > 0
    depth[valid] = K[0, 0] * baseline / disparity[valid]
    depth[(depth < depth_min) | (depth > depth_max)] = 0
    return depth


def prepare(args) -> Path:
    cv2.setNumThreads(1)
    out = Path(args.out).resolve()
    cache = Path(args.cache_dir).resolve() if getattr(args, "cache_dir", None) else out / "downloads"
    cache.mkdir(parents=True, exist_ok=True)
    calib_path = cache / "data_odometry_calib.zip"
    if not calib_path.exists():
        calib_path.write_bytes(http_read(BASE + calib_path.name))
    with zipfile.ZipFile(calib_path) as z:
        calibration = {seq: parse_calibration(z.read(
            f"dataset/sequences/{seq}/calib.txt").decode())
                       for seq in args.train_sequences + args.val_sequences + args.test_sequences}
    gray = RemoteZip(BASE + "data_odometry_gray.zip", cache)
    lidar = RemoteZip(BASE + "data_odometry_velodyne.zip", cache)
    jobs = []
    manifest = {"source": SOURCE, "archives": [gray.url, lidar.url, BASE + calib_path.name],
                "calib_sha256": hashlib.sha256(calib_path.read_bytes()).hexdigest(),
                "depth_source": "rectified image_0/image_1 stereo SGBM; no LiDAR depth",
                "camera": "rectified grayscale camera 0",
                "depth_range_m": [1.0, 60.0], "splits": {},
                "stereo_sgbm": {"numDisparities": 192, "blockSize": 5,
                                "uniquenessRatio": 10, "speckleWindowSize": 100,
                                "speckleRange": 2, "mode": "SGBM_3WAY"},
                "created_local_date": time.strftime("%Y-%m-%d"),
                "frame_sampling": "uniform base; nested maximin time-index spacing", "frames": []}
    include_root = None
    included = {}
    if getattr(args, "include_manifest", None):
        source_manifest = Path(args.include_manifest).resolve()
        include_root = source_manifest.parent
        source_data = json.loads(source_manifest.read_text())
        for record in source_data["frames"]:
            included.setdefault((record["split"], record["sequence"]), {})[record["frame"]] = record
        manifest["included_manifest"] = str(source_manifest)
        manifest["included_manifest_sha256"] = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    groups = [("train", args.train_sequences, args.train_frames),
              ("val", args.val_sequences, args.eval_frames),
              ("test", args.test_sequences, args.eval_frames)]
    all_sequences = args.train_sequences + args.val_sequences + args.test_sequences
    if len(all_sequences) != len(set(all_sequences)):
        raise ValueError("Train/validation/test sequences must not overlap")
    for split, sequences, n_frames in groups:
        manifest["splits"][split] = sequences
        for seq in sequences:
            K, T, baseline = calibration[seq]
            session = out / "prepared" / split / seq
            session.mkdir(parents=True, exist_ok=True)
            (session / "extrinsic.yaml").write_text(yaml.safe_dump(
                {"T_cam_lidar": T.tolist(), "source": f"KITTI odometry sequence {seq} calib.txt Tr"}))
            prefix = f"dataset/sequences/{seq}/image_0/"
            ids = sorted(Path(n).stem for n in gray.members if n.startswith(prefix) and n.endswith(".png"))
            ids = [i for i in ids if f"dataset/sequences/{seq}/image_1/{i}.png" in gray.members
                   and f"dataset/sequences/{seq}/velodyne/{i}.bin" in lidar.members]
            if not ids:
                raise ValueError(f"No synchronized data in sequence {seq}")
            required = included.get((split, seq), {})
            selected = select_frame_ids(ids, n_frames, required)
            for index in selected:
                if index in required:
                    source = include_root / required[index]["path"]
                    target = session / f"frame_{index}.npz"
                    if source.is_file() and not target.exists():
                        os.link(source, target)
                jobs.append((split, seq, index, session, K, baseline))
    # Interleave sequences so an interrupted download still has broad coverage.
    sequence_ends = {}
    for job in jobs:
        sequence_ends[job[1]] = max(sequence_ends.get(job[1], 1), int(job[2]))
    jobs.sort(key=lambda j: (int(j[2]) / sequence_ends[j[1]], j[1]))
    expected_paths = {str((j[3] / f"frame_{j[2]}.npz").resolve()) for j in jobs}
    existing_paths = {str(p.resolve()) for p in (out / "prepared").glob("*/*/frame_*.npz")}
    if existing_paths - expected_paths:
        raise ValueError("Prepared directory contains frames outside this selection; "
                         "use a new --out directory when changing the split or sample count")
    print(f"[prepare] {len(jobs)} stereo/LiDAR frames, workers={args.workers}", flush=True)

    def one(job):
        split, seq, frame_id, session, K, baseline = job
        target = session / f"frame_{frame_id}.npz"
        if target.exists():
            with np.load(target, allow_pickle=False) as f:
                stats = {"valid_depth_pixels": int(np.count_nonzero(f["depth"])),
                         "lidar_points": len(f["points_lidar"])}
        else:
            prefix = f"dataset/sequences/{seq}"
            left = cv2.imdecode(np.frombuffer(gray.read(f"{prefix}/image_0/{frame_id}.png"),
                                              np.uint8), cv2.IMREAD_GRAYSCALE)
            right = cv2.imdecode(np.frombuffer(gray.read(f"{prefix}/image_1/{frame_id}.png"),
                                               np.uint8), cv2.IMREAD_GRAYSCALE)
            points = np.frombuffer(lidar.read(f"{prefix}/velodyne/{frame_id}.bin"),
                                   dtype="<f4").reshape(-1, 4)[:, :3].copy()
            points = points[np.isfinite(points).all(axis=1)]
            depth = stereo_depth(left, right, K, baseline)
            if np.count_nonzero(depth) < 1000 or len(points) < 1000:
                raise ValueError(f"Insufficient valid data: {seq}/{frame_id}")
            tmp = target.with_suffix(".npz.tmp")
            with tmp.open("wb") as f:
                np.savez_compressed(f, points_lidar=points, depth=depth, K=K,
                                    stamp=float(int(frame_id)) / 10.0)
            tmp.replace(target)
            stats = {"valid_depth_pixels": int(np.count_nonzero(depth)),
                     "lidar_points": len(points)}
        return {"split": split, "sequence": seq, "frame": frame_id,
                "path": str(target.relative_to(out)), **stats}

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            record = future.result()
            manifest["frames"].append(record)
            if i % 10 == 0 or i == 1 or i == len(jobs):
                print(f"[prepare {i}/{len(jobs)}] {record['sequence']}/{record['frame']} "
                      f"elapsed={time.monotonic() - started:.0f}s", flush=True)
    manifest["frames"].sort(key=lambda r: (r["split"], r["sequence"], r["frame"]))
    temp_manifest = out / "manifest.json.tmp"
    temp_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    temp_manifest.replace(out / "manifest.json")
    print(f"[ready] {out / 'manifest.json'}", flush=True)
    return out


def select_frame_ids(ids, count, required=()):
    """Retain source frames and add frames farthest from existing time indices.

    Uniformly sampling only the remaining list would select immediate neighbors
    of a uniform source subset. Maximin spacing instead fills its temporal gaps.
    """
    ids = sorted(ids)
    required = set(required)
    target = min(int(count), len(ids))
    if not required.issubset(ids) or len(required) > target:
        raise ValueError("Included manifest does not fit the requested selection")
    if not required:
        return [ids[i] for i in np.linspace(0, len(ids) - 1, target, dtype=int)]
    positions = np.arange(len(ids))
    selected = np.array([i for i, frame in enumerate(ids) if frame in required])
    distance = np.abs(positions[:, None] - selected[None, :]).min(axis=1)
    for _ in range(target - len(required)):
        index = int(np.argmax(distance))
        required.add(ids[index])
        distance = np.minimum(distance, np.abs(positions - index))
    return sorted(required)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/kitti_odometry")
    p.add_argument("--train-sequences", nargs="+", default=[f"{i:02d}" for i in range(9)])
    p.add_argument("--val-sequences", nargs="+", default=["09"])
    p.add_argument("--test-sequences", nargs="+", default=["10"])
    p.add_argument("--train-frames", type=int, default=64, help="Evenly sampled frames per training sequence")
    p.add_argument("--eval-frames", type=int, default=48)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cache-dir", default=None, help="Shared CRC-verified archive-member cache")
    p.add_argument("--include-manifest", default=None, help="Retain all selected frames from a smaller dataset")
    args = p.parse_args()
    if min(args.train_frames, args.eval_frames, args.workers) < 1:
        p.error("Frame counts and workers must be positive")
    prepare(args)


if __name__ == "__main__":
    main()
