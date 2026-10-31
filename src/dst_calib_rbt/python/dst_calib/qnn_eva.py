"""Direct persistent QNN HTP adapter; never substitutes a CPU evaluator."""
from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path
import time

import numpy as np


class QnnEvaluator:
    _dst_strict_backend = True

    def __init__(self, context, runtime, library, manifest, profile=False):
        context, runtime, library = Path(context), Path(runtime), Path(library)
        info = json.loads(Path(manifest).read_text())
        expected = info["qnn"]["context_sha256"]
        if hashlib.sha256(context.read_bytes()).hexdigest() != expected:
            raise ValueError("QNN context SHA256 does not match its export manifest")
        if info["input"]["shape"] != [1, 3, 256, 512]:
            raise ValueError("QNN evaluator requires NCHW [1,3,256,512]")
        self._dst_eva_meta = {"arch": "sb", "virtual_camera": info["virtual_camera"]}
        self._handle = None
        self.times_ms = []
        self.lib = ctypes.CDLL(str(library.resolve()))
        ptr = ctypes.POINTER(ctypes.c_float)
        self.lib.dst_qnn_error.restype = ctypes.c_char_p
        self.lib.dst_qnn_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        self.lib.dst_qnn_create.restype = ctypes.c_void_p
        self.lib.dst_qnn_layout.argtypes = [ctypes.c_void_p]
        self.lib.dst_qnn_layout.restype = ctypes.c_int
        self.lib.dst_qnn_infer.argtypes = [ctypes.c_void_p, ptr, ctypes.c_size_t, ptr]
        self.lib.dst_qnn_infer.restype = ctypes.c_int
        self.lib.dst_qnn_profile.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.dst_qnn_profile.restype = ctypes.c_int
        self.lib.dst_qnn_destroy.argtypes = [ctypes.c_void_p]
        self.lib.dst_qnn_destroy.restype = None
        start = time.perf_counter()
        self._handle = self.lib.dst_qnn_create(str(runtime.resolve()).encode(), str(context.resolve()).encode(), int(profile))
        self.init_ms = (time.perf_counter() - start) * 1000
        if not self._handle:
            raise RuntimeError(self.lib.dst_qnn_error().decode())
        self.layout = "NHWC" if self.lib.dst_qnn_layout(self._handle) else "NCHW"

    def infer(self, value):
        if not self._handle:
            raise RuntimeError("QNN evaluator is closed")
        value = np.asarray(value)
        if value.shape != (1, 3, 256, 512):
            raise ValueError("QNN evaluator input must be [1,3,256,512]")
        value = np.ascontiguousarray(value, dtype=np.float32)
        if not np.isfinite(value).all():
            raise ValueError("QNN evaluator input contains nonfinite values")
        result = np.empty((1, 6), dtype=np.float32)
        ptr = ctypes.POINTER(ctypes.c_float)
        start = time.perf_counter()
        status = self.lib.dst_qnn_infer(self._handle, value.ctypes.data_as(ptr), value.size, result.ctypes.data_as(ptr))
        self.times_ms.append((time.perf_counter() - start) * 1000)
        if status:
            raise RuntimeError(self.lib.dst_qnn_error().decode())
        if not np.isfinite(result).all():
            raise RuntimeError("QNN evaluator returned nonfinite output")
        return result

    def eval(self):
        return self

    def __call__(self, tensor):
        import torch
        return torch.from_numpy(self.infer(tensor.detach().cpu().numpy()))

    def write_profile(self, path):
        if self.lib.dst_qnn_profile(self._handle, str(Path(path).resolve()).encode()):
            raise RuntimeError(self.lib.dst_qnn_error().decode())

    def close(self):
        if self._handle:
            self.lib.dst_qnn_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
