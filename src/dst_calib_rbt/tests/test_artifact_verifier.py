import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('dst_artifact_verifier', ROOT / 'scripts/verify_artifacts.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def inventory(tmp_path, path, blob):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'artifacts': [{'path': path, 'bytes': len(blob), 'sha256': hashlib.sha256(blob).hexdigest()}]}))
    return manifest

def test_artifact_valid_and_corrupt(tmp_path):
    blob = b'weights'
    payload = tmp_path / 'model.pt'
    payload.write_bytes(blob)
    manifest = inventory(tmp_path, 'model.pt', blob)
    assert mod.verify(tmp_path, manifest) == []
    payload.write_bytes(b'Weights')
    assert 'SHA256 mismatch' in mod.verify(tmp_path, manifest)[0]

def test_lfs_pointer_rejected(tmp_path):
    blob = b'version https://git-lfs.github.com/spec/v1' + bytes([10])
    (tmp_path / 'model.pt').write_bytes(blob)
    manifest = inventory(tmp_path, 'model.pt', blob)
    assert 'git lfs pull' in mod.verify(tmp_path, manifest)[0]

def test_missing_and_unsafe_paths(tmp_path):
    manifest = inventory(tmp_path, 'missing.pt', b'x')
    assert 'Missing' in mod.verify(tmp_path, manifest)[0]
    manifest = inventory(tmp_path, '../model.pt', b'x')
    assert 'Unsafe' in mod.verify(tmp_path, manifest)[0]
