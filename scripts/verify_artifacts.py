#!/usr/bin/env python3
"""Verify model/example files and reject unhydrated Git LFS pointers."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]

def verify(root, manifest):
    failures = []
    data = json.loads(manifest.read_text())
    for item in data['artifacts']:
        relative = Path(item['path'])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root.resolve()):
            failures.append(f'Unsafe manifest path: {relative}')
            continue
        if not path.is_file():
            failures.append(f'Missing: {relative}')
            continue
        with path.open('rb') as handle:
            if handle.read(130).startswith(b'version https://git-lfs.github.com/spec/v1'):
                failures.append(f'Run git lfs pull (pointer): {relative}')
                continue
        if path.stat().st_size != item['bytes']:
            failures.append(f'Size mismatch: {relative}')
            continue
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != item['sha256']:
            failures.append(f'SHA256 mismatch: {relative}')
    if not data['artifacts']:
        failures.append('Empty artifact inventory')
    return failures

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    manifest = args.manifest or args.root / 'models/artifacts.sha256.json'
    try:
        failures = verify(args.root, manifest)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'Artifact verification failed: {exc}', file=sys.stderr)
        return 1
    if failures:
        print(chr(10).join(failures), file=sys.stderr)
        return 1
    count = len(json.loads(manifest.read_text())['artifacts'])
    print(f'OK: {count} payloads, SHA256/size match, no LFS pointers.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
