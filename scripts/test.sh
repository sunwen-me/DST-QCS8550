#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src/dst_calib_rbt/python:$PWD/src/dst_calib_rbt/tests${PYTHONPATH:+:$PYTHONPATH}"
# Explicitly omit the multi-frame long synthetic optimization suite.
"${PYTHON:-python3}" -m pytest src/dst_calib_rbt/tests --ignore=src/dst_calib_rbt/tests/test_synthetic.py "$@"
