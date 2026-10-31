#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build build --target dst_calib_rbt_node --parallel "${LIO_BUILD_JOBS:-2}"
mkdir -p bin
ln -sfn ../src/dst_calib_rbt/scripts/rbt-dst-calib bin/rbt-dst-calib
