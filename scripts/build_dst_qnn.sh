#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
: "${QNN_INCLUDE_DIR:?Set QNN_INCLUDE_DIR to your legal QAIRT include/QNN directory}"
test -f "$QNN_INCLUDE_DIR/QnnInterface.h"
mkdir -p build/dst-calib/lib
"${CXX:-g++}" -std=c++20 -O3 -fPIC -shared -Wall -Wextra -Wpedantic -Werror \
  -Isrc/qnn_runtime/include -isystem "$QNN_INCLUDE_DIR" $(pkg-config --cflags opencv4 | sed 's/-I/-isystem /g') \
  src/dst_calib_rbt/src/qnn_evaluator.cpp src/qnn_runtime/src/session.cpp \
  -lopencv_core -ldl -o build/dst-calib/lib/libdst_qnn.so
