#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
sdk_prefix="${RBT_SDK_ROOT:?Set RBT_SDK_ROOT to your RBT SDK 0.8.0 prefix}"
source "${sdk_prefix}/env.sh"
export RBT_COMPOSER_ADDR="${RBT_COMPOSER_ADDR:-127.0.0.1:7447}"

mode="${DST_CALIB_MODE:-selfsup}"
backend="${DST_CALIB_BACKEND:-cpu}"
python_bin="${DST_CALIB_PYTHON:-python3}"
config="${DST_CALIB_CONFIG:-${workspace_root}/config/dst_calib_rbt/default.yaml}"
node="${DST_CALIB_NODE:-${workspace_root}/build/dst-calib/bin/dst_calib_rbt_node}"
composer_pid=""
cd -- "$workspace_root"
mkdir -p -- "${workspace_root}/runs"
log_dir="$(mktemp -d "${workspace_root}/runs/dst-calib.XXXXXX")"

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "$composer_pid" ]]; then
        kill -TERM "$composer_pid" 2>/dev/null || true
        wait "$composer_pid" 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

if ! python3 - "$RBT_COMPOSER_ADDR" <<'PY'
import socket
import sys
try:
    host, port = sys.argv[1].rsplit(":", 1)
    with socket.create_connection((host.strip("[]"), int(port)), timeout=0.5):
        pass
except (OSError, ValueError):
    raise SystemExit(1)
PY
then
    "${sdk_prefix}/bin/composer" >"${log_dir}/composer.log" 2>&1 &
    composer_pid=$!
    sleep 0.4
    kill -0 "$composer_pid" 2>/dev/null || { cat "${log_dir}/composer.log" >&2; exit 1; }
fi

echo "DST-RBT mode=${mode} backend=${backend} python=${python_bin} config=${config} logs=${log_dir}"
exec "$node" --config "$config" --mode "$mode" --backend "$backend" --python "$python_bin" "$@" \
    2>&1 | tee "${log_dir}/dst.log"
