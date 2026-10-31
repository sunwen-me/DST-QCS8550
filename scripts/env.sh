#!/usr/bin/env bash
# Source after setting RBT_SDK_ROOT to an SDK 0.8.0 prefix; no private host path.
if [[ -z "${RBT_SDK_ROOT:-}" || ! -f "$RBT_SDK_ROOT/env.sh" ]]; then
  printf 'Set RBT_SDK_ROOT to your RBT SDK 0.8.0 directory before sourcing scripts/env.sh.\n' >&2
  return 1 2>/dev/null || exit 1
fi
source "$RBT_SDK_ROOT/env.sh"
