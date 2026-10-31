#!/usr/bin/env bash
# KITTI first-pass training: real stereo depth + sequence-held-out validation.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"
DATA="${KITTI_DATA_DIR:-$ROOT/runs/kitti_odometry}"
OUT="${KITTI_TRAIN_OUT:-$ROOT/runs/train_eva_kitti}"
EPOCHS="${EPOCHS:-100}"
SAMPLES_PER_FRAME="${SAMPLES_PER_FRAME:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
mkdir -p "$DATA" "$OUT"
export PYTHONPATH="$ROOT/src/dst_calib_rbt/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
if [[ ! -f "$DATA/manifest.json" ]]; then
  echo "Review KITTI terms, then explicitly prepare data; nothing downloaded automatically." >&2
  echo "$PY tools/upstream/prepare_kitti_odometry.py --out $DATA --train-frames 64 --eval-frames 48" >&2
  exit 2
fi
shopt -s nullglob
train_dirs=("$DATA"/prepared/train/*)
val_dirs=("$DATA"/prepared/val/*)
test_dirs=("$DATA"/prepared/test/*)
if (( ${#train_dirs[@]} == 0 || ${#val_dirs[@]} == 0 || ${#test_dirs[@]} == 0 )); then
  echo "Prepared training, validation and test sessions required." >&2; exit 2
fi
"$PY" -u -m dst_calib.train \
  --data_dir "${train_dirs[@]}" --val_data_dir "${val_dirs[@]}" \
  --config config/kitti_odometry.yaml --out "$OUT" --arch sb \
  --epochs "$EPOCHS" --samples_per_frame "$SAMPLES_PER_FRAME" \
  --val_samples_per_frame 2 --batch_size 8 --num_workers "$NUM_WORKERS" \
  --seed 0 --device "${DEVICE:-cuda}" --log_every 25
"$PY" -u tools/upstream/evaluate_kitti_odometry.py \
  --checkpoint "$OUT/best.pt" --data-dir "${test_dirs[@]}" \
  --samples-per-frame 8 --out "$OUT/test_metrics.json"
echo "[完成] 模型: $OUT/best.pt; 独立测试: $OUT/test_metrics.json"
