#!/usr/bin/env bash
# NavSim용 MapTR 테스트 및 시각화 스크립트
#
# Usage:
#   ./tools/maptr/test_navsim.sh CONFIG CHECKPOINT GPUS [--show-dir DIR] [--score-thr THR]
#
# Example:
#   ./tools/maptr/test_navsim.sh \
#       projects/configs/maptr/maptr_tiny_r50_navsim_24e.py \
#       work_dirs/maptr_tiny_r50_navsim_24e/epoch_2.pth \
#       1 \
#       --show-dir ./vis_navsim_pred \
#       --num-samples 20

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29505}

PYTHONPATH="$(dirname $0)/../..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test_navsim.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval chamfer
