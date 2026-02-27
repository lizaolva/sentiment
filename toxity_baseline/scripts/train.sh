#!/usr/bin/env bash
set -e

python3 src/train.py \
  --config configs.yaml \
  --model_dir model/ \
  --outputs_dir outputs/ \