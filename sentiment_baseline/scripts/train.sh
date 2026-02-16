#!/usr/bin/env bash
set -e

python3 src/train.py \
  --config configs.yaml \

# добавить скрипт для теста, где будут браться готовые чекпоинты
