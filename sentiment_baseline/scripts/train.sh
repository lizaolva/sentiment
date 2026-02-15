#!/usr/bin/env bash
set -e

python src/train.py \
  --config config.yaml \

# добавить скрипт для теста, где будут браться готовые чекпоинты
