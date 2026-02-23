set -e

python3 src/train.py \
  --config configs.yaml \
  --datasets_path datasets/ \
  --model_dir model/ \
  --checkpoints_dir model/checkpoints/ \
  --outputs_dir outputs/ \