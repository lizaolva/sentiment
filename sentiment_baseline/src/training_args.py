from transformers import TrainingArguments
from pathlib import Path
import runtime_config

cfg = runtime_config.CFG
model_name = cfg['model']['model_name']
num_labels = cfg['model']['num_labels']
batch_size = cfg['training']['batch_size']
epochs = cfg['training']['epochs']
learning_rate = cfg['training']['learning_rate']
weight_decay = cfg['training']['weight_decay']
output_dir = cfg['dirs']['output']
output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    learning_rate = learning_rate,
    weight_decay = weight_decay,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    logging_dir="outputs/logs", # посмотреть
    logging_steps=10,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    gradient_checkpointing=True
)