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
model_dir = cfg['dirs']['model']
model_dir = Path(model_dir)
model_dir.mkdir(parents=True, exist_ok=True)
checkpoints_dir = os.path.join(model_dir, 'checkpoints')

training_args = TrainingArguments(
    output_dir=checkpoints_dir,
    eval_strategy="epoch",
    learning_rate = learning_rate,
    weight_decay = weight_decay,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    logging_steps=10,
    save_strategy="epoch", # возможно сократить
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    gradient_checkpointing=True # что и зачем
)