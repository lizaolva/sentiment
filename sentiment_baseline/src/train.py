from transformers import Trainer
import os, json, argparse
from pathlib import Path
import runtime_config
from config import load_config

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()
runtime_config.CFG = load_config(args.config)
cfg = runtime_config.CFG
output_dir = cfg['dirs']['output']
model_dir = cfg['dirs']['model']
output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
model_dir = Path(model_dir)
model_dir.mkdir(parents=True, exist_ok=True)

from model_and_tokenizer import model, tokenizer
from training_args import training_args
from dataset import val_loader, train_loader
from metrics import compute_metrics
from optimizer_sheduler import optimizer, scheduler

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_loader.dataset,
    eval_dataset=val_loader.dataset,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, scheduler)
    )

train_metrics = trainer.train().metrics

with open(os.path.join(output_dir, "train_metrics.json"), "w") as f:
    json.dump(train_metrics, f, indent=2)

eval_results = trainer.evaluate()
print(f"Evaluation Results: {eval_results}")

with open(os.path.join(output_dir, "val_metrics.json"), "w") as f:
    json.dump(eval_results, f, indent=2)

trainer.save_model(model_dir)
tokenizer.save_pretrained(model_dir)