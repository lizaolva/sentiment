from dataset import preprosessing, slice_token, ToxicDataset
from metrics import compute_metrics
from optimizer_sheduler import optim_sheduler
from training_args import train_args

import numpy as np
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer
from torch.utils.data import DataLoader
import torch, os, json, random, yaml, argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--model_dir", required=True)
parser.add_argument("--outputs_dir", required=True)
args = parser.parse_args()
with open(args.config) as f:
    cfg = yaml.safe_load(f)
batch_size = cfg['training']['batch_size']
max_length = cfg['training']['max_length']
epochs = cfg['training']['epochs']
learning_rate = cfg['training']['learning_rate']
weight_decay = cfg['training']['weight_decay']
model_name = cfg['model']['model_name']
output_dir = Path(args.outputs_dir)
output_dir.mkdir(parents=True, exist_ok=True)
model_dir = Path(args.model_dir)
model_dir.mkdir(parents=True, exist_ok=True)

def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
set_random_seed(224)

tokenizer = BertTokenizerFast.from_pretrained('DeepPavlov/rubert-base-cased')

ds_train, ds_val, ds_test = preprosessing()
dataset_train = ToxicDataset(ds_train['comment'], ds_train['toxic'], tokenizer, max_length)
dataset_test = ToxicDataset(ds_test['comment'], ds_test['toxic'], tokenizer, max_length)
dataset_val = ToxicDataset(ds_val['comment'], ds_val['toxic'], tokenizer, max_length)

test_loader = DataLoader(dataset_test, batch_size)
train_loader = DataLoader(dataset_train, batch_size)
val_loader = DataLoader(dataset_val, batch_size)

num_labels = len(set(ds_train['toxic']))
model = BertForSequenceClassification.from_pretrained('DeepPavlov/rubert-base-cased', num_labels=num_labels)

optimizer, scheduler = optim_sheduler(model, train_loader, epochs, batch_size)
training_args = train_args(learning_rate, weight_decay, batch_size, epochs)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_loader.dataset,
    eval_dataset=val_loader.dataset,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, scheduler)
    )

train_metrics = trainer.train().metrics

with open(os.path.join(outputs_dir, "train_metrics.json"), "w") as f:
    json.dump(train_metrics, f, indent=2)

eval_results = trainer.evaluate()
print(f"Evaluation Results: {eval_results}")

with open(os.path.join(outputs_dir, "eval_metrics.json"), "w") as f:
    json.dump(eval_results, f, indent=2)

trainer.save_model("../model_tokenizer")
tokenizer.save_pretrained("../model_tokenizer")