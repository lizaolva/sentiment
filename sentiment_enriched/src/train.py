from dataloaders import create_loader
from dataset import SemDataset
from metrics import compute_metrics
from model import SentimentClassifier
from optimizer_sheduler import optim_sheduler
from parse_enriched_dataset import classes_slots_dicts, parse_dataset, extract_classes_and_slots
from training_args import train_args
from transformers import BertTokenizerFast, Trainer
from datasets import load_dataset
import torch, os, json, random, yaml, argparse
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--datasets_path", required=True)
parser.add_argument("--model_dir", required=True)
parser.add_argument("--outputs_dir", required=True)
args = parser.parse_args()
with open(args.config) as f:
    cfg = yaml.safe_load(f)
batch_size = cfg['training']['batch_size']
max_length = cfg['training']['max_length']
epochs = cfg['training']['epochs']
lstm_hidden_size = cfg['training']['lstm_hidden_size']
semantic_emb_dim = cfg['training']['semantic_emb_dim']
learning_rate = cfg['training']['learning_rate']
weight_decay = cfg['training']['weight_decay']
dataset_name = cfg['data']['dataset_name']
model_name = cfg['model']['model_name']
output_dir = Path(args.outputs_dir)
output_dir.mkdir(parents=True, exist_ok=True)
model_dir = Path(args.model_dir)
model_dir.mkdir(parents=True, exist_ok=True)
datasets_dir = Path(args.datasets_path)

def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
set_random_seed(224)

ds = load_dataset(dataset_name)
tokenizer = BertTokenizerFast.from_pretrained(model_name)

train_dataset_raw = os.path.join(datasets_dir, "sentiment_train_pred.conllu")
val_dataset_raw = os.path.join(datasets_dir, "sentiment_val_pred.conllu")
test_dataset_raw = os.path.join(datasets_dir, "sentiment_test_pred.conllu")
sem_labels_dict = classes_slots_dicts(train_dataset_raw, val_dataset_raw)

train_labels = ds['train']['label']
val_labels = ds['validation']['label']
test_labels = ds['test']['label']

train_loader = create_loader(train_dataset_raw, train_labels, tokenizer, sem_labels_dict['slot2idx'], sem_labels_dict['class2idx'], max_length=max_length, parse_dataset=parse_dataset, SemDataset=SemDataset, batch_size=batch_size)
val_loader = create_loader(val_dataset_raw, val_labels, tokenizer, sem_labels_dict['slot2idx'], sem_labels_dict['class2idx'], max_length=max_length, parse_dataset=parse_dataset, SemDataset=SemDataset, batch_size=batch_size)
test_loader = create_loader(test_dataset_raw, test_labels, tokenizer, sem_labels_dict['slot2idx'], sem_labels_dict['class2idx'], max_length=max_length, parse_dataset=parse_dataset, SemDataset=SemDataset, batch_size=batch_size)

num_labels = len(set(ds['train']['label']))
num_semantic_classes = len(sem_labels_dict['classes'])
num_semantic_slots = len(sem_labels_dict['slots'])

model = SentimentClassifier(num_labels=num_labels,
    num_semantic_classes=num_semantic_classes,
    num_semantic_slots=num_semantic_slots,
    semantic_emb_dim=semantic_emb_dim,
    lstm_hidden_size=lstm_hidden_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

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

with open(os.path.join(output_dir, "train_metrics.json"), "w") as f:
    json.dump(train_metrics, f, indent=2)

eval_results = trainer.evaluate()
print(f"Evaluation Results: {eval_results}")

with open(os.path.join(output_dir, "eval_metrics.json"), "w") as f:
    json.dump(eval_results, f, indent=2)

trainer.save_model(model_dir)
tokenizer.save_pretrained(model_dir)