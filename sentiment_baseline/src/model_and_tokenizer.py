from transformers import BertTokenizerFast, BertForSequenceClassification
import runtime_config

import torch
print("torch.cuda.is_available() = ", torch.cuda.is_available()) # Должно вернуть True
print("torch.__version__ =", torch.__version__)
device = torch.device("cuda")

cfg = runtime_config.CFG
model_name = cfg['model']['model_name']
num_labels = cfg['model']['num_labels']

tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels= num_labels)
model.to(device)