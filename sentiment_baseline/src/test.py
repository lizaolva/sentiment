from model_and_tokenizer import model
from dataset import test_loader

import torch
import torch.nn as nn
import torch.nn.functional as F
import evaluate
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

def test(model, data_loader):
    model = model.eval()

    all_preds = torch.tensor([])
    all_trues = torch.tensor([])

    with torch.no_grad():
        for d in data_loader:
        input_ids = d["input_ids"]
        attention_mask = d["attention_mask"]
        targets = d["labels"]

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        preds = torch.argmax(outputs['logits'], axis=-1)
        all_preds = torch.cat((all_preds, preds), -1)
        all_trues = torch.cat((all_trues, targets), -1)

    precision, recall, f1, _ = precision_recall_fscore_support(all_trues, all_preds, average='macro')
    acc = accuracy_score(all_trues, all_preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
    
test(model, test_loader)
    
