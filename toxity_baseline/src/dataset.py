import kagglehub, os, torch
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import Dataset

def preprosessing():
    path = kagglehub.dataset_download("blackmoon/russian-language-toxic-comments")
    path = os.path.join(path, "labeled.csv")
    data = pd.read_csv(path)
    ds_train, ds_test = train_test_split(data, test_size=0.1)
    ds_val, ds_test = train_test_split(ds_test, test_size=0.5)
    return ds_train, ds_val, ds_test

def slice_token(index, sentences, labels, tokenizer, max_length):
    start, stop, step = index.indices(len(sentences))
    result = []
    for i in range(start, stop, step):
        encoding = tokenizer(
                [sentences[i]],
                padding='max_length',
                truncation = True,
                max_length = max_length,
                return_tensors = 'pt'
            )
        item = {key: val.squeeze(0) for key, val in encoding.items()} 
        item['labels'] = torch.tensor(labels[i], dtype=torch.long) 
        result.append(item)
    return result

class ToxicDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_length):
        self.sentences = sentences.tolist()
        self.labels = labels.astype(float).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.sentences)
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return slice_token(idx, self.sentences, self.labels, self.tokenizer, self.max_length)
        elif isinstance(idx, int):
            tokens = self.sentences[idx]
            tag = self.labels[idx]
            
            encoding = self.tokenizer(
                [tokens],
                padding='max_length',
                truncation = True,
                max_length = self.max_length,
                return_tensors = 'pt'
            )
            item = {key: val.squeeze(0) for key, val in encoding.items()} 
            item['labels'] = torch.tensor(tag, dtype=torch.long)
            return item