from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from model_and_tokenizer import tokenizer
import runtime_config
cfg = runtime_config.CFG
dataset_name = cfg['data']['dataset_name']
max_length = cfg['training']['max_length']
batch_size = cfg['training']['batch_size']

ds = load_dataset(dataset_name)

def slice_token(index, sentences, labels, tokenizer, max_length):
    start, stop, step = index.indices(len(sentences))
    result = []
    for i in range(start, stop, step):
        encoding = tokenizer(
                sentences[i],
                padding='max_length',
                truncation = True,
                max_length = max_length,
                return_tensors = 'pt'
            )
        encoding['labels'] = [labels[i]]
        result.append({key : value[0] for key, value in encoding.items()})

    return result

class SemDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_length):
        self.sentences = sentences
        self.labels = labels
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
                tokens,
                padding='max_length',
                truncation = True,
                max_length = self.max_length,
                return_tensors = 'pt'
            )
            encoding['labels'] = [tag]
            return {key : value[0] for key, value in encoding.items()}
        
dataset_train = SemDataset(ds['train']['text'], ds['train']['label'], tokenizer, max_length)
dataset_test = SemDataset(ds['test']['text'], ds['test']['label'], tokenizer, max_length)
dataset_val = SemDataset(ds['validation']['text'], ds['validation']['label'], tokenizer, max_length)

test_loader = DataLoader(dataset_test, batch_size, pin_memory=True)
train_loader = DataLoader(dataset_train, batch_size)
val_loader = DataLoader(dataset_val, batch_size)