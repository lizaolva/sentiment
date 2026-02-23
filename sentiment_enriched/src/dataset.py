from torch.utils.data import Dataset
import torch

class SemDataset(Dataset):
    def __init__(self, texts, slots, classes, labels, tokenizer, slot2id, class2id, max_length):
        self.texts = texts
        self.slots = slots
        self.classes = classes
        self.labels = labels
        self.tokenizer = tokenizer
        self.slot2id = slot2id
        self.class2id = class2id
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.slice_token(idx)
        elif isinstance(idx, int):
            return self.get_instance(idx)

    def align_tokens_and_labels(self, tokens, slots, classes):
        word_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        aligned_slots = []
        aligned_classes = []

        current_word_idx = 0
        for word in word_ids:
            current_word = tokens[current_word_idx]
            original_word_token_count = len(self.tokenizer.tokenize(current_word))

            for subtoken in range(original_word_token_count):
                aligned_slots.append(slots[current_word_idx])
                aligned_classes.append(classes[current_word_idx])
            current_word_idx += 1
        return aligned_slots, aligned_classes
    
    def get_instance(self, index):
        tokens = self.texts[index]
        classes = self.classes[index]
        slots = self.slots[index]
        label = self.labels[index]

        encoding = self.tokenizer(
                    tokens,
                    is_split_into_words=True,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )

        slots, classes = self.align_tokens_and_labels(tokens, slots, classes)

        pad_len = self.max_length - len(slots)
        if pad_len > 0:
            for i in range(pad_len):
                slots.append('PAD')
                classes.append('PAD')
        else:
            slots = slots[:self.max_length]
            classes = classes[:self.max_length]

        slots = [self.slot2id[slot] for slot in slots]
        classes = [self.class2id[cls] for cls in classes]

        encoding["slots"] = slots
        encoding["classes"] = classes
        encoding["label"] = label
        encoding["input_ids"] = torch.squeeze(encoding["input_ids"], 0)
        encoding["token_type_ids"] = torch.squeeze(encoding["token_type_ids"], 0)
        encoding["attention_mask"] = torch.squeeze(encoding["attention_mask"], 0)

        return {key: torch.tensor(val) for key, val in encoding.items()}

    def slice_token(self, index):
        start, stop, step = index.indices(len(self.texts))
        result = []
        for index in range(start, stop, step):
            item = self.get_instance(self, index)
            result.append({key: torch.tensor(val) for key, val in item.items()})
        return result