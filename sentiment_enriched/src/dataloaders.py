from torch.utils.data import DataLoader

def create_loader(path_to_dataset, labels, tokenizer, slots2id, classes2id, max_length, parse_dataset, SemDataset, batch_size):
    texts, slots, classes = parse_dataset(path_to_dataset)
    dataset = SemDataset(texts, slots, classes, labels, tokenizer, slots2id, classes2id, max_length)
    return DataLoader(dataset, batch_size, shuffle=False)