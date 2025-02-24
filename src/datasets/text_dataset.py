import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, texts, labels, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encodings = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item