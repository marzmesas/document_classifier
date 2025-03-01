import pytest
import torch
from src.datasets.text_dataset import TextDataset

def test_text_dataset_creation():
    texts = ["sample text one", "sample text two"]
    labels = [0, 1]
    dataset = TextDataset(texts, labels, "roberta-base")
    
    assert len(dataset) == 2
    assert isinstance(dataset[0], dict)
    assert "input_ids" in dataset[0]
    assert "attention_mask" in dataset[0]
    assert "labels" in dataset[0]
    assert isinstance(dataset[0]["labels"], torch.Tensor)

def test_text_dataset_getitem():
    texts = ["sample text"]
    labels = [0]
    dataset = TextDataset(texts, labels, "roberta-base")
    
    item = dataset[0]
    assert isinstance(item["input_ids"], torch.Tensor)
    assert isinstance(item["attention_mask"], torch.Tensor)
    assert isinstance(item["labels"], torch.Tensor)
    assert item["labels"].item() == 0 