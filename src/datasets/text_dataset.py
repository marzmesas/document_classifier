import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset

class TextDataset(Dataset):
    """
    PyTorch Dataset for text classification tasks.
    
    This dataset handles tokenization of text inputs and pairing them with
    corresponding labels for model training and evaluation.
    
    Args:
        texts (list): List of text documents to be tokenized
        labels (list): List of corresponding labels (integers)
        model_name (str): Name of the pre-trained transformer model to use for tokenization
    """
    
    def __init__(self, texts, labels, model_name):
        """
        Initialize the dataset with texts, labels, and tokenizer.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encodings = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        """
        Return the number of samples in the dataset.
        
        Returns:
            int: Size of the dataset
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieve a sample by index.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            dict: Dictionary containing input_ids, attention_mask, and label
        """
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item