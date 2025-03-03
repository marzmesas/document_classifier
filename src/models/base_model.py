import torch.nn as nn
from transformers import AutoModel

class TransformerMLP(nn.Module):
    """
    Transformer model with multi-layer perceptron head for text classification.
    
    This model consists of a pre-trained transformer encoder (e.g., RoBERTa)
    followed by a classification head implemented as a multi-layer perceptron.
    
    Args:
        model_name (str): Name or path of the pre-trained transformer model
        num_classes (int): Number of output classes for classification
    """
    
    def __init__(self, model_name, num_classes):
        """
        Initialize the model with a transformer encoder and MLP classifier.
        """
        super(TransformerMLP, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.fc = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.
        
        This method passes input through the transformer encoder, extracts the 
        CLS token embedding, and feeds it through the classification head.
        
        Args:
            input_ids (torch.Tensor): Token indices of input sequence(s)
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding tokens
            
        Returns:
            torch.Tensor: Logits for each class
        """
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = output.last_hidden_state[:, 0, :]
        return self.fc(cls_embedding)