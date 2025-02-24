import torch.nn as nn
from transformers import AutoModel

class TransformerMLP(nn.Module):
    def __init__(self, model_name, num_classes):
        super(TransformerMLP, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.fc = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = output.last_hidden_state[:, 0, :]
        return self.fc(cls_embedding)