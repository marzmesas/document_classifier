import pytest
import torch
from src.models.base_model import TransformerMLP

@pytest.fixture
def model():
    return TransformerMLP("roberta-base", num_classes=8)

def test_model_structure(model):
    assert hasattr(model, 'encoder')
    assert hasattr(model, 'fc')
    
def test_model_forward(model):
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))
    
    outputs = model(input_ids, attention_mask)
    assert outputs.shape == (batch_size, 8)  # 8 classes
    assert not torch.isnan(outputs).any()

def test_model_output_range(model):
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))
    
    outputs = model(input_ids, attention_mask)
    # Check if outputs are reasonable (not too extreme)
    assert outputs.abs().max() < 100 