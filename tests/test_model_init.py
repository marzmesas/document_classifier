import pytest
import torch
import os
from unittest.mock import patch, MagicMock
from src.workflows.inference import initialize_model

@pytest.fixture
def mock_torch_jit():
    mock_model = MagicMock()
    mock_model.eval.return_value = None
    return mock_model

def test_initialize_model_with_mock(mock_torch_jit, tmp_path):
    # Create a dummy model file
    model_path = os.path.join(tmp_path, "dummy_model.pt")
    with open(model_path, 'wb') as f:
        f.write(b'dummy model content')
    
    # Mock torch.jit.load to return our mock model
    with patch('torch.jit.load', return_value=mock_torch_jit):
        # Mock AutoTokenizer.from_pretrained
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
            # Initialize the model
            initialize_model(model_path)
            
            # Check that the model was loaded and set to eval mode
            mock_torch_jit.eval.assert_called_once()
            
            # Check that the tokenizer was initialized
            mock_tokenizer.assert_called_once() 