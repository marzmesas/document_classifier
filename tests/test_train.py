import pytest
import torch
import torch.nn as nn
import os
from unittest.mock import patch, MagicMock
from src.models.train import train_model, save_best_model
from src.models.evaluate import evaluate_model

class MockDataLoader:
    def __init__(self, num_batches=2):
        self.num_batches = num_batches
        
    def __iter__(self):
        for i in range(self.num_batches):
            # Create a batch with predictable labels (0-7 for 8 classes)
            batch = {
                "input_ids": torch.randint(0, 1000, (8, 10)),
                "attention_mask": torch.ones(8, 10),
                "labels": torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])  # One sample for each class
            }
            yield batch
    
    def __len__(self):
        return self.num_batches

@pytest.fixture
def mock_model():
    class PredictableModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Add a parameter so the optimizer has something to work with
            self.dummy_param = nn.Parameter(torch.randn(1))
            
        def forward(self, input_ids, attention_mask):
            batch_size = input_ids.shape[0]
            # Return logits for 8 classes
            logits = torch.zeros(batch_size, 8)
            
            # Set high values for the classes we want to predict
            # For perfect accuracy, we'll make each sample predict its index
            for i in range(batch_size):
                if i < 8:  # Ensure we don't go out of bounds
                    logits[i, i] = 10.0  # Predict class i for sample i
            
            return logits
    
    return PredictableModel()

@pytest.fixture
def mock_optimizer(mock_model):
    return torch.optim.Adam(mock_model.parameters(), lr=0.001)

@pytest.fixture
def mock_criterion():
    return nn.CrossEntropyLoss()

@pytest.fixture
def mock_dataloader():
    return MockDataLoader()

def test_train_model_runs_without_error(mock_model, mock_dataloader, mock_optimizer, mock_criterion, tmp_path):
    # Create temporary directories for checkpoints and final model
    checkpoint_dir = os.path.join(tmp_path, "checkpoints")
    final_model_dir = os.path.join(tmp_path, "final_model")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(final_model_dir, exist_ok=True)
    
    # Run a minimal training loop (1 epoch)
    device = torch.device("cpu")
    
    # Mock the actual training loop to avoid running it
    with patch('torch.save') as mock_save:
        # Mock the loss.backward() call to avoid gradient issues
        with patch('torch.Tensor.backward', return_value=None):
            # Call train_model with our mock objects
            train_model(
                mock_model, 
                mock_dataloader,  # train_loader
                mock_dataloader,  # val_loader
                mock_optimizer, 
                mock_criterion, 
                device, 
                checkpoint_dir, 
                final_model_dir,
                epochs=1
            )
            
            # Check that torch.save was called at least once (for model checkpointing)
            assert mock_save.call_count >= 1

def test_save_best_model(mock_model, mock_dataloader, tmp_path):
    # Create a temporary directory for the final model
    final_dir = os.path.join(tmp_path, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    device = torch.device("cpu")
    
    # Call save_best_model
    save_best_model(mock_model, device, mock_dataloader, final_dir)
    
    # Check that the model files were created
    assert os.path.exists(os.path.join(final_dir, "roberta_mlp_best_model.pth"))
    assert os.path.exists(os.path.join(final_dir, "roberta_mlp_best_model_torchscript.pt"))

def test_model_evaluation(mock_model, mock_dataloader):
    """Test that model evaluation generates robustness report correctly."""
    device = torch.device("cpu")
    
    # Create a mock report to return
    mock_report_data = {
        "Accuracy": 1.0,
        "Precision": 1.0,
        "Recall": 1.0,
        "F1-Score": 1.0,
        "Class Distribution": {0: {"Actual Count": 1, "Actual %": 50.0, 
                                  "Predicted Count": 1, "Predicted %": 50.0},
                              1: {"Actual Count": 1, "Actual %": 50.0, 
                                  "Predicted Count": 1, "Predicted %": 50.0}}
    }
    
    # Mock the entire dataloader with a fresh MagicMock that we control
    my_mock_dataloader = MagicMock()
    
    # Create mock batch with required keys
    mock_batch = {
        "input_ids": torch.ones(2, 10, dtype=torch.long),
        "attention_mask": torch.ones(2, 10),
        "labels": torch.tensor([0, 1])  # Match expected shape
    }
    my_mock_dataloader.__iter__.return_value = [mock_batch]
    
    # Set up all the mocks we need
    with patch('src.models.evaluate.generate_robustness_report') as mock_report:
        mock_report.return_value = mock_report_data
        
        with patch('src.models.evaluate.tqdm', return_value=my_mock_dataloader):
            # Call evaluate_model
            result = evaluate_model(mock_model, my_mock_dataloader, device)
            
            # Check if model was put in evaluation mode
            assert mock_model.training is False, "Model was not put in eval mode"
            
            # Verify report generation was called
            mock_report.assert_called_once()
            
            # Check the returned result matches our mock data
            assert result == mock_report_data
            assert result["Accuracy"] == 1.0 