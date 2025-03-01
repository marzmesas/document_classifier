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
    # Rename the test to avoid confusion with the actual evaluate_model function
    device = torch.device("cpu")
    
    # Patch the print function to capture the accuracy value
    printed_values = []
    def mock_print(*args, **kwargs):
        printed_values.append(args[0])
    
    # Test the evaluate_model function with our predictable model and dataloader
    with patch('builtins.print', side_effect=mock_print):
        with patch('src.models.evaluate.tqdm', return_value=mock_dataloader):
            # Call evaluate_model instead of evaluate_model
            result = evaluate_model(mock_model, mock_dataloader, device)
            
            # Check if the function printed the expected accuracy
            assert any("Test Accuracy: 1.0000" in str(val) for val in printed_values)
            
            # If evaluate_model returns None, we can still verify the accuracy from the printed output
            if result is None:
                # Extract the accuracy from the printed output
                accuracy_line = next((val for val in printed_values if "Test Accuracy:" in str(val)), None)
                if accuracy_line:
                    accuracy = float(str(accuracy_line).split(":")[1].strip())
                    assert accuracy == 1.0
            else:
                # If evaluate_model returns the accuracy, check it directly
                assert isinstance(result, float)
                assert result == 1.0
            
            # Check that the model was put in evaluation mode
            assert not mock_model.training 