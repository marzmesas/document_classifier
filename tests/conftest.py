import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the project root directory to Python's path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

# Set up environment for testing
os.environ["TESTING"] = "true"

# Import pytest before any app imports
import pytest
from fastapi.testclient import TestClient

# Import app after setting environment
from src.app.main import app, create_app
from src.workflows.inference import initialize_model

# Set up mocks for OpenTelemetry
mock_tracer_provider = MagicMock()
mock_meter_provider = MagicMock()
mock_meter = MagicMock()
mock_counter = MagicMock()

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture(scope="session")
def mock_model():
    import torch
    import torch.nn as nn
    
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 8)
            
        def forward(self, input_ids, attention_mask):
            # Just return random outputs of the expected shape
            batch_size = input_ids.shape[0]
            return torch.randn(batch_size, 8)
    
    return MockModel()

@pytest.fixture(scope="session", autouse=True)
def setup_model():
    # Initialize model for testing
    model_path = "src/models/final_model/roberta_mlp_best_model_torchscript.pt"
    if os.path.exists(model_path):
        initialize_model(model_path) 