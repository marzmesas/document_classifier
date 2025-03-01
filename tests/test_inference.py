import pytest
import torch
from src.workflows.inference import predict, initialize_model
import os

@pytest.mark.skipif(not os.path.exists("src/models/final_model/roberta_mlp_best_model_torchscript.pt"),
                    reason="Model file not found")
def test_prediction_output_structure():
    text = "sample text for testing"
    result = predict(text)
    
    assert isinstance(result, dict)
    assert "predicted_class" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert isinstance(result["predicted_class"], int)
    assert isinstance(result["confidence"], float)
    assert isinstance(result["probabilities"], list)
    assert 0 <= result["confidence"] <= 1
    assert len(result["probabilities"]) > 0

def test_model_initialization_error():
    with pytest.raises((RuntimeError, ValueError)):
        # Try to initialize with non-existent path
        initialize_model("nonexistent/path/model.pt") 