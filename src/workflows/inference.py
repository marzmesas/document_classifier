import torch
from transformers import AutoTokenizer
import torch.nn.functional as F
from typing import Dict, Tuple, Any

# Global model and tokenizer instances
_model = None
_tokenizer = None
_device = None

def initialize_model(model_path: str, tokenizer_name: str = "roberta-base", device: str = None) -> None:
    """Initialize the model and tokenizer."""
    global _model, _tokenizer, _device
    
    # Set device
    _device = device if device else torch.device(
        "cuda" if torch.cuda.is_available() else 
        "mps" if torch.backends.mps.is_available() else 
        "cpu"
    )
    
    # Load tokenizer and model
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    _model = torch.jit.load(model_path)
    _model.to(_device)
    _model.eval()

def preprocess(text: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Preprocess the input text."""
    if _tokenizer is None:
        raise RuntimeError("Model not initialized. Call initialize_model first.")
        
    inputs = _tokenizer(
        text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    return inputs['input_ids'].to(_device), inputs['attention_mask'].to(_device)

@torch.no_grad()
def predict(text: str) -> Dict[str, Any]:
    """Generate prediction for the input text."""
    if _model is None:
        raise RuntimeError("Model not initialized. Call initialize_model first.")
    
    # Preprocess input
    input_ids, attention_mask = preprocess(text)
    
    # Get model predictions
    outputs = _model(input_ids, attention_mask)
    
    # Get probabilities
    probs = F.softmax(outputs, dim=1)
    
    # Get predicted class (add 1 to shift to original label ranges (1-8))
    predicted_class = torch.argmax(probs, dim=1).item() + 1
    confidence = probs[0][predicted_class - 1].item()
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': probs[0].tolist()
    }

# Example usage can be kept for testing
def main():
    model_path = "src/models/final_model/roberta_mlp_best_model_torchscript.pt"
    
    # Initialize model
    initialize_model(model_path)
    
    # Example text
    text = "esquire radio and electronics inc ee th qtr shr profit cts vs profit four cts annual div cts vs"
    
    # Get prediction
    result = predict(text)
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Class probabilities: {result['probabilities']}")

if __name__ == "__main__":
    main()
