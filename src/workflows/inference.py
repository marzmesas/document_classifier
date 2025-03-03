import torch
from transformers import AutoTokenizer
import torch.nn.functional as F
from typing import Dict, Tuple, Any
import logging

# Global model and tokenizer instances
_model = None
_tokenizer = None
_device = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def initialize_model(model_path: str, tokenizer_name: str = "roberta-base", device: str = None) -> None:
    """
    Initialize the model and tokenizer for inference.
    
    This function loads a pre-trained model from the specified path and initializes
    the tokenizer needed for text preprocessing. It automatically selects the best 
    available device (CUDA, MPS, or CPU) if none is specified.
    
    Args:
        model_path (str): Path to the saved TorchScript model file
        tokenizer_name (str, optional): Name or path of the pre-trained tokenizer.
                                       Defaults to "roberta-base".
        device (str, optional): Device to run the model on. If None, the best
                              available device is selected automatically.
                              
    Raises:
        RuntimeError: If the model file cannot be loaded
    """
    global _model, _tokenizer, _device
    
    # Set device
    _device = device if device else torch.device(
        "cuda" if torch.cuda.is_available() else 
        "mps" if torch.backends.mps.is_available() else 
        "cpu"
    )
    logging.info(f"The detected device is:{ _device}")
    # Load tokenizer and model
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    _model = torch.jit.load(model_path, map_location=_device)
    _model.to(_device)
    _model.eval()

def preprocess(text: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess text input for model inference.
    
    This function tokenizes the input text and converts it to the appropriate
    format required by the model, including padding and truncation.
    
    Args:
        text (str): Raw text to be preprocessed
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - input_ids: Token IDs from the tokenizer
            - attention_mask: Attention mask indicating valid tokens
            
    Raises:
        RuntimeError: If the model is not initialized
    """
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
    """
    Generate classification prediction for the input text.
    
    This function takes raw text input, preprocesses it, and runs inference
    using the initialized model to produce classification results.
    
    Args:
        text (str): The text to classify
        
    Returns:
        Dict[str, Any]: A dictionary containing:
            - predicted_class (int): The predicted class ID (1-8)
            - confidence (float): Confidence score for the prediction (0-1)
            - probabilities (List[float]): Probability scores for all classes
            
    Raises:
        RuntimeError: If the model is not initialized
    """
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

def main():
    """
    Example usage of the inference module for testing purposes.
    
    This function demonstrates how to initialize the model, perform prediction
    on a sample text, and display the results.
    """
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
