import torch
from transformers import AutoTokenizer
import torch.nn.functional as F

class InferencePipeline:
    def __init__(self, model_path, tokenizer_name="roberta-base", device=None):
        # Set device
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else 
            "mps" if torch.backends.mps.is_available() else 
            "cpu"
        )
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = torch.jit.load(model_path)
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, text):
        # Tokenize the input text
        inputs = self.tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return inputs['input_ids'].to(self.device), inputs['attention_mask'].to(self.device)

    @torch.no_grad()
    def predict(self, text):
        # Preprocess input
        input_ids, attention_mask = self.preprocess(text)
        
        # Get model predictions
        outputs = self.model(input_ids, attention_mask)
        
        # Get probabilities
        probs = F.softmax(outputs, dim=1)
        
        # Get predicted class
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probs[0].tolist()
        }

def main():
    # Example usage
    model_path = "src/models/final_model/roberta_mlp_best_model_torchscript.pt"
    
    # Initialize pipeline
    pipeline = InferencePipeline(model_path)
    
    # Example text
    text = "esquire radio and electronics inc ee th qtr shr profit cts vs profit four cts annual div cts vs"
    
    # Get prediction
    result = pipeline.predict(text)
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Class probabilities: {result['probabilities']}")

if __name__ == "__main__":
    main()
