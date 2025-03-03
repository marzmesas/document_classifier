import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from src.models.utils import generate_robustness_report

def evaluate_model(model, test_loader, device):
    """
    Evaluate model performance on the test dataset.
    
    This function runs inference on the test dataset and computes performance metrics
    using the generate_robustness_report function, which provides detailed metrics
    including accuracy, precision, recall, F1-score, and class distribution analysis.
    
    Args:
        model (torch.nn.Module): The model to evaluate
        test_loader (DataLoader): DataLoader containing test dataset
        device (torch.device): Device to run evaluation on (CPU or CUDA)
        
    Returns:
        dict: A comprehensive robustness report with performance metrics and 
              class distribution statistics
    """
    model.eval()
    predictions, actuals = [], []
    loop = tqdm(test_loader, desc="Evaluating", leave=True)
    
    with torch.no_grad():
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs.cpu(), 1)
            predictions.extend(preds.numpy())
            actuals.extend(labels.cpu().numpy().flatten())

    # Generate robustness report with class distribution
    print("Generating robustness report for testing set...")
    report = generate_robustness_report(actuals, predictions)
    return report
    
