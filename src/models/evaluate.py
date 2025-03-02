import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from src.models.utils import generate_robustness_report

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set and generate robustness report."""
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
    
