import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score

def test_model(model, test_loader, device):
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

    acc = accuracy_score(actuals, predictions)
    print(f"Test Accuracy: {acc:.4f}")