import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def generate_robustness_report(actuals, predictions):
    """
    Generate a comprehensive robustness report for model evaluation.
    
    This function calculates key classification metrics and analyzes class distribution
    to help identify potential issues like class imbalance or biased predictions.
    
    Args:
        actuals (list): List of actual labels
        predictions (list): List of predicted labels
        
    Returns:
        dict: Report containing:
            - Accuracy: Overall classification accuracy
            - Precision: Weighted precision score
            - Recall: Weighted recall score
            - F1-Score: Weighted F1 score
            - Class Distribution: Detailed statistics for each class showing actual and
              predicted counts and percentages
    """
    # Calculate performance metrics
    accuracy = accuracy_score(actuals, predictions)
    precision = precision_score(actuals, predictions, average='weighted', zero_division=0)
    recall = recall_score(actuals, predictions, average='weighted', zero_division=0)
    f1 = f1_score(actuals, predictions, average='weighted', zero_division=0)

    # Compute class distributions
    actual_counts = Counter(actuals)
    pred_counts = Counter(predictions)
    total_samples = len(actuals)
    
    class_distribution = {
        cls: {
            "Actual Count": actual_counts.get(cls, 0),
            "Actual %": (actual_counts.get(cls, 0) / total_samples) * 100,
            "Predicted Count": pred_counts.get(cls, 0),
            "Predicted %": (pred_counts.get(cls, 0) / total_samples) * 100
        }
        for cls in set(actuals) | set(predictions)
    }
    
    # Print Report
    print("\n===== Robustness Report =====")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}\n")

    print("===== Class Distribution =====")
    print(f"{'Class':<10}{'Actual Count':<15}{'Actual %':<10}{'Predicted Count':<18}{'Predicted %':<10}")
    print("="*65)
    for cls, stats in sorted(class_distribution.items()):
        print(f"{cls:<10}{stats['Actual Count']:<15}{stats['Actual %']:.2f}%{' ' * 5}"
              f"{stats['Predicted Count']:<18}{stats['Predicted %']:.2f}%")

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Class Distribution": class_distribution
    }
