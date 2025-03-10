import torch
import os
from tqdm import tqdm
from src.models.utils import generate_robustness_report

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    """
    Save model checkpoint including state dictionaries and metadata.
    
    Args:
        model (torch.nn.Module): The model to save
        optimizer (torch.optim.Optimizer): The optimizer to save
        epoch (int): Current epoch number
        loss (float): Current validation loss
        checkpoint_dir (str): Directory path where checkpoint will be saved
        
    Returns:
        None: Checkpoint is saved to disk
    """
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
      print(f"Created directory: {checkpoint_dir}")
    
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def save_best_model(model, device, train_loader, final_dir):
    """
    Save best model in TorchScript and regular PyTorch formats for deployment.
    
    This function saves the model in both native PyTorch format and optimized 
    TorchScript format for faster inference in production.
    
    Args:
        model (torch.nn.Module): The trained model to save
        device (torch.device): The device to use for tracing (CPU or CUDA)
        train_loader (DataLoader): DataLoader to get example inputs for tracing
        final_dir (str): Directory path where the model files will be saved
        
    Returns:
        None: Model files are saved to disk
    """
    if not os.path.exists(final_dir):
        os.makedirs(final_dir)
        print(f"Created directory: {final_dir}")
    
    best_model_path = os.path.join(final_dir, "roberta_mlp_best_model.pth")
    torch.save(model.state_dict(), best_model_path)

    # Get example inputs for tracing
    real_sample = next(iter(train_loader))
    real_input_ids = real_sample["input_ids"][0].unsqueeze(0).to(device)
    real_attention_mask = real_sample["attention_mask"][0].unsqueeze(0).to(device)

    # Convert to TorchScript using trace
    scripted_model_path = os.path.join(final_dir, "roberta_mlp_best_model_torchscript.pt")
    model.eval()  # Set to eval mode before tracing
    traced_model = torch.jit.trace(model, (real_input_ids, real_attention_mask))
    traced_model.save(scripted_model_path)

def train_model(model, train_loader, val_loader, optimizer, criterion, device, 
                checkpoint_dir, final_dir, epochs=3, mlflow_run=None):
    """
    Train and validate the model, saving checkpoints and the best model.
    
    This function:
    - Trains the model for the specified number of epochs
    - Evaluates performance on validation data after each epoch
    - Generates robustness reports with accuracy, precision, recall and F1-score
    - Logs metrics to MLflow if a run is provided
    - Saves checkpoints and the best model based on validation loss
    
    Args:
        model (torch.nn.Module): Model to train
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        optimizer (torch.optim.Optimizer): Optimizer for training
        criterion (torch.nn.Module): Loss function
        device (torch.device): Device to use for training (CPU or CUDA)
        checkpoint_dir (str): Directory to save checkpoints
        final_dir (str): Directory to save the best model
        epochs (int, optional): Number of training epochs. Defaults to 3.
        mlflow_run (mlflow.ActiveRun, optional): MLflow run for logging. Defaults to None.
        
    Returns:
        None: Model is trained and saved to disk
    """
    model.train()
    best_val_loss = float("inf")

    for epoch in range(epochs):
        total_train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

        # Training phase
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            loop.set_postfix(train_loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        total_val_loss = 0
        predictions, actuals = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                _, preds = torch.max(outputs.cpu(), 1)
                predictions.extend(preds.numpy())
                actuals.extend(labels.cpu().numpy().flatten())

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"\nEpoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}\n")

        print("Generating robustness report for epoch:  ", epoch+1)
        # Generate robustness report with class distribution
        robustness_report = generate_robustness_report(actuals, predictions)
        
        # Log metrics to MLflow if available
        if mlflow_run:
            import mlflow
            
            # Log basic training metrics
            mlflow.log_metrics({
                f"train_loss_epoch_{epoch+1}": avg_train_loss,
                f"val_loss_epoch_{epoch+1}": avg_val_loss,
                f"epoch_{epoch+1}_accuracy": robustness_report["Accuracy"],
                f"epoch_{epoch+1}_precision": robustness_report["Precision"],
                f"epoch_{epoch+1}_recall": robustness_report["Recall"],
                f"epoch_{epoch+1}_f1": robustness_report["F1-Score"]
            }, step=epoch+1)
            
            # Log class distribution as a separate metric for each class
            for class_id, stats in robustness_report["Class Distribution"].items():
                mlflow.log_metrics({
                    f"epoch_{epoch+1}_class_{class_id}_actual_count": stats["Actual Count"],
                    f"epoch_{epoch+1}_class_{class_id}_predicted_count": stats["Predicted Count"],
                }, step=epoch+1)

        # Save checkpoint after every epoch
        save_checkpoint(model, optimizer, epoch, avg_val_loss, checkpoint_dir)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_best_model(model, device, train_loader, final_dir)
            
            # Log best model metrics to MLflow
            if mlflow_run:
                mlflow.log_metrics({
                    "best_val_loss": best_val_loss,
                    "best_model_epoch": epoch + 1
                })
