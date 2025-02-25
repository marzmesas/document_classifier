import torch
from tqdm import tqdm
import os

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    """Save model checkpoint (state_dict only)."""
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

def save_best_model(model, device,train_loader,final_dir):
    """Save best model in TorchScript and ONNX formats for optimized inference."""
    if not os.path.exists(final_dir):
      os.makedirs(final_dir)
    print(f"Created directory: {final_dir}")
    
    best_model_path = os.path.join(final_dir, "roberta_mlp_best_model.pth")
    torch.save(model.state_dict(), best_model_path)

    real_sample = next(iter(train_loader))
    real_input_ids = real_sample["input_ids"][0].unsqueeze(0).to(device)
    real_attention_mask = real_sample["attention_mask"][0].unsqueeze(0).to(device)

    # Convert to TorchScript
    scripted_model_path = os.path.join(final_dir, "roberta_mlp_best_model_torchscript.pt")
    scripted_model = torch.jit.trace(model,(real_input_ids, real_attention_mask))
    scripted_model.save(scripted_model_path)

    # Convert to ONNX
    onnx_model_path = os.path.join(final_dir, "roberta_mlp_best_model.onnx")
    torch.onnx.export(
    model, 
    (real_input_ids, real_attention_mask), 
    onnx_model_path,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch_size"}, "attention_mask": {0: "batch_size"}},
    opset_version=14
    )
    print(f"Best model saved: {best_model_path}")
    print(f"Converted to TorchScript: {scripted_model_path}")
    print(f"Converted to ONNX: {onnx_model_path}")

def train_model(model, train_loader, val_loader, 
                optimizer, criterion, device, 
                checkpoint_dir, final_dir, epochs=3):
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
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"\nEpoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}\n")

        # Save checkpoint after every epoch
        save_checkpoint(model, optimizer, epoch, avg_val_loss, checkpoint_dir)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_best_model(model, device, train_loader, final_dir)