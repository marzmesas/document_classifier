import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import yaml
import os
import logging
import mlflow
# Set the TOKENIZERS_PARALLELISM environment variable before importing transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer
from src.models.train import train_model
from src.models.evaluate import evaluate_model
from src.models.base_model import TransformerMLP
from src.data.data_prep import prepare_data
from src.datasets.text_dataset import TextDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load configuration from config.yaml
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    logger.info("Configuration loaded successfully")
except Exception as e:
    logger.error(f"Error loading configuration: {str(e)}")
    # Fallback to defaults if config can't be loaded
    config = {
        "training": {
            "backbone_transformer": "roberta-base",
            "checkpoint_dir": "src/models/checkpoints",
            "final_model_dir": "src/models/final_model",
            "batch_size": 8,
            "learning_rate": 2e-5,
            "train_test_split": 0.1,
            "val_split": 0.1,
            "random_seed": 42
        },
        "data": {
            "path": "src/data/raw/file.txt"
        },
        "mlflow": {
            "tracking_uri": "sqlite:///mlruns.db",
            "experiment_name": "document-classifier"
        }
    }
    logger.warning("Using default training configuration")

# Get values from config
training_config = config.get("training", {})
data_config = config.get("data", {})
mlflow_config = config.get("mlflow", {})

# Get model parameters from config with defaults
MODEL_NAME = training_config.get("backbone_transformer", "roberta-base")
CHECKPOINT_DIR = training_config.get("checkpoint_dir", "src/models/checkpoints")
FINAL_MODEL_DIR = training_config.get("final_model_dir", "src/models/final_model")
BATCH_SIZE = int(training_config.get("batch_size", 8))
LEARNING_RATE = float(training_config.get("learning_rate", 2e-5))
TRAIN_TEST_SPLIT = float(training_config.get("train_test_split", 0.1))
VAL_SPLIT = float(training_config.get("val_split", 0.1))
RANDOM_SEED = int(training_config.get("random_seed", 42))
EPOCHS = int(training_config.get("epochs", 3))

# Configure MLflow
MLFLOW_TRACKING_URI = mlflow_config.get("tracking_uri", "sqlite:///mlruns.db")
MLFLOW_EXPERIMENT_NAME = mlflow_config.get("experiment_name", "document-classifier")

# Set up MLflow tracking
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# Load data
path_csv = data_config.get("path", "src/data/raw/file.txt")
logger.info(f"Loading data from {path_csv}")
df = pd.read_csv(path_csv)
X, y, df = prepare_data(df)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

logger.info(f"Using model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X.tolist(), y.tolist(), 
    test_size=TRAIN_TEST_SPLIT, 
    random_state=RANDOM_SEED
)

# Subsplit training into training and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, 
    test_size=VAL_SPLIT, 
    random_state=RANDOM_SEED
)

logger.info(f"Training set size: {len(X_train)}")
logger.info(f"Validation set size: {len(X_val)}")
logger.info(f"Test set size: {len(X_test)}")

# Create datasets
train_dataset = TextDataset(X_train, y_train, MODEL_NAME)
val_dataset = TextDataset(X_val, y_val, MODEL_NAME)
test_dataset = TextDataset(X_test, y_test, MODEL_NAME)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f'Running the workflow on device: {device}')

# Get the number of unique classes from your labels
num_classes = len(set(y_train)) 
logger.info(f"Number of classes: {num_classes}")

# Start MLflow run
with mlflow.start_run() as run:
    run_id = run.info.run_id
    logger.info(f"MLflow Run ID: {run_id}")
    
    # Log parameters
    mlflow.log_params({
        "model_name": MODEL_NAME,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "random_seed": RANDOM_SEED,
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "num_classes": num_classes,
        "device": str(device),
        "epochs": EPOCHS
    })
    
    # Initialize model
    model = TransformerMLP(MODEL_NAME, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Ensure directories exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    
    # Train model
    logger.info("Starting model training...")
    train_model(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        criterion, 
        device, 
        CHECKPOINT_DIR, 
        FINAL_MODEL_DIR,
        epochs=EPOCHS,
        mlflow_run=run  # Pass the MLflow run to the train function
    )
    
    # Test trained model 
    logger.info("Evaluating model on test set...")
    test_results = evaluate_model(model, test_loader, device)
    logger.info(f"Test results: {test_results}")
    
    # Log metrics from test results
    mlflow.log_metrics({
        "test_accuracy": test_results["Accuracy"],
        "test_precision": test_results["Precision"],
        "test_recall": test_results["Recall"],
        "test_f1": test_results["F1-Score"]
    })
    
    # Log models as artifacts
    best_model_path = os.path.join(FINAL_MODEL_DIR, "roberta_mlp_best_model.pth")
    scripted_model_path = os.path.join(FINAL_MODEL_DIR, "roberta_mlp_best_model_torchscript.pt")
    
    mlflow.log_artifact(best_model_path, "model")
    mlflow.log_artifact(scripted_model_path, "model/torchscript")
    
    # Log model with the MLflow model registry
    # Get a sample input for model signature inference
    sample_input = next(iter(train_loader))
    sample_input_ids = sample_input["input_ids"][0].unsqueeze(0).to(device)
    sample_attention_mask = sample_input["attention_mask"][0].unsqueeze(0).to(device)
    
    # Create an input example dictionary
    input_example = {
        "input_ids": sample_input_ids.cpu().numpy(),
        "attention_mask": sample_attention_mask.cpu().numpy()
    }
    
    # Log model with input example for signature inference
    mlflow.pytorch.log_model(
        model, 
        "pytorch_model",
        registered_model_name="document_classifier",
        input_example=input_example
    )
    
    logger.info(f"Training completed. MLflow run ID: {run_id}")

