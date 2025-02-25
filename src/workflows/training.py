import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from transformers import AutoTokenizer
from models.train import train_model
from models.test import test_model
from models.base_model import TransformerMLP
from data.data_prep import prepare_data
from datasets.text_dataset import TextDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


path_csv = 'data/file.txt'
df = pd.read_csv(path_csv)
X, y, df = prepare_data(df)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# TODO: Convert constant into config.yaml loading
MODEL_NAME = "roberta-base"  # Change to "distilbert-base-uncased" for DistilBERT

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X.tolist(), y.tolist(), test_size=0.1, random_state=42)

# Subsplit training into training and validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)


train_dataset = TextDataset(X_train, y_train)
val_dataset = TextDataset(X_val, y_val)
test_dataset = TextDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Define model, loss function & optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print('Running the workflow in the following device: ', device)

# Get the number of unique classes from your labels
num_classes = len(set(y_train)) 

model = TransformerMLP(MODEL_NAME, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# Train model
train_model(model, train_loader, val_loader, optimizer, criterion, device)

# Test trained model 
test_model(model, test_loader, device)

