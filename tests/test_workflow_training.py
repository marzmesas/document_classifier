# import pytest
# import os
# import torch
# import torch.nn as nn
# from unittest.mock import patch, MagicMock
# import pandas as pd
# from src.workflows.training import train_workflow # Esta funcion no existe, por que no se rompe el import?

# @pytest.fixture
# def mock_data():
#     # Create a small mock dataset
#     data = {
#         '5485': ['1Sample text one', '2Sample text two', '3Sample text three', '4Sample text four']
#     }
#     return pd.DataFrame(data)

# @pytest.fixture
# def mock_model():
#     # Simple model that just returns random outputs
#     class SimpleModel(nn.Module):
#         def __init__(self, model_name, num_classes):
#             super().__init__()
#             self.num_classes = num_classes
#             # Just a dummy parameter so the optimizer has something to work with
#             self.dummy_param = nn.Parameter(torch.randn(1))
            
#         def forward(self, input_ids, attention_mask):
#             batch_size = input_ids.shape[0]
#             return torch.randn(batch_size, self.num_classes)
    
#     return SimpleModel

# @pytest.fixture
# def mock_dataset():
#     # Simple dataset that returns random tensors
#     class SimpleDataset:
#         def __init__(self, texts, labels, tokenizer_name, max_length=128):
#             self.texts = texts
#             self.labels = labels
#             self.length = len(texts)
            
#         def __len__(self):
#             return self.length
            
#         def __getitem__(self, idx):
#             return {
#                 "input_ids": torch.randint(0, 1000, (10,)),
#                 "attention_mask": torch.ones(10),
#                 "labels": torch.tensor(0)  # Always return 0 for simplicity
#             }
    
#     return SimpleDataset

# def test_train_workflow(mock_data, mock_model, mock_dataset, tmp_path):
#     # Create directories for model outputs
#     checkpoint_dir = os.path.join(tmp_path, "checkpoints")
#     final_model_dir = os.path.join(tmp_path, "final_model")
#     os.makedirs(checkpoint_dir, exist_ok=True)
#     os.makedirs(final_model_dir, exist_ok=True)
    
#     # Mock dependencies
#     with patch('src.models.base_model.TransformerMLP', mock_model):
#         with patch('src.datasets.text_dataset.TextDataset', mock_dataset):
#             with patch('torch.utils.data.DataLoader') as mock_dataloader:
#                 with patch('src.models.train.train_model') as mock_train:
#                     with patch('src.models.evaluate.test_model') as mock_test:
#                         # Configure mocks
#                         mock_dataloader.return_value = MagicMock()
#                         mock_test.return_value = 0.85  # 85% accuracy
                        
#                         # Call the training workflow
#                         train_workflow(
#                             data=mock_data,
#                             model_name=mock_model,
#                             num_classes=8,
#                             batch_size=16,
#                             checkpoint_dir=checkpoint_dir,
#                             final_model_dir=final_model_dir
#                         )
                        
#                         # Verify the workflow called the expected functions
#                         assert mock_train.call_count == 1
#                         assert mock_test.call_count == 1

# def test_train_workflow_with_custom_params(mock_data, mock_model, mock_dataset, tmp_path):
#     # Create directories for model outputs
#     checkpoint_dir = os.path.join(tmp_path, "checkpoints")
#     final_model_dir = os.path.join(tmp_path, "final_model")
#     os.makedirs(checkpoint_dir, exist_ok=True)
#     os.makedirs(final_model_dir, exist_ok=True)
    
#     # Mock dependencies
#     with patch('src.models.base_model.TransformerMLP', mock_model):
#         with patch('src.datasets.text_dataset.TextDataset', mock_dataset):
#             with patch('torch.utils.data.DataLoader') as mock_dataloader:
#                 with patch('src.models.train.train_model') as mock_train:
#                     with patch('src.models.evaluate.test_model') as mock_test:
#                         with patch('torch.optim.AdamW'):
#                             with patch('torch.nn.CrossEntropyLoss'):
#                                 # Configure mocks
#                                 mock_dataloader.return_value = MagicMock()
#                                 mock_test.return_value = 0.90  # 90% accuracy
                                
#                                 # Call the training workflow with custom parameters
#                                 train_workflow(
#                                     data=mock_data,
#                                     model_name=mock_model,
#                                     num_classes=4,
#                                     batch_size=32,
#                                     learning_rate=5e-5,
#                                     epochs=5,
#                                     checkpoint_dir=checkpoint_dir,
#                                     final_model_dir=final_model_dir
#                                 )
                                
#                                 # Verify the workflow called the expected functions with custom params
#                                 mock_train.assert_called_once()
#                                 # Check that epochs=5 was passed to train_model
#                                 assert mock_train.call_args[1]['epochs'] == 5 