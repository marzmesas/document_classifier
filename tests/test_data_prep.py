import pytest
import pandas as pd
import numpy as np
from src.data.data_prep import prepare_data, preprocess_labels_and_text

def test_preprocess_labels_and_text():
    # Create a sample dataframe with the expected column '5485'
    # The format appears to be a string where first character is the label
    # and the rest is the document text
    data = {
        '5485': ['1Sample text one', '2Sample text two', '3Sample text three']
    }
    df = pd.DataFrame(data)
    
    # Process the dataframe
    processed_df = preprocess_labels_and_text(df)
    
    # Check that the dataframe has the expected columns
    assert 'doc_text' in processed_df.columns
    assert 'labels' in processed_df.columns
    
    # Check that text preprocessing was applied (first character removed)
    assert processed_df['doc_text'].iloc[0] == 'Sample text one'
    assert processed_df['doc_text'].iloc[1] == 'Sample text two'
    assert processed_df['doc_text'].iloc[2] == 'Sample text three'
    
    # Check that labels are properly extracted from first character
    assert processed_df['labels'].iloc[0] == 1
    assert processed_df['labels'].iloc[1] == 2
    assert processed_df['labels'].iloc[2] == 3

def test_prepare_data():
    # Create a sample dataframe with the expected column '5485'
    data = {
        '5485': ['1Sample text one', '2Sample text two', '3Sample text three']
    }
    df = pd.DataFrame(data)
    
    # Call prepare_data
    X, y, processed_df = prepare_data(df)
    
    # Check outputs
    assert len(X) == len(df)
    assert len(y) == len(df)
    assert isinstance(X, pd.Series)
    assert isinstance(y, pd.Series)
    assert isinstance(processed_df, pd.DataFrame)
    
    # Check that X contains the text data with first character removed
    assert X.iloc[0] == 'Sample text one'
    assert X.iloc[1] == 'Sample text two'
    assert X.iloc[2] == 'Sample text three'
    
    # Check that y contains the labels from the first character
    assert y.iloc[0] == 1
    assert y.iloc[1] == 2
    assert y.iloc[2] == 3 