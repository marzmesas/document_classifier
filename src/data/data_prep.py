def preprocess_labels_and_text(df):
    """
    Extract label information and format text from the input dataframe.
    
    The function assumes the label is the first character of the '5485' column 
    and the document text is the remainder of that same field.
    
    Args:
        df (pandas.DataFrame): DataFrame containing a '5485' column with combined label and text data
        
    Returns:
        pandas.DataFrame: Processed dataframe with separate 'labels' and 'doc_text' columns
    """
    labels = [int(i[0]) for i in df['5485']]
    df['labels'] = labels
    df = df.rename({'5485': 'doc_text'}, axis=1)
    df['doc_text'] = df['doc_text'].str[1:]
    return df

def prepare_data(df):
    """
    Prepare dataset for model training by separating features and target labels.
    
    This function calls preprocess_labels_and_text() to extract labels from the text,
    then returns the document text as features (X) and the labels as targets (y).
    
    Args:
        df (pandas.DataFrame): Raw dataframe with '5485' column
        
    Returns:
        tuple: Contains:
            - X (pandas.Series): Document text features
            - y (pandas.Series): Target labels
            - df (pandas.DataFrame): Processed dataframe with separated features and labels
    """
    df = preprocess_labels_and_text(df)
    X = df['doc_text']
    y = df['labels']
    return X, y, df