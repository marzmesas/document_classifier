def preprocess_labels_and_text(df):
    labels = [int(i[0]) for i in df['5485']]
    df['labels'] = labels
    df = df.rename({'5485': 'doc_text'}, axis=1)
    df['doc_text'] = df['doc_text'].str[1:]
    return df

def prepare_data(df):
    df = preprocess_labels_and_text(df)
    X = df['doc_text']
    y = df['labels']
    return X, y, df