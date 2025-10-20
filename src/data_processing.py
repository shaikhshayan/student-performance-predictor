import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path, sep=';'):
    return pd.read_csv(path, sep=sep)

def preprocess(df):
    df = df.copy()
    mapping = {'yes': 1, 'no': 0}
    for col in df.columns:
        if df[col].dtype == 'object' and set(df[col].unique()) <= {'yes', 'no'}:
            df[col] = df[col].map(mapping)
    df = pd.get_dummies(df, drop_first=True)
    return df

def split(df, target='G3', test_size=0.2, random_state=42):
    X = df.drop(columns=[target])
    y = df[df.columns[-1]]  # dynamically takes last column (usually G3)

    return train_test_split(X, y, test_size=test_size, random_state=random_state)

