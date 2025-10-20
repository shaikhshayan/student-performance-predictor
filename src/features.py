def add_features(df):
    df = df.copy()
    if 'G1' in df.columns and 'G2' in df.columns:
        df['prev_avg'] = (df['G1'] + df['G2']) / 2
    return df
