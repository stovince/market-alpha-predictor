import pandas as pd

def compute_rolling_features(df: pd.DataFrame, windows = [5, 10, 20]) -> pd.DataFrame:
    for w in windows:
        df[f"ma_{w}"] = df['Close'].rolling(window=w).mean()
        df[f"std_{w}"] =df['Close'].rolling(window=w).std()
    return df.dropna()
