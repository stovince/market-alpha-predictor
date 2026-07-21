import numpy as np
import pandas as pd


def compute_rolling_features(df: pd.DataFrame, windows=[5, 10, 20]) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"ma_{w}"] = df["Close"].rolling(window=w, min_periods=1).mean()
        df[f"std_{w}"] = df["Close"].rolling(window=w, min_periods=1).std()
    return df


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return"] = df["Close"].pct_change()
    df["log_return"] = np.log1p(df["return"])
    df = compute_rolling_features(df)
    df["volume_change"] = df["Volume"].pct_change()
    df["volume_ma_10"] = df["Volume"].rolling(window=10, min_periods=1).mean()
    df["macd"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain, index=df.index).rolling(window=14, min_periods=1).mean()
    avg_loss = pd.Series(loss, index=df.index).rolling(window=14, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df["rsi_14"] = df["rsi_14"].fillna(50)
    df["return_1d"] = df["Close"].pct_change(periods=1)
    df["return_2d"] = df["Close"].pct_change(periods=2)
    return df
import pandas as pd

def compute_rolling_features(df: pd.DataFrame, windows = [5, 10, 20]) -> pd.DataFrame:
    for w in windows:
        df[f"ma_{w}"] = df['Close'].rolling(window=w).mean()
        df[f"std_{w}"] =df['Close'].rolling(window=w).std()
    return df.dropna()
