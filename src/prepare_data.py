import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


RAW_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data", "processed")


@dataclass
class DataSplit:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    scaler: StandardScaler
    feature_columns: List[str]
    target_column: str


def ensure_directories() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_raw_csv(symbol: str) -> pd.DataFrame:
    csv_path = os.path.join(RAW_DIR, f"{symbol}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Raw CSV for {symbol} not found at {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how="any")
    df = df[~df.isin([np.inf, -np.inf]).any(axis=1)]
    return df.sort_index()


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return"] = df["Close"].pct_change()
    df["log_return"] = np.log1p(df["return"])
    for window in [5, 10, 20]:
        df[f"ma_{window}"] = df["Close"].rolling(window=window, min_periods=1).mean()
        df[f"ema_{window}"] = df["Close"].ewm(span=window, adjust=False).mean()
        df[f"vol_{window}"] = df["return"].rolling(window=window, min_periods=1).std()
    df["volume_change"] = df["Volume"].pct_change()
    df["volume_ma_10"] = df["Volume"].rolling(window=10, min_periods=1).mean()
    df["macd"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=1).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    df["rsi_14"] = 100 - (100 / (1 + rs)).fillna(50)
    df["return_1d"] = df["Close"].pct_change(periods=1)
    df["return_2d"] = df["Close"].pct_change(periods=2)
    df = df.dropna()
    return df


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["target_return"] = df["Close"].pct_change().shift(-1)
    df["target_direction"] = (df["target_return"] > 0).astype(int)
    return df.dropna()


def split_chronological(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    return train, val, test


def scale_features(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, feature_columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(train[feature_columns]), index=train.index, columns=feature_columns)
    X_val = pd.DataFrame(scaler.transform(val[feature_columns]), index=val.index, columns=feature_columns)
    X_test = pd.DataFrame(scaler.transform(test[feature_columns]), index=test.index, columns=feature_columns)
    return X_train, X_val, X_test, scaler


def prepare(symbol: str, train_ratio: float = 0.7, val_ratio: float = 0.15) -> DataSplit:
    ensure_directories()
    df = load_raw_csv(symbol)
    df = clean_data(df)
    df = add_technical_indicators(df)
    df = build_targets(df)
    train, val, test = split_chronological(df, train_ratio=train_ratio, val_ratio=val_ratio)
    feature_columns = [col for col in df.columns if col not in ["target_return", "target_direction"]]
    X_train, X_val, X_test, scaler = scale_features(train, val, test, feature_columns)
    y_train = train["target_direction"].copy()
    y_val = val["target_direction"].copy()
    y_test = test["target_direction"].copy()
    train.to_csv(os.path.join(PROCESSED_DIR, f"{symbol}_train.csv"))
    val.to_csv(os.path.join(PROCESSED_DIR, f"{symbol}_val.csv"))
    test.to_csv(os.path.join(PROCESSED_DIR, f"{symbol}_test.csv"))
    np.save(os.path.join(PROCESSED_DIR, f"{symbol}_scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join(PROCESSED_DIR, f"{symbol}_scaler_scale.npy"), scaler.scale_)
    return DataSplit(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        scaler=scaler,
        feature_columns=feature_columns,
        target_column="target_direction",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare financial data for TensorFlow modeling")
    parser.add_argument("--symbol", type=str, default="AAPL")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    args = parser.parse_args()
    ds = prepare(args.symbol, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    print(f"Prepared {args.symbol}: {len(ds.X_train)} train, {len(ds.X_val)} val, {len(ds.X_test)} test")
