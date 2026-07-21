import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from indicators.compute_indicators import compute_technical_indicators


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

    df = pd.read_csv(csv_path)
    if "Date" not in df.columns:
        if df.columns[0].lower() == "date":
            df.columns = ["Date"] + list(df.columns[1:])
        else:
            raise ValueError("Raw CSV is missing a Date column")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how="any")
    df = df[~df.isin([np.inf, -np.inf]).any(axis=1)]
    return df.sort_index()


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = compute_technical_indicators(df)
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
