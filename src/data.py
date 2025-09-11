from __future__ import annotations
import pandas as pd
import yfinance as yf




def load_price_history(symbol: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
if df.empty:
raise ValueError(f"Ingen data hittad för {symbol} ({period}, {interval}).")

df = df.rename(columns=str.lower)

cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
df = df[cols].dropna()
df.index.name = "datum"
return df
