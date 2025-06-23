import yfinance as yf
import pandas as pd

def download_symbol(symbol: str, period: str = "1y", interval: str = "1h") -> pd.DataFrame:
    "Fetches the historical price for a given symbol"
    df = yf.download(symbol, period=period, interval=interval)
    df.to_csv(f"data/raw/{symbol}.csv")
    return df

if __name__ == "__main__":
    symbols = ["AAPL", "MSFT", "GOOG"]
    for sym in symbols:
        download_symbol(sym)
