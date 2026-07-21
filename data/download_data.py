import os
import argparse
import pandas as pd
import yfinance as yf

RAW_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data", "raw")


def ensure_raw_dir() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)


def download_symbol(symbol: str, start: str = None, end: str = None, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    ensure_raw_dir()
    if start and end:
        df = yf.download(symbol, start=start, end=end, interval=interval)
    else:
        df = yf.download(symbol, period=period, interval=interval)
    if df.empty:
        raise ValueError(f"No data downloaded for symbol {symbol}")
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df = df.reset_index()
    if df.columns[0] != "Date":
        df.columns = ["Date"] + [str(col) for col in df.columns[1:]]
    csv_path = os.path.join(RAW_DIR, f"{symbol}.csv")
    df.to_csv(csv_path, index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download raw OHLCV data for a stock symbol")
    parser.add_argument("symbol", type=str, help="Ticker symbol")
    parser.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--period", type=str, default="5y", help="Download period if start/end omitted")
    parser.add_argument("--interval", type=str, default="1d", help="Data interval")
    args = parser.parse_args()
    df = download_symbol(args.symbol, start=args.start, end=args.end, period=args.period, interval=args.interval)
    print(f"Downloaded {len(df)} rows for {args.symbol} to {os.path.join(RAW_DIR, f'{args.symbol}.csv')}")
