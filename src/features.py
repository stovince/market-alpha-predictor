from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Tuple, List
import numpy as np
import pandas as pd
import tensorflow as tf

class FeatureConfig:
    col_open: str = "open"
    col_high: str = "high"
    col_low: str = "low"
    col_close: str = "close"
    col_volume: str = "volume"

    sma_windows: Iterable[int] = (5, 10, 20, 50)
    ema_windows: Iterable[int] = (5, 10, 20)
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_window: int = 20
    bb_std: float = 2.0
    atr_window: int = 14
    rv_windows: Iterable[int] = (10, 20) 

    return_lags: Iterable[int] = (1, 5, 10)
    price_lags: Iterable[int] = (1, 2, 3)

    cmf_window: int = 20

    create_target: bool = True
    target_horizon: int = 1
    target_type: Literal["regression", "classification"] = "regression"
    cls_ret_threshold: float = 0.0

    drop_na: bool = True
    as_float32: bool = True

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def _true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    pc = c.shift(1)
    return pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)

def rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(window, min_periods=window).mean()
    roll_down = down.rolling(window, min_periods=window).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(close: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    line = ema_fast - ema_slow
    sig = _ema(line, signal)
    hist = line - sig
    return line, sig, hist

def bollinger(close: pd.Series, window: int, num_std: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ma = close.rolling(window, min_periods=window).mean()
    sd = close.rolling(window, min_periods=window).std()
    upper = ma + num_std * sd
    lower = ma - num_std * sd
    return upper, ma, lower

def atr(h: pd.Series, l: pd.Series, c: pd.Series, window: int) -> pd.Series:
    tr = _true_range(h, l, c)
    return tr.rolling(window, min_periods=window).mean()

def realized_vol(logret: pd.Series, window: int) -> pd.Series:
    return logret.rolling(window, min_periods=window).std() * np.sqrt(252)

def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume.fillna(0)).cumsum()

def chaikin_money_flow(h: pd.Series, l: pd.Series, c: pd.Series, v: pd.Series, window: int) -> pd.Series:
    rng = (h - l).replace(0, np.nan)
    mfm = ((c - l) - (h - c)) / rng
    mfv = mfm.fillna(0.0) * v.fillna(0)
    num = mfv.rolling(window, min_periods=window).sum()
    den = v.fillna(0).rolling(window, min_periods=window).sum().replace(0, np.nan)
    return num / den

def build_features(
    df: pd.DataFrame,
    cfg: Optional[FeatureConfig] = None
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Tar in OHLCV-DataFrame (index=DatetimeIndex) och returnerar (X, y) där X är feature-matris
    och y är mål (om cfg.create_target=True).
    """
    cfg = cfg or FeatureConfig()

    needed = [cfg.col_open, cfg.col_high, cfg.col_low, cfg.col_close, cfg.col_volume]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Saknar kolumn: {col}")

    o, h, l, c, v = cfg.col_open, cfg.col_high, cfg.col_low, cfg.col_close, cfg.col_volume
    data = df.copy().sort_index()

    data["logret_1"] = np.log(data[c] / data[c].shift(1))

    for w in cfg.sma_windows:
        sma = data[c].rolling(w, min_periods=w).mean()
        data[f"sma_{w}"] = sma
        data[f"close_over_sma_{w}"] = data[c] / sma - 1.0

    for w in cfg.ema_windows:
        ema = _ema(data[c], w)
        data[f"ema_{w}"] = ema
        data[f"close_over_ema_{w}"] = data[c] / ema - 1.0

    data[f"rsi_{cfg.rsi_window}"] = rsi(data[c], cfg.rsi_window)

    macd_line, macd_sig, macd_hist = macd(data[c], cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    data["macd_line"] = macd_line
    data["macd_signal"] = macd_sig
    data["macd_hist"] = macd_hist

    bb_u, bb_m, bb_l = bollinger(data[c], cfg.bb_window, cfg.bb_std)
    data["bb_upper"] = bb_u
    data["bb_middle"] = bb_m
    data["bb_lower"] = bb_l
    data["bb_width"] = (bb_u - bb_l) / bb_m

    data["atr"] = atr(data[h], data[l], data[c], cfg.atr_window)
    for w in cfg.rv_windows:
        data[f"rv_{w}"] = realized_vol(data["logret_1"], w)

    data["obv"] = on_balance_volume(data[c], data[v])
    data["cmf"] = chaikin_money_flow(data[h], data[l], data[c], data[v], cfg.cmf_window)

    for L in cfg.return_lags:
        data[f"logret_{L}"] = np.log(data[c] / data[c].shift(L))
    for L in cfg.price_lags:
        data[f"close_lag_{L}"] = data[c].shift(L)

    y: Optional[pd.Series] = None
    if cfg.create_target:
        future_logret = np.log(data[c].shift(-cfg.target_horizon) / data[c])
        if cfg.target_type == "regression":
            y = future_logret.rename("y_reg_logret")
        else:
            y = (future_logret > cfg.cls_ret_threshold).astype("int32").rename("y_cls_up")
        data = data.iloc[:-cfg.target_horizon]
        y = y.iloc[:-cfg.target_horizon]

    if cfg.drop_na:
        if y is not None:
            mask = data.notna().all(axis=1) & y.notna()
            data, y = data.loc[mask], y.loc[mask]
        else:
            data = data.dropna()

    if cfg.as_float32:
        for col in data.columns:
            data[col] = data[col].astype("float32")

    return data, y

def make_supervised_sequences(
    X: pd.DataFrame,
    y: Optional[pd.Series],
    window: int = 60,
    horizon: int = 1,
    as_dataset: bool = True,
    batch_size: int = 64,
    shuffle: bool = False,
    drop_remainder: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]] | tf.data.Dataset:
    Xv = X.values
    yv = y.values if y is not None else None
    n = len(Xv)
    if n < window + horizon:
        raise ValueError("För lite data för valt window/horizon.")

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    last = n - window - (0 if y is None else horizon) + 1
    for i in range(last):
        xs.append(Xv[i:i + window])
        if yv is not None:
            ys.append(yv[i + window - 1])  # target avser 'nästa steg' relativt sista i fönstret

    X_seq = np.stack(xs).astype("float32")
    y_seq = np.array(ys).astype("float32") if ys else None

    if not as_dataset:
        return (X_seq, y_seq)

    ds = tf.data.Dataset.from_tensor_slices((X_seq, y_seq)) if y_seq is not None else tf.data.Dataset.from_tensor_slices(X_seq)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X_seq))
    ds = ds.batch(batch_size, drop_remainder=drop_remainder).prefetch(tf.data.AUTOTUNE)
    return ds

if __name__ == "__main__":
    pass
