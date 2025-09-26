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

