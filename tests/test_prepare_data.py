import os
import pandas as pd

from src.prepare_data import add_technical_indicators, build_targets, clean_data


def test_clean_data_removes_nan_and_inf(tmp_path):
    df = pd.DataFrame({
        "Close": [100.0, float("nan"), 102.0, float("inf")],
        "Volume": [100, 120, 130, 140],
    }, index=pd.date_range("2025-01-01", periods=4))
    clean = clean_data(df)
    assert clean.isna().sum().sum() == 0
    assert not clean.isin([float("inf"), float("-inf")]).any().any()


def test_add_technical_indicators_creates_features():
    df = pd.DataFrame({
        "Close": [100, 102, 101, 103, 105, 104, 106],
        "Volume": [1000, 1100, 1050, 1150, 1200, 1250, 1300],
    }, index=pd.date_range("2025-01-01", periods=7))
    out = add_technical_indicators(df)
    assert "ma_5" in out.columns
    assert "rsi_14" in out.columns
    assert "macd" in out.columns
    assert out.shape[0] > 0


def test_build_targets_shift_forward():
    df = pd.DataFrame({
        "Close": [100, 102, 101, 103, 105],
        "Volume": [1000, 1100, 1050, 1150, 1200],
    }, index=pd.date_range("2025-01-01", periods=5))
    df = add_technical_indicators(df)
    out = build_targets(df)
    assert "target_return" in out.columns
    assert "target_direction" in out.columns
    assert out["target_direction"].isin([0, 1]).all()
