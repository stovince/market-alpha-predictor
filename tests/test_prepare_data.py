import unittest
import pandas as pd

from src.prepare_data import add_technical_indicators, build_targets, clean_data


class TestPrepareData(unittest.TestCase):
    def test_clean_data_removes_nan_and_inf(self):
        df = pd.DataFrame({
            "Close": [100.0, float("nan"), 102.0, float("inf")],
            "Volume": [100, 120, 130, 140],
        }, index=pd.date_range("2025-01-01", periods=4))
        clean = clean_data(df)
        self.assertEqual(clean.isna().sum().sum(), 0)
        self.assertFalse(clean.isin([float("inf"), float("-inf")]).any().any())

    def test_add_technical_indicators_creates_features(self):
        df = pd.DataFrame({
            "Close": [100, 102, 101, 103, 105, 104, 106],
            "Volume": [1000, 1100, 1050, 1150, 1200, 1250, 1300],
        }, index=pd.date_range("2025-01-01", periods=7))
        out = add_technical_indicators(df)
        self.assertIn("ma_5", out.columns)
        self.assertIn("rsi_14", out.columns)
        self.assertIn("macd", out.columns)
        self.assertGreater(out.shape[0], 0)

    def test_build_targets_shift_forward(self):
        df = pd.DataFrame({
            "Close": [100, 102, 101, 103, 105],
            "Volume": [1000, 1100, 1050, 1150, 1200],
        }, index=pd.date_range("2025-01-01", periods=5))
        df = add_technical_indicators(df)
        out = build_targets(df)
        self.assertIn("target_return", out.columns)
        self.assertIn("target_direction", out.columns)
        self.assertTrue(out["target_direction"].isin([0, 1]).all())


if __name__ == "__main__":
    unittest.main()
