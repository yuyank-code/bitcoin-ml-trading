import numpy as np
import pandas as pd

from trading_bot import Config, make_features, validate_ohlcv


def sample_ohlcv(n=900):
    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = 40000 + np.cumsum(np.sin(np.arange(n) / 20) * 20 + 5)
    close = np.maximum(close, 1000)
    return pd.DataFrame({
        "Date": dates,
        "Open": close,
        "High": close * 1.005,
        "Low": close * 0.995,
        "Close": close,
        "Volume": np.full(n, 100.0),
    })


def test_validation_accepts_valid_data():
    validate_ohlcv(sample_ohlcv())


def test_features_have_no_future_feature_dependency():
    cfg = Config(horizon_bars=6)
    d = sample_ohlcv()
    a = make_features(d, cfg)
    altered = d.copy()
    altered.loc[len(altered) - 1, "Close"] *= 10
    b = make_features(altered, cfg)
    cols = [c for c in a.columns if c in [
        "ret_1", "ret_24", "rsi14", "macd", "atr_pct", "volume_ratio", "trend_24_168"
    ]]
    # All rows except the final look-ahead region must be identical.
    merged = a.merge(b, on="Date", suffixes=("_a", "_b"))
    stable = merged[merged.Date < merged.Date.max() - pd.Timedelta(hours=7)]
    for c in cols:
        assert np.allclose(stable[f"{c}_a"], stable[f"{c}_b"], equal_nan=True)
