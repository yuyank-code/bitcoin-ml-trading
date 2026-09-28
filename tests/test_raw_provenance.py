import pandas as pd
import pytest

from research_v3_event_time import validate_raw_ohlcv
from trading_bot import Config


def _raw():
    dates = pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC")
    return pd.DataFrame({
        "Date": dates,
        "Open": [100.0] * 8,
        "High": [101.0] * 8,
        "Low": [99.0] * 8,
        "Close": [100.0] * 8,
        "Volume": [10.0] * 8,
    })


def test_valid_hourly_stream_passes():
    validate_raw_ohlcv(_raw(), Config())


@pytest.mark.parametrize("mutate", [
    lambda d: d.assign(Date=d["Date"].iloc[::-1].to_numpy()),
    lambda d: d.drop(index=3).reset_index(drop=True),
    lambda d: d.assign(Date=d["Date"].where(d.index != 3, d["Date"].iloc[2])),
    lambda d: d.assign(Close=d["Close"].where(d.index != 3, float("nan"))),
    lambda d: d.assign(High=d["High"].where(d.index != 3, 98.0)),
    lambda d: d.assign(Low=d["Low"].where(d.index != 3, 102.0)),
])
def test_malformed_raw_stream_fails_closed(mutate):
    with pytest.raises(AssertionError):
        validate_raw_ohlcv(mutate(_raw()), Config())
