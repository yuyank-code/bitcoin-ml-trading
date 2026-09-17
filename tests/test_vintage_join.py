import pandas as pd
import pytest

from vintage_join import asof_vintage_join


def test_revision_is_not_visible_before_availability():
    decisions = pd.DataFrame({"Date": pd.to_datetime([
        "2024-01-02 12:00", "2024-01-03 12:00"
    ], utc=True)})
    source = pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-01", "2024-01-01"], utc=True),
        "availability_time": pd.to_datetime([
            "2024-01-02 00:00", "2024-01-03 00:00"
        ], utc=True),
        "value": [100.0, 120.0],
    })
    out = asof_vintage_join(decisions, source)
    assert out["value"].tolist() == [100.0, 120.0]


def test_missing_availability_time_fails_closed():
    decisions = pd.DataFrame({"Date": pd.to_datetime(["2024-01-02"], utc=True)})
    source = pd.DataFrame({"Date": pd.to_datetime(["2024-01-01"], utc=True), "value": [1.0]})
    with pytest.raises(ValueError, match="availability_time"):
        asof_vintage_join(decisions, source)
