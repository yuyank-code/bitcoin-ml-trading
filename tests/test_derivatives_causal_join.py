import pandas as pd

from derivatives_data import _causal_asof_merge


def test_future_point_is_never_selected_when_it_is_nearest():
    left = pd.DataFrame({"Date": pd.to_datetime(["2026-01-01 10:00"], utc=True), "target": [1]})
    right = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-01-01 09:50", "2026-01-01 10:05"], utc=True),
            "value": [10, 20],
        }
    )
    out = _causal_asof_merge(left, right, tolerance="2h")
    assert out.loc[0, "value"] == 10


def test_exact_timestamp_is_allowed():
    left = pd.DataFrame({"Date": pd.to_datetime(["2026-01-01 10:00"], utc=True)})
    right = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-01-01 10:00"], utc=True),
            "value": [7],
        }
    )
    out = _causal_asof_merge(left, right, tolerance="2h")
    assert out.loc[0, "value"] == 7


def test_source_outside_tolerance_is_missing():
    left = pd.DataFrame({"Date": pd.to_datetime(["2026-01-01 10:00"], utc=True)})
    right = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-01-01 07:59"], utc=True),
            "value": [7],
        }
    )
    out = _causal_asof_merge(left, right, tolerance="2h")
    assert pd.isna(out.loc[0, "value"])
