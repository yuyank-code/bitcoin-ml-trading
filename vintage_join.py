"""Point-in-time vintage helpers for exogenous research data.

A source with revisions must carry an explicit availability/release timestamp.
Observation-date deduplication is intentionally not used for revision-prone data.
"""
from __future__ import annotations
import pandas as pd


def validate_vintage_schema(df: pd.DataFrame) -> None:
    required = {"Date", "availability_time"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Revision-aware source missing required columns: {sorted(missing)}")


def asof_vintage_join(decisions: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    """Backward as-of join using source availability, not observation date.

    For each decision timestamp, the selected row is the latest source vintage
    whose availability_time is no later than that decision. Duplicate
    observation dates are allowed because multiple revisions are expected.
    """
    validate_vintage_schema(source)
    left = decisions.copy()
    right = source.copy()
    if "Date" not in left.columns:
        raise ValueError("decisions must contain Date")
    left["Date"] = pd.to_datetime(left["Date"], utc=True, errors="coerce")
    right["Date"] = pd.to_datetime(right["Date"], utc=True, errors="coerce")
    right["availability_time"] = pd.to_datetime(right["availability_time"], utc=True, errors="coerce")
    if right["availability_time"].isna().any():
        raise ValueError("Source contains invalid availability_time")
    right = right.sort_values(["availability_time", "Date"]).reset_index(drop=True)
    left = left.sort_values("Date").reset_index(drop=True)
    return pd.merge_asof(
        left,
        right,
        left_on="Date",
        right_on="availability_time",
        direction="backward",
        suffixes=("", "_source"),
    )
