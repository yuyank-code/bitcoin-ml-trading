"""Free macro/global-market data layer.

Primary source: FRED public graph CSV downloads (no API key required for these
public series downloads).  The loader intentionally applies a one-observation
lag before macro observations become model features.  For release-sensitive
backtests, an ALFRED/FRED vintage feed should be used when an API key is
available; the current module never invents a release timestamp.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import requests

LOG = logging.getLogger("btc-ml.macro")
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


@dataclass(frozen=True)
class MacroSeries:
    name: str
    series_id: str
    # Approximate conservative availability lag used when joining a daily BTC model.
    lag_days: int


SERIES = (
    MacroSeries("fed_funds", "FEDFUNDS", 1),
    MacroSeries("cpi", "CPIAUCSL", 35),
    MacroSeries("treasury_2y", "DGS2", 1),
    MacroSeries("treasury_5y", "DGS5", 1),
    MacroSeries("treasury_10y", "DGS10", 1),
    MacroSeries("vix", "VIXCLS", 1),
    MacroSeries("sp500", "SP500", 1),
    MacroSeries("nasdaq100", "NASDAQ100", 1),
    MacroSeries("wti", "DCOILWTICO", 1),
    MacroSeries("gold", "GOLDAMGBD228NLBM", 1),
    MacroSeries("broad_dollar", "DTWEXBGS", 1),
)


def _read_fred(s: MacroSeries, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    url = FRED_CSV.format(series=s.series_id)
    params = {}
    if start:
        params["cosd"] = start
    if end:
        params["coed"] = end
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    d = pd.read_csv(io.BytesIO(r.content))
    d.columns = ["Date", s.name]
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    d[s.name] = pd.to_numeric(d[s.name], errors="coerce")
    return d.dropna(subset=["Date"]).drop_duplicates("Date").sort_values("Date")


def fetch_macro(start: str | date | None = None, end: str | date | None = None) -> pd.DataFrame:
    """Download all configured free FRED series and combine by observation date."""
    start_s = str(start)[:10] if start else None
    end_s = str(end)[:10] if end else None
    frames = []
    for s in SERIES:
        try:
            frames.append(_read_fred(s, start_s, end_s))
            LOG.info("Loaded FRED %s (%s)", s.series_id, s.name)
        except Exception as exc:
            LOG.warning("FRED series %s unavailable: %s", s.series_id, exc)
    if not frames:
        raise RuntimeError("No FRED macro series could be downloaded")
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="Date", how="outer")
    return out.sort_values("Date").reset_index(drop=True)


def engineer_macro_features(macro: pd.DataFrame) -> pd.DataFrame:
    """Create lagged, stationary macro features.

    A conservative lag is applied to each raw observation before it is exposed
    to the model. Daily market series use one day; CPI uses 35 days. This is a
    safety approximation, not a claim about the exact historical publication
    timestamp. Release-aware vintages are recommended for final research.
    """
    x = macro.copy().sort_values("Date").reset_index(drop=True)
    lag_map = {s.name: s.lag_days for s in SERIES}
    for name in [s.name for s in SERIES]:
        if name not in x:
            continue
        lag = lag_map[name]
        x[name] = pd.to_numeric(x[name], errors="coerce").shift(lag)
        x[f"macro_{name}_chg1"] = x[name].pct_change(1)
        x[f"macro_{name}_chg5"] = x[name].pct_change(5)
        x[f"macro_{name}_chg20"] = x[name].pct_change(20)
        roll = x[name].rolling(20, min_periods=10)
        x[f"macro_{name}_z20"] = (x[name] - roll.mean()) / roll.std()

    if {"treasury_10y", "treasury_2y"}.issubset(x.columns):
        x["macro_10y_2y_spread"] = x.treasury_10y - x.treasury_2y
        x["macro_10y_2y_spread_chg5"] = x.macro_10y_2y_spread.diff(5)
    if {"sp500", "nasdaq100"}.issubset(x.columns):
        x["macro_nasdaq_vs_sp500_20"] = x.nasdaq100.pct_change(20) - x.sp500.pct_change(20)
    if {"gold", "broad_dollar"}.issubset(x.columns):
        x["macro_gold_dollar_20"] = x.gold.pct_change(20) - x.broad_dollar.pct_change(20)
    x = x.replace([np.inf, -np.inf], np.nan)
    return x


def save_macro_dataset(out_path: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    d = engineer_macro_features(fetch_macro(start, end))
    d.to_csv(out_path, index=False)
    return d
