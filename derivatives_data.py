"""Binance USD-M derivatives market-data layer.

Public market-data only: no account credentials are required. Historical coverage
is explicitly respected because several Binance derivatives statistics endpoints
only retain recent observations.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://fapi.binance.com"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)


def _get(path: str, params: dict) -> list | dict:
    r = requests.get(BASE + path, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _frame(rows, time_col="timestamp") -> pd.DataFrame:
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    if time_col in d:
        d["Date"] = pd.to_datetime(d[time_col], unit="ms", utc=True)
    return d


def fetch_open_interest(symbol="BTCUSDT", period="1h", limit=500):
    rows = _get("/futures/data/openInterestHist", {"symbol": symbol, "period": period, "limit": min(limit, 500)})
    d = _frame(rows)
    if d.empty: return d
    for c in ["sumOpenInterest", "sumOpenInterestValue"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d[["Date", "sumOpenInterest", "sumOpenInterestValue"]].rename(columns={"sumOpenInterest":"oi_contracts","sumOpenInterestValue":"oi_usdt"})


def fetch_funding(symbol="BTCUSDT", limit=1000):
    rows = _get("/fapi/v1/fundingRate", {"symbol": symbol, "limit": min(limit, 1000)})
    d = _frame(rows, "fundingTime")
    if d.empty: return d
    d["funding_rate"] = pd.to_numeric(d["fundingRate"], errors="coerce")
    return d[["Date", "funding_rate"]]


def fetch_global_long_short(symbol="BTCUSDT", period="1h", limit=500):
    rows = _get("/futures/data/globalLongShortAccountRatio", {"symbol": symbol, "period": period, "limit": min(limit, 500)})
    d = _frame(rows)
    if d.empty: return d
    for c in ["longShortRatio", "longAccount", "shortAccount"]: d[c] = pd.to_numeric(d[c], errors="coerce")
    return d[["Date", "longShortRatio", "longAccount", "shortAccount"]].rename(columns={"longShortRatio":"global_long_short_ratio","longAccount":"global_long_account","shortAccount":"global_short_account"})


def fetch_top_position_ratio(symbol="BTCUSDT", period="1h", limit=500):
    rows = _get("/futures/data/topLongShortPositionRatio", {"symbol": symbol, "period": period, "limit": min(limit, 500)})
    d = _frame(rows)
    if d.empty: return d
    for c in ["longShortRatio", "longAccount", "shortAccount"]: d[c] = pd.to_numeric(d[c], errors="coerce")
    return d[["Date", "longShortRatio", "longAccount", "shortAccount"]].rename(columns={"longShortRatio":"top_position_long_short_ratio","longAccount":"top_position_long","shortAccount":"top_position_short"})


def fetch_top_account_ratio(symbol="BTCUSDT", period="1h", limit=500):
    rows = _get("/futures/data/topLongShortAccountRatio", {"symbol": symbol, "period": period, "limit": min(limit, 500)})
    d = _frame(rows)
    if d.empty: return d
    for c in ["longShortRatio", "longAccount", "shortAccount"]: d[c] = pd.to_numeric(d[c], errors="coerce")
    return d[["Date", "longShortRatio", "longAccount", "shortAccount"]].rename(columns={"longShortRatio":"top_account_long_short_ratio","longAccount":"top_account_long","shortAccount":"top_account_short"})


def fetch_taker_ratio(symbol="BTCUSDT", period="1h", limit=500):
    rows = _get("/futures/data/takerlongshortRatio", {"symbol": symbol, "period": period, "limit": min(limit, 500)})
    d = _frame(rows)
    if d.empty: return d
    for c in ["buySellRatio", "buyVol", "sellVol"]: d[c] = pd.to_numeric(d[c], errors="coerce")
    return d[["Date", "buySellRatio", "buyVol", "sellVol"]].rename(columns={"buySellRatio":"taker_buy_sell_ratio","buyVol":"taker_buy_vol","sellVol":"taker_sell_vol"})


def fetch_basis(pair="BTCUSDT", contract_type="PERPETUAL", period="1h", limit=500):
    rows = _get("/futures/data/basis", {"pair": pair, "contractType": contract_type, "period": period, "limit": min(limit, 500)})
    d = _frame(rows)
    if d.empty: return d
    for c in ["basisRate", "annualizedBasisRate", "basis", "futuresPrice", "indexPrice"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d[["Date", "basisRate", "annualizedBasisRate", "basis", "futuresPrice", "indexPrice"]].rename(columns={"basisRate":"basis_rate","annualizedBasisRate":"annualized_basis_rate","basis":"basis_value","futuresPrice":"futures_price","indexPrice":"index_price"})


def collect_current(symbol="BTCUSDT"):
    """Collect the latest available public derivatives statistics."""
    frames = [
        fetch_open_interest(symbol, limit=500),
        fetch_funding(symbol, limit=1000),
        fetch_global_long_short(symbol, limit=500),
        fetch_top_position_ratio(symbol, limit=500),
        fetch_top_account_ratio(symbol, limit=500),
        fetch_taker_ratio(symbol, limit=500),
        fetch_basis(symbol, limit=500),
    ]
    merged = None
    for f in frames:
        if f.empty: continue
        f = f.sort_values("Date")
        merged = f if merged is None else pd.merge_asof(merged, f, on="Date", direction="nearest", tolerance=pd.Timedelta("2h"))
    if merged is None: raise RuntimeError("No derivatives data returned")
    merged.to_csv(OUT / "binance_derivatives.csv", index=False)
    return merged


def make_features(d: pd.DataFrame) -> pd.DataFrame:
    x = d.sort_values("Date").copy()
    numeric = [c for c in x.columns if c not in {"Date"}]
    for c in numeric:
        x[c] = pd.to_numeric(x[c], errors="coerce")
        x[f"{c}_chg1"] = x[c].pct_change()
        x[f"{c}_z24"] = (x[c] - x[c].rolling(24).mean()) / x[c].rolling(24).std()
    if "funding_rate" in x:
        x["funding_annualized_proxy"] = x["funding_rate"] * 3 * 365
    if "oi_usdt" in x and "index_price" in x:
        x["oi_price_scaled"] = x["oi_usdt"] / x["index_price"].replace(0, pd.NA)
    return x.replace([float("inf"), float("-inf")], pd.NA)


if __name__ == "__main__":
    data = collect_current()
    features = make_features(data)
    features.to_csv(OUT / "derivatives_features.csv", index=False)
    print(json.dumps({"rows": len(features), "columns": list(features.columns)}, indent=2))
