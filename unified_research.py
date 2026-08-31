"""Build the unified research dataset from all downloaded source tables.

The market table is first transformed with the same causal technical feature
engineering used by the trading model. Other sources are merged with backward
as-of joins and then lagged conservatively so they cannot leak information from
a later publication period.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from trading_bot import Config, make_features

ROOT=Path(__file__).resolve().parent; OUT=ROOT/"outputs"

def _load(path:Path)->pd.DataFrame:
    if not path.exists(): return pd.DataFrame()
    d=pd.read_csv(path)
    if "Date" not in d: return pd.DataFrame()
    d["Date"]=pd.to_datetime(d.Date,utc=True,errors="coerce"); return d.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date",keep="last")

def _prefix(d:pd.DataFrame,source:str)->pd.DataFrame:
    return d.rename(columns={c:f"{source}_{c}" for c in d.columns if c!="Date"})

def build()->pd.DataFrame:
    raw=_load(OUT/"binance_btcusdt_history.csv")
    if raw.empty: raise FileNotFoundError("Run trading_bot.py --mode data first")
    raw=raw[[c for c in ["Date","Open","High","Low","Close","Volume"] if c in raw]]
    cfg=Config(interval="1h",horizon_bars=6)
    result=make_features(raw,cfg).sort_values("Date").reset_index(drop=True)
    sources=[
        ("onchain",[OUT/"bitcoin_onchain_daily.csv"]),
        ("derivatives",[OUT/"derivatives_features.csv",OUT/"binance_derivatives.csv"]),
        ("macro",[OUT/"macro_features.csv",OUT/"macro_data.csv"]),
        ("sentiment",[OUT/"sentiment_features.csv"]),
        ("breadth",[OUT/"breadth_features.csv",OUT/"alternative_signals.csv",OUT/"etf_flows.csv"]),
    ]
    for name,paths in sources:
        src=next((_load(p) for p in paths if p.exists()),pd.DataFrame())
        if src.empty: continue
        src=_prefix(src,name).sort_values("Date")
        # A full UTC-day lag is deliberately used for non-price sources. This is
        # conservative for research and avoids assuming an exact publication time.
        src["Date"]=src.Date+pd.Timedelta(days=1)
        result=pd.merge_asof(result.sort_values("Date"),src,on="Date",direction="backward")
    result.to_csv(OUT/"unified_dataset.csv",index=False)
    return result

if __name__=="__main__":
    d=build(); print(f"Unified rows: {len(d)}; columns: {len(d.columns)}")
