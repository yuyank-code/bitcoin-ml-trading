"""Free alternative signals: Binance market breadth and optional ETF/Trends files.

ETF flow and Google Trends providers change frequently, so the production code
accepts normalized CSVs rather than scraping brittle pages. Binance breadth is
fully programmatic and public.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import requests

OUT=Path(__file__).resolve().parent/"outputs"; OUT.mkdir(exist_ok=True)
BASE="https://api.binance.com"


def _get(path, params):
    r=requests.get(BASE+path,params=params,timeout=30); r.raise_for_status(); return r.json()


def market_breadth(limit=1000):
    rows=_get("/api/v3/ticker/24hr",{})
    d=pd.DataFrame(rows)
    if d.empty: return pd.DataFrame(columns=["Date"])
    d=d[d.symbol.str.endswith("USDT")].copy()
    d["priceChangePercent"]=pd.to_numeric(d.priceChangePercent,errors="coerce")
    d["quoteVolume"]=pd.to_numeric(d.quoteVolume,errors="coerce")
    d["Date"]=pd.Timestamp.now(tz="UTC").floor("h")
    adv=(d.priceChangePercent>0).sum(); dec=(d.priceChangePercent<0).sum(); unchanged=(d.priceChangePercent==0).sum()
    totalvol=d.quoteVolume.sum(); upvol=d.loc[d.priceChangePercent>0,"quoteVolume"].sum(); downvol=d.loc[d.priceChangePercent<0,"quoteVolume"].sum()
    return pd.DataFrame([{"Date":d.Date.iloc[0],"breadth_advancers":int(adv),"breadth_decliners":int(dec),"breadth_unchanged":int(unchanged),"breadth_ratio":float(adv/max(dec,1)),"breadth_volume_ratio":float(upvol/max(downvol,1)),"breadth_up_volume_share":float(upvol/max(totalvol,1))}])


def normalize_external_csv(path:str|Path, source:str)->pd.DataFrame:
    d=pd.read_csv(path); date_col=next((c for c in d.columns if c.lower() in {"date","datetime","timestamp","time"}),None)
    if date_col is None: raise ValueError("External CSV requires a date/datetime/timestamp column")
    d=d.rename(columns={date_col:"Date"}); d["Date"]=pd.to_datetime(d.Date,utc=True,errors="coerce"); d=d.dropna(subset=["Date"]).sort_values("Date")
    return d.rename(columns={c:f"{source}_{c}" for c in d.columns if c!="Date"})


def add_features(d:pd.DataFrame)->pd.DataFrame:
    x=d.sort_values("Date").copy()
    for c in x.columns:
        if c=="Date": continue
        x[c]=pd.to_numeric(x[c],errors="coerce")
        x[f"{c}_chg1"]=x[c].pct_change(); x[f"{c}_z30"]=(x[c]-x[c].rolling(30).mean())/x[c].rolling(30).std()
    return x.replace([np.inf,-np.inf],np.nan)

if __name__=="__main__":
    b=market_breadth(); f=add_features(b); f.to_csv(OUT/"breadth_features.csv",index=False); print(f.to_json(orient="records"))
