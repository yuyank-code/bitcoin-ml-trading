from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOG = logging.getLogger("btc-ml")

@dataclass
class Config:
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    history_years: int = 10
    initial_train_bars: int = 24 * 180
    retrain_every_bars: int = 24 * 7
    horizon_bars: int = 6
    starting_capital: float = 100_000.0
    risk_per_trade: float = 0.005
    max_position_pct: float = 0.25
    max_drawdown_pct: float = 0.15
    max_daily_loss_pct: float = 0.03
    max_trades_per_day: int = 2
    stop_atr_multiple: float = 2.0
    target_atr_multiple: float = 3.0
    fee_bps: float = 5.0
    slippage_bps: float = 2.0
    min_edge_after_costs: float = 0.0015
    probability_long: float = 0.56
    probability_short: float = 0.44
    long_only: bool = True
    model: str = "hist_gradient_boosting"
    seed: int = 42

CORE_FEATURES = [
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_24", "ret_72", "ret_168",
    "vol_6", "vol_24", "vol_72", "atr_pct", "range_pct", "body_pct",
    "rsi14", "macd", "macd_signal", "macd_hist", "sma24_ratio", "sma72_ratio",
    "ema24_ratio", "ema168_ratio", "bb_pos", "bb_width", "volume_z", "volume_ratio",
    "trend_24_168", "drawdown_168", "drawdown_720"
]

def _request(url: str, params: dict) -> list | dict:
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def binance_klines(symbol: str, interval: str, limit: int = 1000, start: int | None = None, end: int | None = None) -> pd.DataFrame:
    p = {"symbol": symbol, "interval": interval, "limit": min(limit, 1000)}
    if start is not None: p["startTime"] = start
    if end is not None: p["endTime"] = end
    rows = _request("https://api.binance.com/api/v3/klines", p)
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    d = pd.DataFrame(rows, columns=cols)
    if d.empty: return d
    d["Date"] = pd.to_datetime(d.open_time, unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})[["Date","Open","High","Low","Close","Volume","quote_volume","trades","taker_buy_base","taker_buy_quote"]]

def download_history(cfg: Config, years: int | None = None) -> pd.DataFrame:
    years = years or cfg.history_years
    end = int(time.time() * 1000)
    start = int((pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=365.25 * years)).timestamp() * 1000)
    chunks = []
    cursor = start
    while cursor < end:
        d = binance_klines(cfg.symbol, cfg.interval, 1000, cursor, end)
        if d.empty: break
        chunks.append(d)
        last = int(d.Date.iloc[-1].timestamp() * 1000)
        if last <= cursor: break
        cursor = last + 1
    if not chunks: raise RuntimeError("Binance returned no candles")
    d = pd.concat(chunks, ignore_index=True).drop_duplicates("Date").sort_values("Date").reset_index(drop=True)
    validate_ohlcv(d)
    d.to_csv(OUT / "binance_btcusdt_history.csv", index=False)
    LOG.info("Downloaded %d %s bars: %s -> %s", len(d), cfg.interval, d.Date.iloc[0], d.Date.iloc[-1])
    return d

def download_recent(cfg: Config, bars: int) -> pd.DataFrame:
    chunks, end = [], int(time.time() * 1000)
    cursor_end = end
    while sum(len(x) for x in chunks) < bars:
        d = binance_klines(cfg.symbol, cfg.interval, 1000, None, cursor_end)
        if d.empty: break
        chunks.insert(0, d)
        first = int(d.Date.iloc[0].timestamp() * 1000)
        cursor_end = first - 1
        if len(d) < 1000: break
    out = pd.concat(chunks, ignore_index=True).drop_duplicates("Date").sort_values("Date").tail(bars).reset_index(drop=True)
    validate_ohlcv(out)
    return out

def validate_ohlcv(d: pd.DataFrame) -> None:
    required = {"Date","Open","High","Low","Close","Volume"}
    if required - set(d.columns): raise ValueError(f"Missing columns: {required-set(d.columns)}")
    if d.Date.duplicated().any(): raise ValueError("Duplicate timestamps")
    if d[["Open","High","Low","Close","Volume"]].isna().any().any(): raise ValueError("NaN market data")
    if (d[["Open","High","Low","Close"]] <= 0).any().any() or (d.Volume < 0).any(): raise ValueError("Invalid OHLCV")
    if (d.High < d[["Open","Close"]].max(axis=1)).any() or (d.Low > d[["Open","Close"]].min(axis=1)).any(): raise ValueError("Invalid OHLC relationship")

def add_market_context(d: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    d = d.sort_values("Date").copy()
    try:
        oi = _request("https://fapi.binance.com/futures/data/openInterestHist", {"symbol":cfg.symbol,"period":"1h","limit":500})
        o = pd.DataFrame(oi)
        if not o.empty:
            o["Date"] = pd.to_datetime(o.timestamp, unit="ms", utc=True)
            o["open_interest"] = pd.to_numeric(o.sumOpenInterest, errors="coerce")
            o["oi_value"] = pd.to_numeric(o.sumOpenInterestValue, errors="coerce")
            d = pd.merge_asof(d, o[["Date","open_interest","oi_value"]].sort_values("Date"), on="Date", direction="backward")
    except Exception as e: LOG.warning("OI unavailable: %s", e)
    try:
        fr = _request("https://fapi.binance.com/fapi/v1/fundingRate", {"symbol":cfg.symbol,"limit":1000})
        f = pd.DataFrame(fr)
        if not f.empty:
            f["Date"] = pd.to_datetime(f.fundingTime, unit="ms", utc=True)
            f["funding_rate"] = pd.to_numeric(f.fundingRate, errors="coerce")
            d = pd.merge_asof(d, f[["Date","funding_rate"]].sort_values("Date"), on="Date", direction="backward")
    except Exception as e: LOG.warning("Funding unavailable: %s", e)
    return d

def _add_indicators(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy(); c,h,l,v,o = d.Close,d.High,d.Low,d.Volume,d.Open; r=c.pct_change()
    for n in [1,3,6,12,24,72,168]: d[f"ret_{n}"] = c.pct_change(n)
    for n in [6,24,72]: d[f"vol_{n}"] = r.rolling(n).std()
    tr=pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1); d["atr_pct"]=tr.rolling(14).mean()/c; d["range_pct"]=(h-l)/c; d["body_pct"]=(c-o)/o
    delta=c.diff(); gain=delta.clip(lower=0).rolling(14).mean(); loss=(-delta.clip(upper=0)).rolling(14).mean(); d["rsi14"]=100-100/(1+gain/loss.replace(0,np.nan))
    e12,e26=c.ewm(span=12,adjust=False).mean(),c.ewm(span=26,adjust=False).mean(); d["macd"]=e12-e26; d["macd_signal"]=d.macd.ewm(span=9,adjust=False).mean(); d["macd_hist"]=d.macd-d.macd_signal
    for n in [24,72]: d[f"sma{n}_ratio"]=c/c.rolling(n).mean()-1
    for n in [24,168]: d[f"ema{n}_ratio"]=c/c.ewm(span=n,adjust=False).mean()-1
    mid,sd=c.rolling(24).mean(),c.rolling(24).std(); d["bb_pos"]=(c-(mid-2*sd))/(4*sd).replace(0,np.nan); d["bb_width"]=4*sd/mid
    d["volume_z"]=(v-v.rolling(72).mean())/v.rolling(72).std(); d["volume_ratio"]=v/v.rolling(24).mean(); d["trend_24_168"]=c.rolling(24).mean()/c.rolling(168).mean()-1; d["drawdown_168"]=c/c.rolling(168).max()-1; d["drawdown_720"]=c/c.rolling(720).max()-1
    return d

def make_features(raw: pd.DataFrame, cfg: Config, with_context: bool = False, labeled: bool = True) -> pd.DataFrame:
    d=raw.copy()
    if with_context: d=add_market_context(d,cfg)
    d=_add_indicators(d)
    if labeled:
        d["future_return"]=d.Close.shift(-cfg.horizon_bars)/d.Close-1; d["label"]=(d.future_return>0).astype(int); d=d.replace([np.inf,-np.inf],np.nan).dropna(subset=CORE_FEATURES+["future_return","label"]).reset_index(drop=True)
    else:
        d=d.replace([np.inf,-np.inf],np.nan).dropna(subset=CORE_FEATURES).reset_index(drop=True)
    return d

def model_pipeline(cfg: Config):
    if cfg.model=="random_forest":
        m=RandomForestClassifier(n_estimators=400,min_samples_leaf=12,max_features="sqrt",class_weight="balanced",random_state=cfg.seed,n_jobs=-1); return Pipeline([("imputer",SimpleImputer(strategy="median")),("model",m)])
    if cfg.model=="logistic": return Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",LogisticRegression(C=.5,max_iter=3000,random_state=cfg.seed))])
    m=HistGradientBoostingClassifier(max_iter=250,learning_rate=.04,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=1.0,random_state=cfg.seed); return Pipeline([("imputer",SimpleImputer(strategy="median")),("model",m)])

def walk_forward(d: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out=d.copy(); out["prob_up"]=np.nan; out["model_version"]=-1; version=0
    for start in range(cfg.initial_train_bars,len(out),cfg.retrain_every_bars):
        end=min(start+cfg.retrain_every_bars,len(out)); train=out.iloc[:start]; test=out.iloc[start:end]
        if train.label.nunique()<2: continue
        m=model_pipeline(cfg); m.fit(train[CORE_FEATURES],train.label); out.loc[test.index,"prob_up"]=m.predict_proba(test[CORE_FEATURES])[:,1]; out.loc[test.index,"model_version"]=version; version+=1
    return out.dropna(subset=["prob_up"]).reset_index(drop=True)

def backtest(d: pd.DataFrame, cfg: Config):
    cash=cfg.starting_capital; peak=cash; trades=[]; equity=[]; day0=None; halted=False; trades_today=0
    for i in range(len(d)-1):
        row,nxt=d.iloc[i],d.iloc[i+1]
        if row.Date.date()!=day0: day0=row.Date.date(); daily_anchor=cash; halted=False; trades_today=0
        dd=cash/max(peak,1)-1
        if dd<=-cfg.max_drawdown_pct or (daily_anchor-cash)/max(daily_anchor,1)>=cfg.max_daily_loss_pct: halted=True
        p=float(row.prob_up); expected=float(row.future_return); cost=2*(cfg.fee_bps+cfg.slippage_bps)/10000+cfg.min_edge_after_costs
        if halted or trades_today>=cfg.max_trades_per_day or p<cfg.probability_long or expected<=cost:
            equity.append((row.Date,cash,cash/max(peak,1)-1)); continue
        entry_raw=float(nxt.Open); entry=entry_raw*(1+cfg.slippage_bps/10000); atr=float(row.atr_pct)*entry; stop=entry-cfg.stop_atr_multiple*atr; target=entry+cfg.target_atr_multiple*atr; qty=min(cash*cfg.risk_per_trade/max(entry-stop,entry*1e-6),cash*cfg.max_position_pct/entry)
        if qty<=0: continue
        hi,lo=float(nxt.High),float(nxt.Low)
        if lo<=stop: exit_raw,reason=stop,"stop"
        elif hi>=target: exit_raw,reason=target,"target"
        else: exit_raw,reason=float(nxt.Close),"horizon"
        exit_px=exit_raw*(1-cfg.slippage_bps/10000); gross=qty*(exit_px-entry); fees=(qty*entry+qty*exit_px)*cfg.fee_bps/10000; net=gross-fees; cash+=net; peak=max(peak,cash); trades_today+=1
        trades.append({"signal_time":row.Date,"entry_time":nxt.Date,"prob_up":p,"expected_return":expected,"entry":entry,"stop":stop,"target":target,"exit":exit_px,"qty":qty,"gross_pnl":gross,"fees":fees,"net_pnl":net,"exit_reason":reason,"capital_after":cash})
        equity.append((nxt.Date,cash,cash/max(peak,1)-1))
    eq=pd.DataFrame(equity,columns=["Date","capital","drawdown"]).drop_duplicates("Date").sort_values("Date"); tr=pd.DataFrame(trades); return eq,tr

def summarize(eq,tr,cfg):
    final=float(eq.capital.iloc[-1]); daily=eq.set_index("Date").capital.resample("1D").last().ffill().pct_change().dropna(); vol=daily.std()*math.sqrt(365) if len(daily)>1 else np.nan; sharpe=daily.mean()/daily.std()*math.sqrt(365) if daily.std()>0 else np.nan; down=daily[daily<0].std(); sortino=daily.mean()/down*math.sqrt(365) if down and down>0 else np.nan; wins=int((tr.net_pnl>0).sum()) if len(tr) else 0; gp=float(tr.loc[tr.net_pnl>0,"net_pnl"].sum()) if wins else 0; gl=float(-tr.loc[tr.net_pnl<=0,"net_pnl"].sum()) if len(tr)-wins else 0
    return {"starting_capital":cfg.starting_capital,"final_capital":final,"return_pct":(final/cfg.starting_capital-1)*100,"max_drawdown_pct":float(eq.drawdown.min()*100),"annualized_volatility_pct":float(vol*100) if pd.notna(vol) else None,"sharpe":float(sharpe) if pd.notna(sharpe) else None,"sortino":float(sortino) if pd.notna(sortino) else None,"trades":len(tr),"wins":wins,"losses":len(tr)-wins,"win_rate_pct":wins/len(tr)*100 if len(tr) else None,"profit_factor":gp/gl if gl else None,"expectancy":float(tr.net_pnl.mean()) if len(tr) else None}

def latest_signal(cfg: Config):
    """Train on enough fully labeled history and score the latest CLOSED candle."""
    # 720 bars are needed for the longest rolling feature (drawdown_720), plus
    # the prediction horizon and a small safety margin. The previous version only
    # requested initial_train_bars, which left fewer usable labeled rows.
    feature_warmup = 720
    bars = cfg.initial_train_bars + feature_warmup + cfg.horizon_bars + 20
    raw = download_recent(cfg, bars)
    # Binance may return the currently-forming candle. Never score that candle.
    if len(raw) < 2:
        raise RuntimeError("Not enough market data")
    closed = raw.iloc[:-1].copy()
    train = make_features(closed, cfg, with_context=True, labeled=True)
    current = make_features(closed, cfg, with_context=True, labeled=False).iloc[[-1]]
    if len(train) < cfg.initial_train_bars or train.label.nunique() < 2:
        raise RuntimeError(f"Insufficient labeled history: got {len(train)} usable bars, need {cfg.initial_train_bars}")
    m=model_pipeline(cfg); m.fit(train[CORE_FEATURES],train.label); p=float(m.predict_proba(current[CORE_FEATURES])[:,1][0]); price=float(current.Close.iloc[0]); atr=float(current.atr_pct.iloc[0]); signal="HOLD" if p<cfg.probability_long else "LONG"
    context={}
    if "funding_rate" in current: context["funding_rate"]=float(current.funding_rate.iloc[0]) if pd.notna(current.funding_rate.iloc[0]) else None
    if "open_interest" in current: context["open_interest"]=float(current.open_interest.iloc[0]) if pd.notna(current.open_interest.iloc[0]) else None
    return {"timestamp":current.Date.iloc[0].isoformat(),"symbol":cfg.symbol,"interval":cfg.interval,"price":price,"probability_up":p,"atr_pct":atr,"signal":signal,"context":context,"model":cfg.model,"max_trades_per_day":cfg.max_trades_per_day}

def main():
    p=argparse.ArgumentParser(description="Research-grade Bitcoin ML trading system"); p.add_argument("--mode",choices=["backtest","signal","data"],default="backtest"); p.add_argument("--model",choices=["hist_gradient_boosting","random_forest","logistic"],default="hist_gradient_boosting"); p.add_argument("--interval",choices=["1m","5m","15m","1h","4h","1d"],default="1h"); args=p.parse_args(); cfg=Config(model=args.model,interval=args.interval)
    if args.mode=="data": download_history(cfg)
    elif args.mode=="signal":
        s=latest_signal(cfg); (OUT/"latest_signal.json").write_text(json.dumps(s,indent=2)); print(json.dumps(s,indent=2))
    else:
        raw=download_history(cfg); feat=make_features(raw,cfg); pred=walk_forward(feat,cfg); eq,tr=backtest(pred,cfg); feat.to_csv(OUT/"features.csv",index=False); pred.to_csv(OUT/"predictions.csv",index=False); eq.to_csv(OUT/"equity_curve.csv",index=False); tr.to_csv(OUT/"trades.csv",index=False); report={"strategy":summarize(eq,tr,cfg),"data_start":str(raw.Date.iloc[0]),"data_end":str(raw.Date.iloc[-1]),"bars":len(raw),"config":asdict(cfg)}; (OUT/"performance_summary.json").write_text(json.dumps(report,indent=2,default=str)); print(json.dumps(report,indent=2))

if __name__=="__main__": main()
