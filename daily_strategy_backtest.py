"""Validate the high-conviction daily strategy without optimizing on OOS data."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent; OUT=ROOT/"outputs"; OUT.mkdir(exist_ok=True)

def evaluate(df:pd.DataFrame, fee_bps:float=10, slippage_bps:float=5, initial:float=1.0)->dict:
    d=df.sort_values("Date").copy(); d["day"]=pd.to_datetime(d.Date,utc=True).dt.floor("D")
    if "future_return" not in d: raise ValueError("future_return required for evaluation")
    cost=(fee_bps+slippage_bps)*2/10000
    d["gross_return"]=np.where(d.signal=="LONG",d.future_return,np.where(d.signal=="SHORT",-d.future_return,0.0))
    d["net_return"]=np.where(d.signal=="HOLD",0,d.gross_return-cost)
    # Multiple intraday entries are evaluated independently in research; a live
    # implementation must apply portfolio/risk limits before each entry.
    equity=(1+d.net_return).cumprod()*initial
    d["equity"]=equity
    rets=d.net_return
    total=float(equity.iloc[-1]/initial-1) if len(equity) else 0
    std=float(rets.std(ddof=1)) if len(rets)>1 else 0
    sharpe=float(rets.mean()/std*np.sqrt(max(1,d.day.nunique()))) if std>0 else 0
    downside=rets[rets<0].std(ddof=1) if len(rets[rets<0])>1 else 0
    sortino=float(rets.mean()/downside*np.sqrt(max(1,d.day.nunique()))) if downside and downside>0 else 0
    dd=(equity/equity.cummax()-1).min() if len(equity) else 0
    trades=d[d.signal!="HOLD"]
    wins=(trades.net_return>0).sum(); losses=(trades.net_return<0).sum()
    gross_win=trades.loc[trades.net_return>0,"net_return"].sum(); gross_loss=-trades.loc[trades.net_return<0,"net_return"].sum()
    return {"rows":len(d),"trades":len(trades),"trade_days":int(trades.day.nunique()),
            "max_trades_per_day":2,"win_rate":float(wins/len(trades)) if len(trades) else 0,
            "profit_factor":float(gross_win/gross_loss) if gross_loss>0 else None,
            "total_return":total,"sharpe":sharpe,"sortino":sortino,
            "max_drawdown":float(dd),"avg_trade":float(trades.net_return.mean()) if len(trades) else 0,
            "fee_bps_per_side":fee_bps,"slippage_bps_per_side":slippage_bps}

def select_two(d:pd.DataFrame, threshold:float)->pd.DataFrame:
    x=d.copy(); x["signal"]="HOLD"; x["day"]=x.Date.dt.floor("D")
    eligible=x[(x.conviction>=threshold)&(x.available_groups>=4)&(x.direction!="HOLD")].copy()
    for day, idx in eligible.groupby("day").groups.items():
        c=eligible.loc[idx].sort_values(["conviction","model_confidence","model_uncertainty"],ascending=[False,False,True])
        if len(c):
            x.loc[c.index[0],"signal"]=c.iloc[0].direction
        if len(c)>1:
            best=float(c.iloc[0].conviction); second=c.iloc[1]
            close_enough=float(second.conviction)>=max(78.0,best*0.92)
            different_direction=second.direction!=c.iloc[0].direction
            exceptionally_strong_same_side=float(second.conviction)>=85.0
            if close_enough and (different_direction or exceptionally_strong_same_side):
                x.loc[second.name,"signal"]=second.direction
    return x

def run():
    src=OUT/"daily_conviction_candidates.csv"
    if not src.exists(): raise SystemExit("Run daily_conviction.py first")
    d=pd.read_csv(src,parse_dates=["Date"])
    reports={}
    for threshold in (65,70,75,80,85,90):
        x=select_two(d,threshold)
        reports[str(threshold)]=evaluate(x)
    (OUT/"daily_strategy_sensitivity.json").write_text(json.dumps(reports,indent=2))
    print(json.dumps(reports,indent=2))
if __name__=="__main__": run()
