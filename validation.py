"""Research validation utilities.

The validator is deliberately independent of the signal generator. It checks
walk-forward predictions, probability quality, benchmark performance, and cost
sensitivity. It never uses future returns to decide whether a trade should be
entered.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)


def classification_metrics(y: pd.Series, p: pd.Series) -> dict:
    y = pd.Series(y).astype(int).to_numpy(); p = np.clip(pd.Series(p).astype(float).to_numpy(), 1e-6, 1 - 1e-6)
    out = {"n": int(len(y)), "brier": float(np.mean((p-y)**2)), "log_loss": float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))), "accuracy": float(np.mean((p>=.5)==y))}
    if len(np.unique(y)) == 2:
        order = np.argsort(p); ranks = np.empty_like(order); ranks[order] = np.arange(len(p))+1
        pos = y == 1; neg = ~pos
        out["auc"] = float((ranks[pos].sum() - pos.sum()*(pos.sum()+1)/2) / (pos.sum()*neg.sum())) if pos.sum() and neg.sum() else None
    return out


def equity_metrics(equity: pd.Series, dates: pd.Series, periods_per_year: float = 365.25) -> dict:
    e = pd.Series(equity).astype(float).reset_index(drop=True); d = pd.to_datetime(dates, utc=True).reset_index(drop=True)
    if len(e) < 2: return {"n": int(len(e))}
    daily = pd.DataFrame({"Date": d, "equity": e}).set_index("Date").resample("1D").last().ffill().equity.pct_change().dropna()
    ret = float(e.iloc[-1]/e.iloc[0]-1); years=max((d.iloc[-1]-d.iloc[0]).total_seconds()/31557600, 1/365.25)
    cagr=(float(e.iloc[-1]/e.iloc[0])**(1/years)-1) if e.iloc[-1]>0 else -1
    peak=e.cummax(); dd=e/peak-1; maxdd=float(dd.min())
    sd=float(daily.std()) if len(daily)>1 else np.nan; downside=float(daily[daily<0].std()) if (daily<0).sum()>1 else np.nan
    sharpe=float(daily.mean()/sd*np.sqrt(periods_per_year)) if sd>0 else None
    sortino=float(daily.mean()/downside*np.sqrt(periods_per_year)) if downside and downside>0 else None
    return {"return_pct":ret*100,"cagr_pct":cagr*100,"max_drawdown_pct":maxdd*100,"sharpe":sharpe,"sortino":sortino,"days":int((d.iloc[-1]-d.iloc[0]).days)}


def trade_metrics(trades: pd.DataFrame) -> dict:
    if trades.empty: return {"trades":0}
    pnl=pd.to_numeric(trades.net_pnl, errors="coerce").dropna(); wins=pnl[pnl>0]; losses=pnl[pnl<=0]
    return {"trades":int(len(pnl)),"wins":int(len(wins)),"losses":int(len(losses)),"win_rate_pct":float(len(wins)/len(pnl)*100),"profit_factor":float(wins.sum()/-losses.sum()) if losses.sum()<0 else None,"expectancy":float(pnl.mean()),"median_trade":float(pnl.median()),"fees":float(pd.to_numeric(trades.get("fees",0),errors="coerce").sum()),"slippage":float(pd.to_numeric(trades.get("slippage_cost",0),errors="coerce").sum())}


def stress_costs(pred: pd.DataFrame, backtest_fn, fee_bps: float, slippage_bps: float, **backtest_kwargs) -> pd.DataFrame:
    """Run a deterministic cost grid against scalar-argument backtests.

    The authoritative signal_backtest accepts fee_bps/slip_bps as scalars.
    Keeping this adapter explicit prevents a stale config-object interface from
    silently turning the cost-stress test into an exception-only report.
    """
    rows=[]
    for mult in [0.5,1.0,1.5,2.0,3.0]:
        try:
            eq,tr=backtest_fn(pred, fee_bps=fee_bps*mult, slip_bps=slippage_bps*mult, **backtest_kwargs)
            rows.append({"cost_multiplier":mult,**equity_metrics(eq.capital,eq.Date),**trade_metrics(tr)})
        except Exception as exc:
            rows.append({"cost_multiplier":mult,"error":str(exc)})
    return pd.DataFrame(rows)


def validate_predictions(path: str | Path) -> dict:
    p=Path(path); d=pd.read_csv(p,parse_dates=["Date"])
    required={"Date","label","prob_up"}
    missing=required-set(d.columns)
    if missing: raise ValueError(f"Missing required columns: {sorted(missing)}")
    raw_duplicate_dates=int(d.Date.duplicated().sum())
    d=d.sort_values("Date")
    result={"file":str(p),"time_sorted_before_dedup":bool(d.Date.is_monotonic_increasing),"duplicate_dates":raw_duplicate_dates,"classification":classification_metrics(d.label,d.prob_up)}
    if raw_duplicate_dates:
        result["validation_error"]="duplicate_dates"
    d=d.drop_duplicates("Date").reset_index(drop=True)
    if "capital_after" in d.columns: result["equity"] = equity_metrics(d.capital_after,d.Date)
    if "signal" in d.columns: result["signal_counts"] = d.signal.value_counts(dropna=False).to_dict()
    return result


def run() -> None:
    files=list(OUT.glob("*predictions.csv")); reports={}
    for f in files:
        try: reports[f.name]=validate_predictions(f)
        except Exception as exc: reports[f.name]={"error":str(exc)}
    (OUT/"validation_report.json").write_text(json.dumps(reports,indent=2,default=str))
    print(json.dumps(reports,indent=2,default=str))

if __name__ == "__main__": run()
