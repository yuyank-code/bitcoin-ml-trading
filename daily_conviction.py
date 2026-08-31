"""High-conviction daily decision layer.

Ranks each eligible daily decision by independent signal-group agreement and
allows at most two new trades per UTC day. The second trade is admitted only
when it is independently strong and sufficiently close to the day's best setup.
This is research/paper-trading only.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"outputs"; OUT.mkdir(exist_ok=True)
GROUPS=("technical","onchain","derivatives","macro","sentiment","breadth")


def _score_probability(p: float) -> float:
    return float(np.clip((p-0.5)*200, -100, 100))


def build_conviction(df: pd.DataFrame) -> pd.DataFrame:
    d=df.sort_values("Date").copy()
    d["Date"]=pd.to_datetime(d["Date"],utc=True)
    score_cols=[]
    for g in GROUPS:
        pcol=f"{g}_prob_up"
        scol=f"{g}_score"
        if pcol in d:
            d[scol]=d[pcol].apply(lambda x: _score_probability(x) if pd.notna(x) else np.nan)
            score_cols.append(scol)
    if not score_cols:
        raise ValueError("Need at least one *_prob_up column")
    vals=d[score_cols]
    d["available_groups"]=vals.notna().sum(axis=1)
    d["agreement_score"]=vals.mean(axis=1).fillna(0)
    d["agreement_strength"]=vals.apply(lambda r: float(np.mean(np.abs(r.dropna())/100)) if r.notna().any() else 0,axis=1)
    d["direction_consistency"]=vals.apply(lambda r: float(abs(np.sign(r.dropna()).mean())) if r.notna().any() else 0,axis=1)
    d["model_confidence"]=((d.get("prob_up",0)-0.5).abs()*2).clip(0,1)
    d["model_uncertainty"]=d.get("prob_std",pd.Series(0,index=d.index)).fillna(0).clip(0,1)
    d["conviction"]=(0.45*(d["agreement_score"].abs()/100)+0.25*d["agreement_strength"]+
                      0.20*d["direction_consistency"]+0.10*d["model_confidence"]-
                      0.15*d["model_uncertainty"]).clip(0,1)*100
    d["direction"]=np.where(d["agreement_score"]>0,"LONG",np.where(d["agreement_score"]<0,"SHORT","HOLD"))
    return d


def select_daily_trades(df:pd.DataFrame, min_conviction:float=75, min_groups:int=4,
                        max_trades_per_day:int=2, second_trade_min_conviction:float=78,
                        second_trade_relative_to_best:float=0.92)->pd.DataFrame:
    """Select up to two setups/day.

    The second setup is allowed only if its conviction is both absolutely strong
    and at least a configurable fraction of the day's strongest eligible setup.
    This prevents a weak second trade from being admitted merely because the
    daily limit was increased.
    """
    if max_trades_per_day < 1:
        raise ValueError("max_trades_per_day must be >= 1")
    d=build_conviction(df)
    d["trade_candidate"]=(d.conviction>=min_conviction)&(d.available_groups>=min_groups)&(d.direction!="HOLD")
    d["day"]=d.Date.dt.floor("D")
    d["daily_trade"]=False

    for day, idx in d[d.trade_candidate].groupby("day").groups.items():
        candidates=d.loc[idx].sort_values(["conviction","model_confidence","model_uncertainty"],
                                           ascending=[False,False,True])
        chosen=[]
        if len(candidates):
            best=candidates.iloc[0]
            chosen.append(best.name)
            if max_trades_per_day>=2 and len(candidates)>1:
                second=candidates.iloc[1]
                # A second trade must be nearly as strong as the first and must
                # independently clear a slightly higher absolute conviction bar.
                close_enough=float(second.conviction)>=max(float(second_trade_min_conviction),
                                                          float(best.conviction)*second_trade_relative_to_best)
                # Avoid taking two same-direction entries in the same day when
                # they are merely duplicate bars of one setup.
                different_direction=second.direction != best.direction
                if close_enough and (different_direction or float(second.conviction)>=85):
                    chosen.append(second.name)
        d.loc[chosen,"daily_trade"]=True

    d["signal"]=np.where(d.daily_trade,d.direction,"HOLD")
    d.to_csv(OUT/"daily_conviction_candidates.csv",index=False)
    selected=d[d.daily_trade].copy()
    selected.to_csv(OUT/"daily_conviction_trades.csv",index=False)
    report={"candidate_rows":int(len(d)),"selected_trades":int(len(selected)),
            "trade_days":int(selected.day.nunique()),"max_trades_per_day":max_trades_per_day,
            "min_conviction":min_conviction,"min_groups":min_groups,
            "second_trade_min_conviction":second_trade_min_conviction,
            "second_trade_relative_to_best":second_trade_relative_to_best}
    (OUT/"daily_conviction_report.json").write_text(json.dumps(report,indent=2))
    return selected

if __name__=="__main__":
    src=OUT/"unified_predictions.csv"
    if not src.exists(): raise SystemExit("Run unified_engine.py first")
    df=pd.read_csv(src,parse_dates=["Date"])
    select_daily_trades(df, max_trades_per_day=2)
