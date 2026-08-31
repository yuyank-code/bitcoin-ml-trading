"""Leakage-safe final research engine.

This is the reference path for model selection. It deliberately does NOT use
future_return as an entry feature or trading decision. Future returns are used
only after the signal timestamp for evaluation.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parent; OUT=ROOT/"outputs"; OUT.mkdir(exist_ok=True)
TECH=["ret_1","ret_3","ret_6","ret_12","ret_24","ret_72","ret_168","vol_6","vol_24","vol_72","atr_pct","range_pct","body_pct","rsi14","macd","macd_signal","macd_hist","sma24_ratio","sma72_ratio","ema24_ratio","ema168_ratio","bb_pos","bb_width","volume_z","volume_ratio","trend_24_168","drawdown_168","drawdown_720"]
GROUP_PREFIX={"onchain":("oc_","onchain_"),"derivatives":("oi_","funding_","global_","top_","taker_","basis_"),"macro":("macro_",),"sentiment":("fear_greed","news_","sentiment_"),"breadth":("breadth_","etf_","trends_","dominance_")}


def features(df,groups):
    cols=[c for c in TECH if c in df]
    for g in groups:
        for c in df.columns:
            if c not in cols and any(c.startswith(p) for p in GROUP_PREFIX[g]): cols.append(c)
    return cols


def model(kind,seed=42):
    if kind=="logistic": return Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler()),("m",LogisticRegression(C=.5,max_iter=3000,random_state=seed))])
    if kind=="rf": return Pipeline([("imp",SimpleImputer(strategy="median")),("m",RandomForestClassifier(n_estimators=300,min_samples_leaf=12,max_features="sqrt",class_weight="balanced",n_jobs=-1,random_state=seed))])
    return Pipeline([("imp",SimpleImputer(strategy="median")),("m",HistGradientBoostingClassifier(max_iter=250,learning_rate=.04,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=1.,random_state=seed))])


def walk_forward(d,cols,initial=24*180,step=24*7,seed=42):
    x=d.sort_values("Date").reset_index(drop=True).copy(); x["prob_up"]=np.nan; x["model_disagreement"]=np.nan
    for start in range(initial,len(x),step):
        end=min(start+step,len(x)); tr=x.iloc[:start]; te=x.iloc[start:end]
        if tr.label.nunique()<2: continue
        ps=[]
        for k in ("logistic","rf","hgb"):
            m=model(k,seed); m.fit(tr[cols],tr.label); ps.append(m.predict_proba(te[cols])[:,1])
        a=np.vstack(ps); x.loc[te.index,"prob_up"]=a.mean(0); x.loc[te.index,"model_disagreement"]=a.std(0)
    return x.dropna(subset=["prob_up"]).reset_index(drop=True)


def signal_backtest(d,fee_bps=5.,slip_bps=2.,risk=.005,max_position=.25,prob=0.56,stop_atr=2.,target_atr=3.,initial_cash=100000.):
    cash=initial_cash; peak=cash; eq=[]; trades=[]; last_day=None; day_start=cash; halted=False
    for i in range(len(d)-1):
        r,n=d.iloc[i],d.iloc[i+1]
        if r.Date.date()!=last_day: last_day=r.Date.date(); day_start=cash; halted=False
        dd=cash/max(peak,1)-1
        if dd<=-.15 or (day_start-cash)/max(day_start,1)>=.03: halted=True
        p=float(r.prob_up); atr=float(r.atr_pct)
        entry=float(n.Open)*(1+slip_bps/10000); stop_dist=max(entry*atr*stop_atr,entry*.002); target_dist=entry*atr*target_atr
        # Expected value is calculated from the model probability and known-at-signal
        # stop/target distances. It never references the future realized return.
        expected_gross=p*target_dist-(1-p)*stop_dist
        roundtrip_cost=entry*2*(fee_bps+slip_bps)/10000
        if halted or p<prob or expected_gross<=roundtrip_cost+entry*.0015: eq.append((r.Date,cash)); continue
        qty=min(cash*risk/stop_dist,cash*max_position/entry)
        hi,lo=float(n.High),float(n.Low); stop=entry-stop_dist; target=entry+target_dist
        if lo<=stop: exit_px=stop; reason="stop"
        elif hi>=target: exit_px=target; reason="target"
        else: exit_px=float(n.Close); reason="horizon"
        exit_px*=1-slip_bps/10000; gross=qty*(exit_px-entry); fees=(qty*entry+qty*exit_px)*fee_bps/10000; net=gross-fees; cash+=net; peak=max(peak,cash)
        trades.append({"signal_time":r.Date,"entry_time":n.Date,"prob_up":p,"expected_gross":expected_gross,"entry":entry,"stop":stop,"target":target,"exit":exit_px,"qty":qty,"gross_pnl":gross,"fees":fees,"net_pnl":net,"exit_reason":reason,"capital_after":cash})
        eq.append((n.Date,cash))
    e=pd.DataFrame(eq,columns=["Date","capital"]).drop_duplicates("Date").sort_values("Date"); t=pd.DataFrame(trades); return e,t


def metrics(e,t):
    if e.empty:return {"trades":0}
    ret=e.capital.iloc[-1]/e.capital.iloc[0]-1; peak=e.capital.cummax(); dd=e.capital/peak-1; days=max((e.Date.iloc[-1]-e.Date.iloc[0]).days,1); years=days/365.25
    cagr=(e.capital.iloc[-1]/e.capital.iloc[0])**(1/years)-1
    daily=e.set_index("Date").capital.resample("1D").last().ffill().pct_change().dropna(); sd=daily.std(); down=daily[daily<0].std()
    wins=(t.net_pnl>0).sum() if len(t) else 0; losses=(t.net_pnl<=0).sum() if len(t) else 0
    gp=t.loc[t.net_pnl>0,"net_pnl"].sum() if wins else 0; gl=-t.loc[t.net_pnl<=0,"net_pnl"].sum() if losses else 0
    return {"start":float(e.capital.iloc[0]),"final":float(e.capital.iloc[-1]),"return_pct":float(ret*100),"cagr_pct":float(cagr*100),"max_drawdown_pct":float(dd.min()*100),"sharpe":float(daily.mean()/sd*np.sqrt(365.25)) if sd>0 else None,"sortino":float(daily.mean()/down*np.sqrt(365.25)) if down>0 else None,"trades":int(len(t)),"win_rate_pct":float(wins/len(t)*100) if len(t) else None,"profit_factor":float(gp/gl) if gl else None,"expectancy":float(t.net_pnl.mean()) if len(t) else None}


def run(source=OUT/"unified_dataset.csv"):
    if not source.exists(): raise FileNotFoundError("Build outputs/unified_dataset.csv first")
    d=pd.read_csv(source,parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    d["future_return"]=d.Close.shift(-6)/d.Close-1; d["label"]=(d.future_return>0).astype(int)
    d=d.dropna(subset=["future_return"])
    experiments={"technical":[],"technical_onchain":["onchain"],"technical_derivatives":["derivatives"],"technical_macro":["macro"],"technical_sentiment":["sentiment"],"technical_breadth":["breadth"],"all_sources":list(GROUP_PREFIX)}
    report={}
    for name,groups in experiments.items():
        cols=features(d,groups)
        pred=walk_forward(d,cols)
        pred.to_csv(OUT/f"final_{name}_predictions.csv",index=False)
        if len(pred):
            auc=None
            y=pred.label.to_numpy(); p=pred.prob_up.to_numpy();
            if len(np.unique(y))==2:
                order=np.argsort(p); rank=np.empty_like(order); rank[order]=np.arange(len(p))+1; pos=y==1; neg=~pos; auc=float((rank[pos].sum()-pos.sum()*(pos.sum()+1)/2)/(pos.sum()*neg.sum()))
            report[name]={"features":cols,"rows":len(pred),"brier":float(np.mean((p-y)**2)),"accuracy":float(np.mean((p>=.5)==y)),"auc":auc,"trade_rate":float((p>=.56).mean())}
            eq,tr=signal_backtest(pred); report[name]["backtest"]=metrics(eq,tr); tr.to_csv(OUT/f"final_{name}_trades.csv",index=False); eq.to_csv(OUT/f"final_{name}_equity.csv",index=False)
    (OUT/"final_model_report.json").write_text(json.dumps(report,indent=2,default=str)); return report

if __name__=="__main__": print(json.dumps(run(),indent=2,default=str))
