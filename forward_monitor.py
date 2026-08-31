"""Continuous forward monitor using Binance public data and the paper portfolio.

Paper-only: it records signals and virtual fills and never creates signed
exchange orders.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
from datetime import datetime,timezone
from trading_bot import Config,latest_signal
from paper_trader import PaperConfig,load_state,apply_signal

OUT=Path(__file__).resolve().parent/"outputs"; OUT.mkdir(exist_ok=True)

def run(interval_seconds:int=900,once:bool=False):
    cfg=Config(interval="1h"); pcfg=PaperConfig(); state=load_state(pcfg)
    while True:
        try:
            sig=latest_signal(cfg)
            action=sig["signal"] if sig["signal"] in {"LONG","EXIT"} else "HOLD"
            state,event=apply_signal(action,sig["price"],pcfg,state,sig["timestamp"])
            record={"observed_at":datetime.now(timezone.utc).isoformat(),"signal":sig,"paper_event":event,"paper_equity":state["equity"]}
            with (OUT/"forward_monitor.jsonl").open("a") as f: f.write(json.dumps(record,default=str)+"\n")
            print(json.dumps(record,indent=2,default=str))
        except Exception as exc:
            print(json.dumps({"error":str(exc),"observed_at":datetime.now(timezone.utc).isoformat()}))
        if once: break
        time.sleep(max(60,int(interval_seconds)))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--once",action="store_true"); p.add_argument("--interval-seconds",type=int,default=900); a=p.parse_args(); run(a.interval_seconds,a.once)
