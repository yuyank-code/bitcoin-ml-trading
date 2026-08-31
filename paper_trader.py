"""Paper portfolio engine. No exchange credentials and no real orders.

Consumes externally generated signals and simulates fills with fees/slippage.
It persists state so a process restart does not reset the virtual account.
"""
from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass, asdict

import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"outputs"; OUT.mkdir(exist_ok=True)
STATE=OUT/"paper_state.json"

@dataclass
class PaperConfig:
    starting_cash: float=100000.0
    max_position_pct: float=0.25
    fee_bps: float=5.0
    slippage_bps: float=2.0


def load_state(cfg: PaperConfig):
    if STATE.exists(): return json.loads(STATE.read_text())
    s={"cash":cfg.starting_cash,"btc":0.0,"last_price":None,"equity":cfg.starting_cash,"trades":0,"fees":0.0,"created":""}
    STATE.write_text(json.dumps(s,indent=2)); return s


def mark(state:dict, price:float):
    state["last_price"]=float(price); state["equity"]=float(state["cash"]+state["btc"]*price); return state


def apply_signal(signal:str, price:float, cfg:PaperConfig, state:dict, timestamp:str=""):
    signal=str(signal).upper(); price=float(price)
    if signal not in {"LONG","HOLD","EXIT"}: signal="HOLD"
    fee=cfg.fee_bps/10000; slip=cfg.slippage_bps/10000
    max_value=state["equity"]*cfg.max_position_pct
    if signal=="LONG" and state["btc"]<=0 and state["cash"]>0:
        fill=price*(1+slip); notional=min(state["cash"],max_value); qty=notional/fill; cost=notional*fee
        state["cash"]-=notional+cost; state["btc"]+=qty; state["fees"]+=cost; state["trades"]+=1
        event={"timestamp":timestamp,"action":"BUY","price":fill,"qty":qty,"fee":cost}
    elif signal=="EXIT" and state["btc"]>0:
        fill=price*(1-slip); notional=state["btc"]*fill; cost=notional*fee
        state["cash"]+=notional-cost; state["fees"]+=cost; state["btc"]=0.0; state["trades"]+=1
        event={"timestamp":timestamp,"action":"SELL","price":fill,"qty":notional/fill,"fee":cost}
    else:
        event={"timestamp":timestamp,"action":"HOLD","price":price,"qty":state["btc"],"fee":0.0}
    mark(state,price)
    events=OUT/"paper_trades.jsonl"; events.open("a").write(json.dumps(event)+"\n"); STATE.write_text(json.dumps(state,indent=2)); return state,event

if __name__=="__main__":
    cfg=PaperConfig(); state=load_state(cfg); print(json.dumps(state,indent=2))
