"""Hard safety gate before any future live execution integration.

This module only evaluates readiness. It never places an order. It requires
forward paper evidence and clean validation artifacts before returning ready.
"""
from __future__ import annotations
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent/"outputs"

REQUIRED={"final_model_report.json","validation_report.json"}

def check()->dict:
    result={"ready":False,"checks":{},"reasons":[]}
    for name in REQUIRED:
        ok=(OUT/name).exists(); result["checks"][name]=ok
        if not ok: result["reasons"].append(f"missing {name}")
    state=OUT/"paper_state.json"
    if state.exists():
        s=json.loads(state.read_text()); result["checks"]["paper_state"] = True
        if s.get("trades",0)<100: result["reasons"].append("fewer than 100 paper trades")
    else:
        result["checks"]["paper_state"]=False; result["reasons"].append("no paper trading state")
    # This project deliberately has no live-order implementation. The gate can
    # only become informationally ready; an independent execution review remains required.
    result["reasons"].append("live execution adapter is intentionally disabled")
    result["ready"]=False
    (OUT/"live_readiness.json").write_text(json.dumps(result,indent=2))
    return result

if __name__=="__main__": print(json.dumps(check(),indent=2))
