"""Guarded self-learning layer for the BTC research/paper system.

The system records predictions, waits for outcomes to become known, diagnoses
errors by regime/signal group/confidence, and proposes parameter/model updates.
It NEVER changes the live strategy automatically: adaptations are validated on
an untouched validation window and only promoted by an explicit deployment gate.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent; OUT=ROOT/"outputs"; OUT.mkdir(exist_ok=True)
MEMORY=OUT/"learning_memory.json"


def diagnose(predictions: pd.DataFrame, horizon_col="future_return") -> dict:
    d=predictions.copy()
    if "prob_up" not in d.columns or horizon_col not in d.columns:
        raise ValueError("prob_up and future_return are required")
    d["actual_up"]=(d[horizon_col]>0).astype(int)
    d["correct"]=((d.prob_up>=.5).astype(int)==d.actual_up)
    d["confidence"]=(d.prob_up-.5).abs()*2
    d["error"]=(d.prob_up-d.actual_up).abs()
    report={
        "rows":int(len(d)),
        "accuracy":float(d.correct.mean()) if len(d) else 0,
        "brier":float(((d.prob_up-d.actual_up)**2).mean()) if len(d) else 0,
        "high_conf_accuracy":float(d.loc[d.confidence>=.5,"correct"].mean()) if (d.confidence>=.5).any() else None,
        "overconfidence_rate":float(((d.confidence>=.6)&(~d.correct)).mean()) if len(d) else 0,
    }
    for col in ("regime","signal","direction","available_groups"):
        if col in d:
            grouped=d.groupby(col).agg(rows=("correct","size"),accuracy=("correct","mean"),mean_error=("error","mean"))
            report[f"by_{col}"]=grouped.reset_index().to_dict(orient="records")
    return report


def update_memory(report:dict)->None:
    old=json.loads(MEMORY.read_text()) if MEMORY.exists() else {"runs":[]}
    old["runs"].append(report)
    old["runs"]=old["runs"][-100:]
    MEMORY.write_text(json.dumps(old,indent=2,default=str))


def propose_adaptation(report:dict)->dict:
    """Turn observed errors into bounded proposals, not uncontrolled self-editing."""
    proposals=[]
    if report.get("overconfidence_rate",0)>0.20:
        proposals.append({"type":"calibration","action":"increase_probability_calibration_strength","reason":"high-confidence errors are elevated"})
    if report.get("high_conf_accuracy") is not None and report["high_conf_accuracy"]<0.55:
        proposals.append({"type":"threshold","action":"raise_trade_threshold","reason":"high-confidence subset lacks sufficient directional edge"})
    if report.get("accuracy",0)<0.50:
        proposals.append({"type":"model","action":"retrain_and_retest","reason":"aggregate directional accuracy below baseline"})
    return {"proposals":proposals,"requires_validation":True,"auto_deploy":False}


def analyze_file(path: str)->dict:
    d=pd.read_csv(path,parse_dates=["Date"])
    report=diagnose(d)
    update_memory(report)
    proposal=propose_adaptation(report)
    out={"diagnosis":report,"proposal":proposal}
    (OUT/"latest_learning_review.json").write_text(json.dumps(out,indent=2,default=str))
    return out

if __name__=="__main__":
    src=OUT/"unified_predictions.csv"
    if not src.exists(): raise SystemExit("Run the prediction engine first")
    print(json.dumps(analyze_file(str(src)),indent=2,default=str))
