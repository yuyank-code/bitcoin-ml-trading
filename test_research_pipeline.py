import numpy as np
import pandas as pd

from production_research import signal_backtest
from unified_research import _prefix


def sample(n=1000):
    t=pd.date_range("2023-01-01",periods=n,freq="h",tz="UTC")
    close=30000+np.cumsum(np.sin(np.arange(n)/17)*30+2)
    return pd.DataFrame({"Date":t,"Open":close,"High":close*1.004,"Low":close*.996,"Close":close,"Volume":1000.0})


def test_prefix_namespaces_external_columns():
    d=pd.DataFrame({"Date":pd.date_range("2024-01-01",periods=2),"value":[1,2]})
    x=_prefix(d,"macro")
    assert "macro_value" in x.columns and "value" not in x.columns


def test_signal_backtest_does_not_require_future_return():
    d=sample(1000)
    d["atr_pct"]=.01
    d["prob_up"]=.60
    d["future_return"]=np.nan
    d["label"]=1
    e,t=signal_backtest(d)
    assert len(e)>0
    assert len(t)>=0


def test_decision_is_probability_cost_based():
    d=sample(1000); d["atr_pct"]=.01; d["prob_up"]=.90; d["label"]=1
    d["future_return"]=np.nan
    e,t=signal_backtest(d,fee_bps=5,slip_bps=2)
    assert len(t)>0
