import numpy as np
import pandas as pd

from nested_policy_research import _select_threshold, nested_walk_forward
from trading_bot import Config


def sample(n=700):
    t = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = 30000 + np.cumsum(np.sin(np.arange(n) / 17) * 30 + 2)
    d = pd.DataFrame({"Date": t, "Open": close, "High": close * 1.004,
                      "Low": close * .996, "Close": close, "Volume": 1000.0})
    d["atr_pct"] = .01
    d["label"] = (np.arange(n) % 3 != 0).astype(int)
    for c in ["ret_1"]:
        d[c] = d.Close.pct_change().fillna(0)
    return d


def test_threshold_selection_uses_only_calibration():
    cfg = Config(initial_train_bars=200, retrain_every_bars=100, horizon_bars=6)
    d = sample(400)
    d["prob_up"] = np.linspace(.50, .70, len(d))
    chosen = _select_threshold(d.iloc[:100], cfg)
    changed = d.iloc[:100].copy()
    # Altering observations outside calibration must not be able to change it.
    other = d.iloc[100:].copy()
    other["Close"] = other["Close"] * 100
    assert chosen == _select_threshold(d.iloc[:100], cfg)
    assert len(other) == 300


def test_nested_policy_has_frozen_threshold_per_outer_fold():
    cfg = Config(initial_train_bars=200, retrain_every_bars=100, horizon_bars=6)
    pred, policies = nested_walk_forward(d=sample(700), cfg=cfg,
                                         model_name="logistic", features=["ret_1"],
                                         calibration_bars=50)
    assert len(pred) > 0
    assert not policies.empty
    assert pred.groupby("model_version").selected_threshold.nunique().max() == 1
    assert (policies.calibration_end < policies.test_start).all()
