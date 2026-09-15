"""Nested policy-selection runner for the six-bar BTC research pipeline.

The outer test fold is never used to choose an execution threshold. A small,
predeclared threshold family is selected on the tail of each training window,
then frozen for that fold's outer test period. This separates prediction OOS
from policy OOS and prevents the common mistake of ranking thresholds on the
same returns later reported as evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from research_v2 import pipeline, simulate

THRESHOLDS = (0.55, 0.57, 0.60, 0.63, 0.66)
CALIBRATION_BARS = 24 * 30


@dataclass(frozen=True)
class FoldPolicy:
    model: str
    feature_set: str
    fold: int
    calibration_start: str
    calibration_end: str
    test_start: str
    test_end: str
    threshold: float


def _select_threshold(calibration: pd.DataFrame, cfg, thresholds=THRESHOLDS) -> float:
    """Select a fixed threshold using development/calibration returns only."""
    scored = []
    for threshold in thresholds:
        m = simulate(calibration, cfg, threshold, max_trades_per_day=1)
        # Selection criterion is deliberately simple and predeclared.
        score = m.get("sharpe")
        if score is None or not np.isfinite(score):
            score = -np.inf
        scored.append((float(score), float(threshold)))
    # Stable tie-break: choose the lower threshold, never a test-period result.
    return max(scored, key=lambda x: (x[0], -x[1]))[1]


def nested_walk_forward(d: pd.DataFrame, cfg, model_name: str, features: list[str],
                        calibration_bars: int = CALIBRATION_BARS,
                        thresholds=THRESHOLDS):
    """Return outer-OOS predictions with one frozen policy per fold."""
    x = d.sort_values("Date").reset_index(drop=True).copy()
    rows = []
    policies = []
    fold = 0
    for start in range(cfg.initial_train_bars, len(x), cfg.retrain_every_bars):
        end = min(start + cfg.retrain_every_bars, len(x))
        train_end = max(0, start - cfg.horizon_bars)
        train = x.iloc[:train_end]
        test = x.iloc[start:end]
        if len(train) < cfg.initial_train_bars or train.label.nunique() < 2:
            continue

        # Calibration is the most recent development block, ending before the
        # outer test fold and itself separated from fitting by the label horizon.
        cal_end = train_end
        cal_start = max(0, cal_end - calibration_bars)
        fit_end = max(0, cal_start - cfg.horizon_bars)
        fit = x.iloc[:fit_end]
        calibration = x.iloc[cal_start:cal_end]
        if len(fit) < cfg.initial_train_bars or calibration.label.nunique() < 2:
            continue

        m = pipeline(cfg, model_name)
        m.fit(fit[features], fit.label)
        cal = calibration.copy()
        cal["prob_up"] = m.predict_proba(calibration[features])[:, 1]
        te = test.copy()
        te["prob_up"] = m.predict_proba(test[features])[:, 1]
        threshold = _select_threshold(cal, cfg, thresholds)
        te["selected_threshold"] = threshold
        te["model_version"] = fold
        rows.append(te)
        policies.append(FoldPolicy(
            model_name, "custom", fold,
            str(calibration.Date.iloc[0]), str(calibration.Date.iloc[-1]),
            str(test.Date.iloc[0]), str(test.Date.iloc[-1]), threshold))
        fold += 1

    pred = pd.concat(rows, ignore_index=True) if rows else x.iloc[0:0].copy()
    return pred, pd.DataFrame([p.__dict__ for p in policies])


def evaluate_nested(pred: pd.DataFrame, cfg) -> dict:
    """Evaluate the already-frozen per-fold policies without re-selection."""
    if pred.empty:
        return {"rows": 0, "folds": 0}
    fold_results = []
    for fold, part in pred.groupby("model_version", sort=True):
        threshold = float(part.selected_threshold.iloc[0])
        sim = simulate(part, cfg, threshold, max_trades_per_day=1)
        fold_results.append({"fold": int(fold), "threshold": threshold, **sim})
    return {"rows": int(len(pred)), "folds": len(fold_results), "fold_results": fold_results}
