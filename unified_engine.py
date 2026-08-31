"""Unified BTC prediction engine.

Combines candidate feature groups without leaking future information. Models are
trained only on observations strictly before each prediction timestamp. Feature
groups can be enabled independently so their incremental value can be measured.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

BASE_FEATURES = [
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_24", "ret_72", "ret_168",
    "vol_6", "vol_24", "vol_72", "atr_pct", "range_pct", "body_pct", "rsi14",
    "macd", "macd_signal", "macd_hist", "sma24_ratio", "sma72_ratio",
    "ema24_ratio", "ema168_ratio", "bb_pos", "bb_width", "volume_z", "volume_ratio",
    "trend_24_168", "drawdown_168", "drawdown_720"
]

@dataclass
class EngineConfig:
    horizon_bars: int = 6
    initial_train_bars: int = 24 * 180
    retrain_every_bars: int = 24 * 7
    probability_long: float = 0.56
    probability_short: float = 0.44
    min_expected_edge: float = 0.0015
    seed: int = 42


def candidate_columns(df: pd.DataFrame, groups: list[str]) -> list[str]:
    cols = [c for c in BASE_FEATURES if c in df.columns]
    prefixes = {
        "onchain": ("onchain_", "active_", "transaction_", "supply_", "hashrate", "fees_"),
        "derivatives": ("oi_", "funding_", "global_", "top_", "taker_", "basis_"),
        "macro": ("macro_",),
        "sentiment": ("fear_greed", "news_", "sentiment_"),
        "breadth": ("breadth_", "etf_", "trends_", "dominance_"),
    }
    for g in groups:
        for c in df.columns:
            if any(c.startswith(p) or p in c for p in prefixes.get(g, ())):
                if c not in cols and c not in {"label", "future_return", "Date"}:
                    cols.append(c)
    return cols


def make_model(seed: int, kind: str) -> Pipeline:
    if kind == "logistic":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                         ("model", LogisticRegression(C=0.5, max_iter=3000, random_state=seed))])
    if kind == "rf":
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("model", RandomForestClassifier(n_estimators=350, min_samples_leaf=12,
                                                           max_features="sqrt", class_weight="balanced",
                                                           random_state=seed, n_jobs=-1))])
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("model", HistGradientBoostingClassifier(max_iter=250, learning_rate=0.04,
                                                               max_leaf_nodes=15, min_samples_leaf=30,
                                                               l2_regularization=1.0, random_state=seed))])


def walk_forward_ensemble(df: pd.DataFrame, features: list[str], cfg: EngineConfig) -> pd.DataFrame:
    d = df.sort_values("Date").reset_index(drop=True).copy()
    d["prob_up"] = np.nan
    d["prob_std"] = np.nan
    d["ensemble_models"] = 0
    for start in range(cfg.initial_train_bars, len(d), cfg.retrain_every_bars):
        end = min(start + cfg.retrain_every_bars, len(d))
        train, test = d.iloc[:start], d.iloc[start:end]
        if train["label"].nunique() < 2 or test.empty:
            continue
        probs = []
        for kind in ("logistic", "rf", "hgb"):
            model = make_model(cfg.seed, kind)
            model.fit(train[features], train.label)
            probs.append(model.predict_proba(test[features])[:, 1])
        arr = np.vstack(probs)
        d.loc[test.index, "prob_up"] = arr.mean(axis=0)
        d.loc[test.index, "prob_std"] = arr.std(axis=0)
        d.loc[test.index, "ensemble_models"] = arr.shape[0]
    return d.dropna(subset=["prob_up"]).reset_index(drop=True)


def evaluate(pred: pd.DataFrame) -> dict:
    if pred.empty:
        return {"rows": 0}
    y, p = pred.label.astype(int), pred.prob_up.clip(1e-6, 1 - 1e-6)
    result = {"rows": len(pred), "accuracy": float(accuracy_score(y, p >= .5)),
              "log_loss": float(log_loss(y, p)), "mean_probability": float(p.mean()),
              "mean_ensemble_disagreement": float(pred.prob_std.mean())}
    if y.nunique() == 2:
        result["roc_auc"] = float(roc_auc_score(y, p))
    # Signal quality, not a claim of future profitability.
    result["trade_rate"] = float(((p >= .56) | (p <= .44)).mean())
    result["long_rate"] = float((p >= .56).mean())
    result["short_rate"] = float((p <= .44).mean())
    return result


def run_feature_ablation(df: pd.DataFrame, cfg: EngineConfig) -> dict:
    experiments = {
        "technical_only": [],
        "technical_onchain": ["onchain"],
        "technical_derivatives": ["derivatives"],
        "technical_macro": ["macro"],
        "technical_sentiment": ["sentiment"],
        "technical_breadth": ["breadth"],
        "all_sources": ["onchain", "derivatives", "macro", "sentiment", "breadth"],
    }
    reports = {}
    for name, groups in experiments.items():
        cols = candidate_columns(df, groups)
        if len(cols) == len([c for c in BASE_FEATURES if c in df.columns]):
            # Still evaluate the baseline; it is our control.
            pass
        pred = walk_forward_ensemble(df, cols, cfg)
        pred.to_csv(OUT / f"unified_{name}_predictions.csv", index=False)
        reports[name] = {"features": cols, "metrics": evaluate(pred)}
    (OUT / "unified_feature_ablation.json").write_text(json.dumps(reports, indent=2, default=str))
    return reports


def add_decision_layer(pred: pd.DataFrame, cfg: EngineConfig) -> pd.DataFrame:
    d = pred.copy()
    # Expected return is kept separate from probability so the caller can replace
    # it with a calibrated return model later.
    d["signal"] = np.select(
        [d.prob_up >= cfg.probability_long, d.prob_up <= cfg.probability_short],
        ["LONG", "SHORT"], default="HOLD")
    d["confidence"] = (np.abs(d.prob_up - 0.5) * 2).clip(0, 1)
    d["signal_strength"] = d["confidence"] * (1 - d["prob_std"].fillna(0).clip(0, 1))
    d.loc[d["signal_strength"] < 0.10, "signal"] = "HOLD"
    return d


def save_latest(pred: pd.DataFrame, cfg: EngineConfig) -> dict:
    latest = pred.iloc[-1]
    result = {"timestamp": str(latest.Date), "prob_up": float(latest.prob_up),
              "prob_down": float(1 - latest.prob_up), "confidence": float(latest.confidence),
              "ensemble_disagreement": float(latest.prob_std), "signal": str(latest.signal)}
    (OUT / "latest_unified_signal.json").write_text(json.dumps(result, indent=2))
    return result


def prepare_label(df: pd.DataFrame, horizon_bars: int) -> pd.DataFrame:
    d = df.sort_values("Date").reset_index(drop=True).copy()
    d["future_return"] = d.Close.shift(-horizon_bars) / d.Close - 1
    d["label"] = (d.future_return > 0).astype(int)
    return d.dropna(subset=["future_return"])


if __name__ == "__main__":
    source = OUT / "unified_dataset.csv"
    if not source.exists():
        raise SystemExit("Create outputs/unified_dataset.csv by joining the source datasets before running this module.")
    cfg = EngineConfig()
    data = prepare_label(pd.read_csv(source, parse_dates=["Date"]), cfg.horizon_bars)
    reports = run_feature_ablation(data, cfg)
    all_pred = walk_forward_ensemble(data, candidate_columns(data, list({"onchain","derivatives","macro","sentiment","breadth"})), cfg)
    all_pred = add_decision_layer(all_pred, cfg)
    all_pred.to_csv(OUT / "unified_predictions.csv", index=False)
    save_latest(all_pred, cfg)
    print(json.dumps(reports, indent=2, default=str))
