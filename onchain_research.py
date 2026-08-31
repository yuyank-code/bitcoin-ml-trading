"""Task 2: free Bitcoin on-chain feature research.

Uses Coin Metrics Community API and Blockchain.com public Charts API. No paid
provider or API key is required for these public/community endpoints.
All on-chain observations are lagged by one full UTC day before becoming model
features, avoiding accidental use of a metric published after the prediction.
"""
from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from onchain_data import fetch_all, merge_daily_market_onchain
from trading_bot import Config, download_history, make_features, summarize, backtest

ONCHAIN_RAW = [
    "AdrActCnt", "TxCnt", "TxTfrCnt", "FeeTotNtv", "FeeTotUSD", "SplyCur",
    "SplyAct1d", "SplyAct30d", "SplyAct90d", "SplyAct180d", "SplyAct1yr",
    "SplyAct2yr", "SplyAct3yr", "SplyAct5yr", "SplyAct7yr", "SplyAct10yr",
    "RevNtv", "RevUSD", "HashRate", "DiffMean", "transactions_per_second",
    "transaction_fees_usd", "total_btc", "hash_rate", "n_transactions",
]


def add_onchain_features(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy().sort_values("Date").reset_index(drop=True)
    # A full-day lag is deliberately applied to every on-chain series.
    for c in [c for c in ONCHAIN_RAW if c in x.columns]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
        x[f"oc_{c}_chg1"] = x[c].pct_change(1)
        x[f"oc_{c}_chg7"] = x[c].pct_change(7)
        x[f"oc_{c}_z30"] = (x[c] - x[c].rolling(30).mean()) / x[c].rolling(30).std()
        x[c] = x[c].shift(1)
    # Cross-source features are also based on the lagged values.
    if "AdrActCnt" in x:
        x["oc_activity_per_tx"] = x["AdrActCnt"] / x.get("TxCnt", pd.Series(np.nan, index=x.index)).replace(0, np.nan)
    if "SplyCur" in x and "total_btc" in x:
        x["oc_supply_ratio"] = x["SplyCur"] / x["total_btc"].replace(0, np.nan)
    x = x.replace([np.inf, -np.inf], np.nan)
    return x


def prepare_daily_dataset(cfg: Config) -> pd.DataFrame:
    daily_cfg = Config(**asdict(cfg))
    daily_cfg.interval = "1d"
    raw = download_history(daily_cfg)
    start = raw.Date.min().strftime("%Y-%m-%d")
    end = (raw.Date.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    oc = fetch_all(start, end)
    merged = merge_daily_market_onchain(raw, oc)
    merged = add_onchain_features(merged)
    base = make_features(merged[["Date", "Open", "High", "Low", "Close", "Volume"]], daily_cfg)
    # Join the point-in-time-safe on-chain features back onto the model rows.
    oc_cols = [c for c in merged.columns if c.startswith("oc_") or c in ONCHAIN_RAW]
    enriched = base.merge(merged[["Date"] + oc_cols], on="Date", how="left")
    return enriched


def run(cfg: Config) -> None:
    d = prepare_daily_dataset(cfg)
    oc_features = [c for c in d.columns if c.startswith("oc_")]
    # Only retain columns with enough observations to be useful.
    oc_features = [c for c in oc_features if d[c].notna().mean() >= 0.20]
    features = [c for c in __import__("trading_bot").CORE_FEATURES] + oc_features
    trainable = d.dropna(subset=["future_return", "label"]).copy()
    # Walk-forward model with on-chain features. No scaler fitted globally.
    train_n = 730
    step = 30
    probs = np.full(len(trainable), np.nan)
    for start in range(train_n, len(trainable), step):
        end = min(start + step, len(trainable))
        tr = trainable.iloc[:start]
        te = trainable.iloc[start:end]
        if tr.label.nunique() < 2:
            continue
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.04, max_leaf_nodes=12,
                min_samples_leaf=30, l2_regularization=1.0, random_state=cfg.seed
            )),
        ])
        model.fit(tr[features], tr.label)
        probs[start:end] = model.predict_proba(te[features])[:, 1]
    trainable["prob_up"] = probs
    trainable = trainable.dropna(subset=["prob_up"]).reset_index(drop=True)
    eq, trades = backtest(trainable, cfg)
    report = {
        "dataset": "BTCUSDT daily + free on-chain/community data",
        "onchain_features_used": oc_features,
        "rows": len(trainable),
        "strategy": summarize(eq, trades, cfg),
        "data_start": str(trainable.Date.min()),
        "data_end": str(trainable.Date.max()),
    }
    (pd.Path if hasattr(pd, "Path") else None)
    from pathlib import Path
    out = Path(__file__).resolve().parent / "outputs"
    out.mkdir(exist_ok=True)
    trainable.to_csv(out / "onchain_enriched_predictions.csv", index=False)
    (out / "onchain_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run(Config(interval="1d"))
