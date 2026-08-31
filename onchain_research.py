"""Task 2: free Bitcoin on-chain feature research.

Uses Coin Metrics Community API and Blockchain.com public Charts API. No paid
provider or API key is required for these public/community endpoints.
All on-chain observations are lagged by one full UTC day before becoming model
features, avoiding accidental use of a metric published after the prediction.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import trading_bot
from onchain_data import fetch_all, merge_daily_market_onchain

ONCHAIN_RAW = [
    "AdrActCnt", "TxCnt", "TxTfrCnt", "FeeTotNtv", "FeeTotUSD", "SplyCur",
    "SplyAct1d", "SplyAct30d", "SplyAct90d", "SplyAct180d", "SplyAct1yr",
    "SplyAct2yr", "SplyAct3yr", "SplyAct5yr", "SplyAct7yr", "SplyAct10yr",
    "RevNtv", "RevUSD", "HashRate", "DiffMean", "transactions_per_second",
    "transaction_fees_usd", "total_btc", "hash_rate", "n_transactions",
]


def add_onchain_features(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy().sort_values("Date").reset_index(drop=True)
    for c in [c for c in ONCHAIN_RAW if c in x.columns]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
        # Derive changes/z-scores from information available at the prior day,
        # then shift every raw series by one day before it reaches the model.
        x[f"oc_{c}_chg1"] = x[c].pct_change(1).shift(1)
        x[f"oc_{c}_chg7"] = x[c].pct_change(7).shift(1)
        rolling_mean = x[c].rolling(30).mean().shift(1)
        rolling_std = x[c].rolling(30).std().shift(1)
        x[f"oc_{c}_z30"] = (x[c] - rolling_mean) / rolling_std
        x[c] = x[c].shift(1)
    x = x.replace([np.inf, -np.inf], np.nan)
    return x


def prepare_daily_dataset(cfg: trading_bot.Config) -> pd.DataFrame:
    daily_cfg = trading_bot.Config(**asdict(cfg))
    daily_cfg.interval = "1d"
    raw = trading_bot.download_history(daily_cfg)
    start = raw.Date.min().strftime("%Y-%m-%d")
    end = (raw.Date.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    oc = fetch_all(start, end)
    merged = merge_daily_market_onchain(raw, oc)
    merged = add_onchain_features(merged)
    base = trading_bot.make_features(
        merged[["Date", "Open", "High", "Low", "Close", "Volume"]], daily_cfg
    )
    oc_cols = [c for c in merged.columns if c.startswith("oc_")]
    return base.merge(merged[["Date"] + oc_cols], on="Date", how="left")


def run(cfg: trading_bot.Config) -> None:
    d = prepare_daily_dataset(cfg)
    oc_features = [c for c in d.columns if c.startswith("oc_")]
    oc_features = [c for c in oc_features if d[c].notna().mean() >= 0.20]
    features = trading_bot.CORE_FEATURES + oc_features
    trainable = d.dropna(subset=["future_return", "label"]).copy()

    train_n, step = 730, 30
    probs = np.full(len(trainable), np.nan)
    for start in range(train_n, len(trainable), step):
        end = min(start + step, len(trainable))
        tr, te = trainable.iloc[:start], trainable.iloc[start:end]
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
    eq, trades = trading_bot.backtest(trainable, cfg)
    report = {
        "dataset": "BTCUSDT daily + free on-chain/community data",
        "onchain_features_used": oc_features,
        "rows": len(trainable),
        "strategy": trading_bot.summarize(eq, trades, cfg),
        "data_start": str(trainable.Date.min()),
        "data_end": str(trainable.Date.max()),
    }
    out = Path(__file__).resolve().parent / "outputs"
    out.mkdir(exist_ok=True)
    trainable.to_csv(out / "onchain_enriched_predictions.csv", index=False)
    (out / "onchain_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run(trading_bot.Config(interval="1d"))
