"""Task 4 research: evaluate macro features against the BTC baseline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from macro_data import engineer_macro_features, fetch_macro
from trading_bot import CORE_FEATURES, Config, download_history, walk_forward

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)


def _macro_columns(d: pd.DataFrame) -> list[str]:
    cols = [c for c in d.columns if c.startswith("macro_")]
    return [c for c in cols if d[c].notna().mean() >= 0.20]


def _macro_walk_forward(d: pd.DataFrame, features: list[str], cfg: Config) -> pd.DataFrame:
    x = d.dropna(subset=["future_return", "label"]).copy().reset_index(drop=True)
    x["prob_up_macro"] = np.nan
    for start in range(max(cfg.initial_train_bars, 730), len(x), cfg.retrain_every_bars):
        end = min(start + cfg.retrain_every_bars, len(x))
        tr, te = x.iloc[:start], x.iloc[start:end]
        if tr.label.nunique() < 2:
            continue
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                max_iter=250, learning_rate=0.04, max_leaf_nodes=15,
                min_samples_leaf=30, l2_regularization=1.0, random_state=cfg.seed
            )),
        ])
        model.fit(tr[features], tr.label)
        x.loc[te.index, "prob_up_macro"] = model.predict_proba(te[features])[:, 1]
    return x.dropna(subset=["prob_up_macro"]).reset_index(drop=True)


def run() -> None:
    cfg = Config(interval="1d", initial_train_bars=730, retrain_every_bars=30, horizon_bars=1)
    raw = download_history(cfg)
    from trading_bot import make_features
    base = make_features(raw, cfg)
    macro = engineer_macro_features(fetch_macro(str(raw.Date.min())[:10], str(raw.Date.max())[:10]))
    enriched = base.merge(macro, on="Date", how="left")
    macro_cols = _macro_columns(enriched)
    if not macro_cols:
        raise RuntimeError("No macro features have enough history to evaluate")

    baseline = walk_forward(base, cfg)[["Date", "prob_up"]].rename(columns={"prob_up": "prob_up_baseline"})
    combined = _macro_walk_forward(enriched, CORE_FEATURES + macro_cols, cfg)
    combined = combined.merge(baseline, on="Date", how="left")

    def brier(p: pd.Series, y: pd.Series) -> float:
        return float(np.mean((p.to_numpy() - y.to_numpy()) ** 2))

    baseline_eval = combined.dropna(subset=["prob_up_baseline"])
    report = {
        "dataset": "BTCUSDT daily + free FRED macro data",
        "macro_features": macro_cols,
        "rows_evaluated": len(combined),
        "baseline_brier": brier(baseline_eval.prob_up_baseline, baseline_eval.label) if len(baseline_eval) else None,
        "macro_brier": brier(combined.prob_up_macro, combined.label),
        "baseline_accuracy": float(((baseline_eval.prob_up_baseline >= .5).astype(int) == baseline_eval.label).mean()) if len(baseline_eval) else None,
        "macro_accuracy": float(((combined.prob_up_macro >= .5).astype(int) == combined.label).mean()),
        "warning": "FRED graph downloads reflect current observations. Final release-sensitive research should use ALFRED vintages/release timestamps to eliminate revision bias.",
    }
    combined.to_csv(OUT / "macro_enriched_predictions.csv", index=False)
    (OUT / "macro_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
