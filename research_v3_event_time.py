from __future__ import annotations

"""Event-time-correct research runner.

This runner removes the row-count purge shortcut and trains on the same
executable interval used by the economic simulation: signal at t, entry at
Open[t+1], label endpoint at Close[t+H].  Purging is performed from actual
label_end_time < test_decision_time, so missing/irregular bars cannot silently
invalidate the embargo.
"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from research_v2 import FEATURE_SETS, pipeline
from trading_bot import Config, make_features

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def validate_raw_ohlcv(d: pd.DataFrame, cfg: Config) -> None:
    """Fail closed on the untouched market event stream before any transforms."""
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = required - set(d.columns)
    if missing:
        raise AssertionError(f"raw data missing columns: {sorted(missing)}")
    x = d.copy()
    x["Date"] = pd.to_datetime(x["Date"], utc=True, errors="coerce")
    if x["Date"].isna().any():
        raise AssertionError("raw data contains unparseable timestamps")
    if not x["Date"].is_monotonic_increasing:
        raise AssertionError("raw timestamps are not strictly chronological")
    if x["Date"].duplicated().any():
        raise AssertionError("raw data contains duplicate timestamps")
    numeric = ["Open", "High", "Low", "Close", "Volume"]
    if not np.isfinite(x[numeric].to_numpy(dtype=float)).all():
        raise AssertionError("raw data contains NaN/Inf OHLCV")
    if (x[["Open", "High", "Low", "Close"]] <= 0).any().any() or (x["Volume"] < 0).any():
        raise AssertionError("raw data contains non-positive prices or negative volume")
    if (x["High"] < x[["Open", "Close"]].max(axis=1)).any():
        raise AssertionError("raw High is below Open/Close")
    if (x["Low"] > x[["Open", "Close"]].min(axis=1)).any():
        raise AssertionError("raw Low is above Open/Close")
    if cfg.interval == "1h":
        gaps = x["Date"].diff().dropna()
        if (gaps != pd.Timedelta(hours=1)).any():
            raise AssertionError("irregular hourly raw stream; repair data before research")

def add_executable_label(d: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    d = d.sort_values("Date").reset_index(drop=True).copy()
    # Signal is formed at t. The first executable price is Open[t+1].
    d["entry_time"] = d["Date"].shift(-1)
    d["entry_open"] = d["Open"].shift(-1)
    d["label_end_time"] = d["Date"].shift(-cfg.horizon_bars)
    d["executable_return"] = d["Close"].shift(-cfg.horizon_bars) / d["Open"].shift(-1) - 1.0
    d["label"] = (d["executable_return"] > 0).astype(int)
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=FEATURE_SETS["all"] + ["entry_time", "entry_open", "label_end_time", "executable_return"])
    return d.reset_index(drop=True)


def validate_event_time(d: pd.DataFrame, cfg: Config) -> None:
    if not d["Date"].is_monotonic_increasing:
        raise AssertionError("decision timestamps are not sorted")
    if d["Date"].duplicated().any():
        raise AssertionError("duplicate decision timestamps")
    gaps = d["Date"].diff().dropna()
    expected = pd.Timedelta(hours=1)
    if (gaps != expected).any():
        raise AssertionError("irregular hourly bars: event-time validation must be repaired before testing")
    if not (d["label_end_time"] > d["Date"]).all():
        raise AssertionError("label endpoint must be after decision time")


def purged_walk_forward(d: pd.DataFrame, cfg: Config, model_name: str, features: list[str]) -> pd.DataFrame:
    out = d.sort_values("Date").reset_index(drop=True).copy()
    out["prob_up"] = np.nan
    out["model_version"] = -1
    dates = out["Date"].to_numpy()
    version = 0
    for start in range(cfg.initial_train_bars, len(out), cfg.retrain_every_bars):
        test_start = pd.Timestamp(dates[start])
        end = min(start + cfg.retrain_every_bars, len(out))
        # Event-time purge: a training label may not extend into the test period.
        train_mask = out["label_end_time"] < test_start
        train = out.loc[train_mask]
        test = out.iloc[start:end]
        if len(train) < cfg.initial_train_bars or train.label.nunique() < 2:
            continue
        m = pipeline(cfg, model_name)
        m.fit(train[features], train.label)
        out.loc[test.index, "prob_up"] = m.predict_proba(test[features])[:, 1]
        out.loc[test.index, "model_version"] = version
        version += 1
    return out.dropna(subset=["prob_up"]).reset_index(drop=True)


def simulate_executable(d: pd.DataFrame, cfg: Config, threshold: float) -> dict:
    """Simple fixed-horizon economic test matching the executable label interval.

    This is intentionally a diagnostic baseline: it uses Open[t+1] entry and
    Close[t+H] exit, with round-trip fees and symmetric slippage. Stop/target
    logic is left to the later canonical event-time execution ledger.
    """
    fee = cfg.fee_bps / 10000.0
    slip = cfg.slippage_bps / 10000.0
    rets = []
    for _, r in d.iterrows():
        if float(r.prob_up) < threshold:
            continue
        gross = (1.0 + float(r.executable_return))
        # Entry pays slippage; exit pays slippage. Apply multiplicatively so
        # the diagnostic is conservative and consistent across observations.
        net_growth = gross * (1.0 - slip) / (1.0 + slip) * (1.0 - fee) ** 2
        rets.append(net_growth - 1.0)
    if not rets:
        return {"trades": 0, "mean_net_return": None, "win_rate_pct": None}
    a = np.asarray(rets, dtype=float)
    return {
        "trades": int(len(a)),
        "mean_net_return": float(a.mean()),
        "median_net_return": float(np.median(a)),
        "win_rate_pct": float((a > 0).mean() * 100),
    }


def main() -> None:
    cfg = Config()
    raw_path = OUT / "binance_btcusdt_history.csv"
    if not raw_path.exists():
        raise FileNotFoundError("Canonical research input missing: binance_btcusdt_history.csv")
    raw = pd.read_csv(raw_path, parse_dates=["Date"])
    raw["Date"] = pd.to_datetime(raw["Date"], utc=True, errors="coerce")
    validate_raw_ohlcv(raw, cfg)
    d = make_features(raw, cfg)
    d["Date"] = pd.to_datetime(d["Date"], utc=True)
    d = add_executable_label(d, cfg)
    validate_event_time(d, cfg)

    rows = []
    for model_name in ["logistic", "hist_gradient_boosting", "random_forest"]:
        for feature_name, features in FEATURE_SETS.items():
            pred = purged_walk_forward(d, cfg, model_name, features)
            auc = float(roc_auc_score(pred.label, pred.prob_up)) if pred.label.nunique() == 2 else None
            brier = float(brier_score_loss(pred.label, pred.prob_up)) if pred.label.nunique() == 2 else None
            for threshold in [0.55, 0.57, 0.60, 0.63, 0.66]:
                sim = simulate_executable(pred, cfg, threshold)
                rows.append({"model": model_name, "feature_set": feature_name, "threshold": threshold,
                             "roc_auc": auc, "brier_score": brier, **sim})

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "research_v3_event_time_results.csv", index=False)
    report = {
        "method": "event_time_purged_walk_forward",
        "label": "Open[t+1] -> Close[t+H]",
        "purge_rule": "training label_end_time < test decision_time",
        "regularity_rule": "exact 1h timestamps; irregular bars fail closed",
        "rows": int(len(d)),
        "input": str(raw_path),
        "raw_rows": int(len(raw)),
        "oos_start": str(d.Date.iloc[0]),
        "oos_end": str(d.Date.iloc[-1]),
        "config": asdict(cfg),
        "results_file": str(OUT / "research_v3_event_time_results.csv"),
    }
    (OUT / "research_v3_event_time_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
