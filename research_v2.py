from __future__ import annotations

"""Leakage-safe model research runner.

This is deliberately separate from the live trading code. It produces fresh
purged walk-forward predictions, then evaluates model/feature/threshold
variants without using realized future returns as an entry condition.
"""

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from trading_bot import Config, CORE_FEATURES, download_history, make_features

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

FEATURE_SETS = {
    "all": CORE_FEATURES,
    "price_momentum": ["ret_1","ret_3","ret_6","ret_12","ret_24","ret_72","ret_168","trend_24_168","drawdown_168","drawdown_720"],
    "trend_volatility": ["ret_6","ret_24","ret_72","ret_168","vol_6","vol_24","vol_72","atr_pct","sma24_ratio","sma72_ratio","ema24_ratio","ema168_ratio","trend_24_168"],
    "technicals": ["rsi14","macd","macd_signal","macd_hist","bb_pos","bb_width","range_pct","body_pct"],
    "volume": ["volume_z","volume_ratio","vol_6","vol_24","vol_72"],
}


def pipeline(cfg: Config, model_name: str):
    if model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=250, min_samples_leaf=15, max_features="sqrt",
            class_weight="balanced", random_state=cfg.seed, n_jobs=-1,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
    if model_name == "logistic":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=.5, max_iter=3000, random_state=cfg.seed)),
        ])
    model = HistGradientBoostingClassifier(
        max_iter=250, learning_rate=.04, max_leaf_nodes=15,
        min_samples_leaf=30, l2_regularization=1.0, random_state=cfg.seed,
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def purged_walk_forward(d: pd.DataFrame, cfg: Config, model_name: str, features: list[str]) -> pd.DataFrame:
    """Generate strictly OOS probabilities.

    For a label at t with horizon H, its target uses prices through t+H.
    Therefore rows t >= start-H are excluded from training when predicting
    the first test row at start. This embargo is mandatory for honest testing.
    """
    out = d.copy().sort_values("Date").reset_index(drop=True)
    out["prob_up"] = np.nan
    out["model_version"] = -1
    total = len(out)
    version = 0
    for start in range(cfg.initial_train_bars, total, cfg.retrain_every_bars):
        end = min(start + cfg.retrain_every_bars, total)
        train_end = max(0, start - cfg.horizon_bars)
        train = out.iloc[:train_end]
        test = out.iloc[start:end]
        if len(train) < cfg.initial_train_bars or train.label.nunique() < 2:
            continue
        m = pipeline(cfg, model_name)
        m.fit(train[features], train.label)
        out.loc[test.index, "prob_up"] = m.predict_proba(test[features])[:, 1]
        out.loc[test.index, "model_version"] = version
        version += 1
    return out.dropna(subset=["prob_up"]).reset_index(drop=True)


def simulate(d: pd.DataFrame, cfg: Config, threshold: float, max_trades_per_day: int = 1,
             fee_bps: float | None = None, slip_bps: float | None = None) -> dict:
    """Long-only execution simulation; future labels are used only for evaluation."""
    fee = cfg.fee_bps if fee_bps is None else fee_bps
    slip = cfg.slippage_bps if slip_bps is None else slip_bps
    cash = cfg.starting_capital
    peak = cash
    daily_anchor = cash
    day = None
    trades_today = 0
    halted = False
    equity = []
    pnls = []

    i = 0
    while i < len(d) - 1:
        row = d.iloc[i]
        if row.Date.date() != day:
            day = row.Date.date()
            daily_anchor = cash
            trades_today = 0
            halted = False
        dd = cash / max(peak, 1.0) - 1
        if dd <= -cfg.max_drawdown_pct or (daily_anchor - cash) / max(daily_anchor, 1.0) >= cfg.max_daily_loss_pct:
            halted = True

        if halted or trades_today >= max_trades_per_day or float(row.prob_up) < threshold or not np.isfinite(row.atr_pct):
            equity.append((row.Date, cash))
            i += 1
            continue

        entry_row = d.iloc[i + 1]
        entry_raw = float(entry_row.Open)
        entry = entry_raw * (1 + slip / 10000)
        atr = float(row.atr_pct) * entry
        stop = entry - cfg.stop_atr_multiple * atr
        target = entry + cfg.target_atr_multiple * atr
        qty = min(cash * cfg.risk_per_trade / max(entry - stop, entry * 1e-9), cash * cfg.max_position_pct / entry)
        if qty <= 0:
            i += 1
            continue

        exit_idx = min(i + cfg.horizon_bars, len(d) - 1)
        reason = "horizon"
        exit_raw = float(d.iloc[exit_idx].Close)
        actual_exit = exit_idx
        for j in range(i + 1, exit_idx + 1):
            bar = d.iloc[j]
            if float(bar.Low) <= stop:
                exit_raw, reason, actual_exit = stop, "stop", j
                break
            if float(bar.High) >= target:
                exit_raw, reason, actual_exit = target, "target", j
                break

        exit_px = exit_raw * (1 - slip / 10000)
        gross = qty * (exit_px - entry)
        fees = (qty * entry + qty * exit_px) * fee / 10000
        net = gross - fees
        cash += net
        peak = max(peak, cash)
        trades_today += 1
        pnls.append(net)
        equity.append((d.iloc[actual_exit].Date, cash))
        i = actual_exit + 1

    eq = pd.DataFrame(equity, columns=["Date", "capital"]).drop_duplicates("Date").sort_values("Date")
    if eq.empty:
        return {"return_pct": 0.0, "max_drawdown_pct": 0.0, "trades": 0}
    curve = eq.capital
    dd = curve / curve.cummax() - 1
    years = max((eq.Date.iloc[-1] - eq.Date.iloc[0]).total_seconds() / (365.25 * 86400), 1e-9)
    daily = eq.set_index("Date").capital.resample("1D").last().ffill().pct_change().dropna()
    sharpe = daily.mean() / daily.std() * math.sqrt(365) if len(daily) > 1 and daily.std() > 0 else np.nan
    wins = sum(x > 0 for x in pnls)
    losses = sum(x <= 0 for x in pnls)
    gp = sum(x for x in pnls if x > 0)
    gl = -sum(x for x in pnls if x <= 0)
    return {
        "final_capital": float(cash),
        "return_pct": (cash / cfg.starting_capital - 1) * 100,
        "cagr_pct": ((cash / cfg.starting_capital) ** (1 / years) - 1) * 100 if cash > 0 else -100,
        "max_drawdown_pct": float(dd.min() * 100),
        "sharpe": float(sharpe) if pd.notna(sharpe) else None,
        "trades": len(pnls),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": wins / len(pnls) * 100 if pnls else None,
        "profit_factor": gp / gl if gl > 0 else None,
        "expectancy_per_trade": float(np.mean(pnls)) if pnls else None,
    }


def main():
    cfg = Config()
    raw_path = OUT / "binance_btcusdt_history.csv"
    features_path = OUT / "features.csv"
    if features_path.exists():
        d = pd.read_csv(features_path, parse_dates=["Date"])
    elif raw_path.exists():
        raw = pd.read_csv(raw_path, parse_dates=["Date"])
        d = make_features(raw, cfg)
    else:
        raw = download_history(cfg)
        d = make_features(raw, cfg)

    d["Date"] = pd.to_datetime(d["Date"], utc=True)
    results = []
    prediction_sets = {}

    for model_name in ["logistic", "hist_gradient_boosting", "random_forest"]:
        for feature_name, features in FEATURE_SETS.items():
            pred = purged_walk_forward(d, cfg, model_name, features)
            key = f"{model_name}__{feature_name}"
            prediction_sets[key] = pred
            auc = float(roc_auc_score(pred.label, pred.prob_up)) if pred.label.nunique() == 2 else None
            brier = float(brier_score_loss(pred.label, pred.prob_up)) if pred.label.nunique() == 2 else None
            for threshold in [0.55, 0.57, 0.60, 0.63, 0.66]:
                sim = simulate(pred, cfg, threshold, max_trades_per_day=1)
                sim.update({"model": model_name, "feature_set": feature_name, "threshold": threshold, "roc_auc": auc, "brier_score": brier})
                results.append(sim)

    table = pd.DataFrame(results).sort_values(["sharpe", "profit_factor", "return_pct"], ascending=False, na_position="last")
    table.to_csv(OUT / "research_v2_results.csv", index=False)
    report = {
        "method": "purged_walk_forward_model_feature_threshold_search",
        "important_rule": "No realized future return is used to decide entry; training labels are embargoed by the prediction horizon.",
        "oos_start": str(min(pd.to_datetime(x.Date.iloc[0], utc=True) for x in prediction_sets.values())),
        "oos_end": str(max(pd.to_datetime(x.Date.iloc[-1], utc=True) for x in prediction_sets.values())),
        "rows": len(d),
        "top_20": table.head(20).to_dict(orient="records"),
        "config": asdict(cfg),
    }
    (OUT / "research_v2_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
