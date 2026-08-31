from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from trading_bot import Config, download_history, make_features, walk_forward

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def load_oos_predictions(cfg: Config) -> tuple[pd.DataFrame, str]:
    pred_path = OUT / "predictions.csv"
    if pred_path.exists():
        d = pd.read_csv(pred_path, parse_dates=["Date"])
        if {"Date", "prob_up", "label", "future_return", "atr_pct", "Open", "High", "Low", "Close"}.issubset(d.columns):
            d = d.sort_values("Date").reset_index(drop=True)
            return d, "cached_predictions.csv"

    raw_path = OUT / "binance_btcusdt_history.csv"
    raw = pd.read_csv(raw_path, parse_dates=["Date"]) if raw_path.exists() else download_history(cfg)
    feat = make_features(raw, cfg)
    pred = walk_forward(feat, cfg)
    pred.to_csv(pred_path, index=False)
    return pred, "fresh_walk_forward"


def leak_free_backtest(d: pd.DataFrame, cfg: Config, probability: float | None = None,
                       fee_bps: float | None = None, slippage_bps: float | None = None,
                       max_trades_per_day: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Historical simulation using ONLY information available at signal time.

    Critical rule: future_return/label are never used to decide whether to enter.
    They exist only for post-trade evaluation. Positions are held until stop,
    target, or the configured prediction horizon.
    """
    p_threshold = cfg.probability_long if probability is None else probability
    fee = cfg.fee_bps if fee_bps is None else fee_bps
    slip = cfg.slippage_bps if slippage_bps is None else slippage_bps
    daily_limit = cfg.max_trades_per_day if max_trades_per_day is None else max_trades_per_day

    cash = cfg.starting_capital
    peak = cash
    equity = []
    trades = []
    i = 0
    day = None
    daily_anchor = cash
    trades_today = 0
    halted = False

    d = d.sort_values("Date").reset_index(drop=True)
    while i < len(d) - 1:
        row = d.iloc[i]
        current_day = row.Date.date()
        if current_day != day:
            day = current_day
            daily_anchor = cash
            trades_today = 0
            halted = False

        dd = cash / max(peak, 1.0) - 1.0
        if dd <= -cfg.max_drawdown_pct or (daily_anchor - cash) / max(daily_anchor, 1.0) >= cfg.max_daily_loss_pct:
            halted = True

        p = float(row.prob_up)
        can_enter = (
            not halted
            and trades_today < daily_limit
            and p >= p_threshold
            and pd.notna(row.atr_pct)
        )

        if not can_enter:
            equity.append((row.Date, cash, cash / max(peak, 1.0) - 1.0))
            i += 1
            continue

        entry_idx = i + 1
        if entry_idx >= len(d):
            break
        entry_row = d.iloc[entry_idx]
        entry_raw = float(entry_row.Open)
        entry = entry_raw * (1.0 + slip / 10000.0)
        atr = float(row.atr_pct) * entry
        if not np.isfinite(atr) or atr <= 0:
            i += 1
            continue

        stop = entry - cfg.stop_atr_multiple * atr
        target = entry + cfg.target_atr_multiple * atr
        risk_per_unit = max(entry - stop, entry * 1e-9)
        qty = min(
            cash * cfg.risk_per_trade / risk_per_unit,
            cash * cfg.max_position_pct / entry,
        )
        if qty <= 0:
            i += 1
            continue

        exit_idx = min(entry_idx + cfg.horizon_bars - 1, len(d) - 1)
        exit_raw = float(d.iloc[exit_idx].Close)
        reason = "horizon"
        actual_exit_idx = exit_idx

        for j in range(entry_idx, exit_idx + 1):
            bar = d.iloc[j]
            hi, lo = float(bar.High), float(bar.Low)
            # Conservative convention when both levels are touched in one candle.
            if lo <= stop:
                exit_raw, reason, actual_exit_idx = stop, "stop", j
                break
            if hi >= target:
                exit_raw, reason, actual_exit_idx = target, "target", j
                break

        exit_px = exit_raw * (1.0 - slip / 10000.0)
        gross = qty * (exit_px - entry)
        fees = (qty * entry + qty * exit_px) * fee / 10000.0
        net = gross - fees
        cash += net
        peak = max(peak, cash)
        trades_today += 1

        trades.append({
            "signal_time": row.Date,
            "entry_time": entry_row.Date,
            "exit_time": d.iloc[actual_exit_idx].Date,
            "prob_up": p,
            "entry": entry,
            "stop": stop,
            "target": target,
            "exit": exit_px,
            "qty": qty,
            "gross_pnl": gross,
            "fees": fees,
            "net_pnl": net,
            "exit_reason": reason,
            "capital_after": cash,
        })
        equity.append((d.iloc[actual_exit_idx].Date, cash, cash / max(peak, 1.0) - 1.0))

        # Never allow overlapping positions in this research simulation.
        i = actual_exit_idx + 1

    eq = pd.DataFrame(equity, columns=["Date", "capital", "drawdown"])
    if eq.empty:
        eq = pd.DataFrame([[d.Date.iloc[0], cfg.starting_capital, 0.0]], columns=["Date", "capital", "drawdown"])
    eq = eq.drop_duplicates("Date").sort_values("Date").reset_index(drop=True)
    tr = pd.DataFrame(trades)
    return eq, tr


def summarize(eq: pd.DataFrame, tr: pd.DataFrame, cfg: Config) -> dict:
    start = cfg.starting_capital
    final = float(eq.capital.iloc[-1])
    years = max((eq.Date.iloc[-1] - eq.Date.iloc[0]).total_seconds() / (365.25 * 86400), 1e-9)
    cagr = (final / start) ** (1 / years) - 1 if final > 0 else -1
    daily = eq.set_index("Date").capital.resample("1D").last().ffill().pct_change().dropna()
    vol = daily.std() * math.sqrt(365) if len(daily) > 1 else np.nan
    sharpe = daily.mean() / daily.std() * math.sqrt(365) if daily.std() > 0 else np.nan
    neg = daily[daily < 0]
    sortino = daily.mean() / neg.std() * math.sqrt(365) if len(neg) > 1 and neg.std() > 0 else np.nan
    wins = int((tr.net_pnl > 0).sum()) if len(tr) else 0
    gross_profit = float(tr.loc[tr.net_pnl > 0, "net_pnl"].sum()) if wins else 0.0
    gross_loss = float(-tr.loc[tr.net_pnl <= 0, "net_pnl"].sum()) if len(tr) - wins else 0.0
    return {
        "final_capital": final,
        "return_pct": (final / start - 1) * 100,
        "cagr_pct": cagr * 100,
        "max_drawdown_pct": float(eq.drawdown.min() * 100),
        "annualized_volatility_pct": float(vol * 100) if pd.notna(vol) else None,
        "sharpe": float(sharpe) if pd.notna(sharpe) else None,
        "sortino": float(sortino) if pd.notna(sortino) else None,
        "trades": int(len(tr)),
        "trades_per_year": float(len(tr) / years),
        "wins": wins,
        "losses": int(len(tr) - wins),
        "win_rate_pct": wins / len(tr) * 100 if len(tr) else None,
        "profit_factor": gross_profit / gross_loss if gross_loss else None,
        "expectancy_per_trade": float(tr.net_pnl.mean()) if len(tr) else None,
    }


def buy_and_hold(d: pd.DataFrame, cfg: Config) -> dict:
    start_price = float(d.Close.iloc[0])
    end_price = float(d.Close.iloc[-1])
    ret = end_price / start_price - 1
    curve = d.Close / start_price
    dd = curve / curve.cummax() - 1
    years = max((d.Date.iloc[-1] - d.Date.iloc[0]).total_seconds() / (365.25 * 86400), 1e-9)
    return {
        "return_pct": ret * 100,
        "cagr_pct": ((1 + ret) ** (1 / years) - 1) * 100,
        "max_drawdown_pct": float(dd.min() * 100),
    }


def regime_report(d: pd.DataFrame, cfg: Config, threshold: float) -> list[dict]:
    d = d.copy()
    d["year"] = d.Date.dt.year
    years = sorted(d.year.unique())
    out = []
    for y in years:
        part = d[d.year == y].copy()
        if len(part) < 1000:
            continue
        eq, tr = leak_free_backtest(part, cfg, probability=threshold)
        s = summarize(eq, tr, cfg)
        s["year"] = int(y)
        out.append(s)
    return out


def main() -> None:
    cfg = Config()
    pred, source = load_oos_predictions(cfg)
    pred["Date"] = pd.to_datetime(pred["Date"], utc=True)
    pred = pred.sort_values("Date").reset_index(drop=True)

    auc = float(roc_auc_score(pred.label, pred.prob_up)) if pred.label.nunique() == 2 else None
    brier = float(brier_score_loss(pred.label, pred.prob_up)) if pred.label.nunique() == 2 else None

    thresholds = [0.56, 0.58, 0.60, 0.62, 0.65]
    sweep = []
    for threshold in thresholds:
        eq, tr = leak_free_backtest(pred, cfg, probability=threshold)
        row = summarize(eq, tr, cfg)
        row["probability_threshold"] = threshold
        sweep.append(row)

    sensitivity = []
    for fee, slip in [(5, 2), (10, 5), (15, 10)]:
        eq, tr = leak_free_backtest(pred, cfg, probability=0.60, fee_bps=fee, slippage_bps=slip)
        row = summarize(eq, tr, cfg)
        row.update({"probability_threshold": 0.60, "fee_bps": fee, "slippage_bps": slip})
        sensitivity.append(row)

    trade_frequency = []
    for limit in [1, 2]:
        eq, tr = leak_free_backtest(pred, cfg, probability=0.60, max_trades_per_day=limit)
        row = summarize(eq, tr, cfg)
        row.update({"probability_threshold": 0.60, "max_trades_per_day": limit})
        trade_frequency.append(row)

    baseline_eq, baseline_tr = leak_free_backtest(pred, cfg, probability=cfg.probability_long)
    baseline = summarize(baseline_eq, baseline_tr, cfg)
    bh = buy_and_hold(pred, cfg)
    regimes = regime_report(pred, cfg, 0.60)

    report = {
        "method": "leak_free_walk_forward_robustness",
        "prediction_source": source,
        "warning": "The original backtest used realized future_return as an entry filter. These results intentionally do not. Use this report for strategy decisions.",
        "data_start": str(pred.Date.iloc[0]),
        "data_end": str(pred.Date.iloc[-1]),
        "oos_rows": int(len(pred)),
        "calibration": {"roc_auc": auc, "brier_score": brier},
        "baseline_threshold_0_56": baseline,
        "buy_and_hold": bh,
        "threshold_sweep": sweep,
        "fee_slippage_sensitivity_at_0_60": sensitivity,
        "trade_frequency_at_0_60": trade_frequency,
        "yearly_regimes_at_0_60": regimes,
        "config": asdict(cfg),
    }

    (OUT / "robustness_report.json").write_text(json.dumps(report, indent=2, default=str))
    pd.DataFrame(sweep).to_csv(OUT / "robustness_threshold_sweep.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(OUT / "robustness_cost_sensitivity.csv", index=False)
    pd.DataFrame(trade_frequency).to_csv(OUT / "robustness_frequency.csv", index=False)
    pd.DataFrame(regimes).to_csv(OUT / "robustness_yearly.csv", index=False)

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
