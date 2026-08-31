from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("btc-trader")


@dataclass
class Config:
    symbol: str = "BTC-USD"
    interval: str = "1d"
    initial_train_days: int = 730
    retrain_every_days: int = 30
    target_horizon_days: int = 1
    model: str = "gradient_boosting"
    starting_capital: float = 100_000.0
    currency: str = "USD"
    risk_per_trade: float = 0.01
    max_position_pct: float = 0.50
    stop_loss_pct: float = 0.025
    take_profit_pct: float = 0.035
    entry_fee_bps: float = 5.0
    exit_fee_bps: float = 5.0
    slippage_bps: float = 2.0
    probability_threshold: float = 0.55
    long_only: bool = True
    max_daily_loss_pct: float = 0.05
    max_drawdown_pct: float = 0.20
    max_trades_per_day: int = 1
    cooldown_after_losses: int = 3
    cooldown_days: int = 3
    allow_same_day_reentry: bool = False
    seed: int = 42


FEATURES = [
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "ret_50", "ret_100",
    "vol_10", "vol_20", "vol_50", "vol_100",
    "sma20_ratio", "sma50_ratio", "sma100_ratio", "ema20_ratio", "ema50_ratio",
    "rsi14", "macd", "macd_signal", "macd_hist",
    "bb_pos", "bb_width", "atr14_pct", "hl_range", "oc_return",
    "volume_change_1", "volume_ratio_20", "drawdown_50", "drawdown_200",
    "trend_20_100", "vol_regime"
]


def download_data(cfg: Config) -> pd.DataFrame:
    log.info("Downloading %s %s historical data...", cfg.symbol, cfg.interval)
    df = yf.download(cfg.symbol, period="max", interval=cfg.interval, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError("Market data download returned no rows.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_convert(None)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop_duplicates("Date").sort_values("Date").reset_index(drop=True)
    validate_data(df)
    df.to_csv(OUTPUTS / "bitcoin_raw.csv", index=False)
    log.info("Loaded %d candles: %s to %s", len(df), df.Date.iloc[0].date(), df.Date.iloc[-1].date())
    return df


def validate_data(df: pd.DataFrame) -> None:
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df["Date"].duplicated().any():
        raise ValueError("Duplicate timestamps found.")
    if df[["Open", "High", "Low", "Close", "Volume"]].isna().any().any():
        raise ValueError("NaN values found in market data.")
    if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("Non-positive OHLC values found.")
    if (df["Volume"] < 0).any():
        raise ValueError("Negative volume found.")
    if (df["High"] < df[["Open", "Close"]].max(axis=1)).any():
        raise ValueError("Invalid High values found.")
    if (df["Low"] > df[["Open", "Close"]].min(axis=1)).any():
        raise ValueError("Invalid Low values found.")
    gaps = df["Date"].diff().dropna()
    if not gaps.empty:
        suspicious = (gaps > pd.Timedelta(days=3)).sum()
        if suspicious:
            log.warning("Found %d gaps longer than 3 days; inspect data source before trading.", suspicious)


def build_features(raw: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    d = raw.copy()
    close, high, low, volume, open_ = d.Close, d.High, d.Low, d.Volume, d.Open
    for n in [1, 3, 5, 10, 20, 50, 100]:
        d[f"ret_{n}"] = close.pct_change(n)
    ret1 = close.pct_change()
    for n in [10, 20, 50, 100]:
        d[f"vol_{n}"] = ret1.rolling(n).std()
    for n in [20, 50, 100]:
        d[f"sma{n}_ratio"] = close / close.rolling(n).mean() - 1
    for n in [20, 50]:
        d[f"ema{n}_ratio"] = close / close.ewm(span=n, adjust=False).mean() - 1
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macd_signal"] = d.macd.ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d.macd - d.macd_signal
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi14"] = 100 - 100 / (1 + rs)
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    upper, lower = mid + 2 * std, mid - 2 * std
    d["bb_pos"] = (close - lower) / (upper - lower).replace(0, np.nan)
    d["bb_width"] = (upper - lower) / mid.replace(0, np.nan)
    tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    d["atr14_pct"] = tr.rolling(14).mean() / close
    d["hl_range"] = (high - low) / close
    d["oc_return"] = close / open_ - 1
    d["volume_change_1"] = volume.pct_change()
    d["volume_ratio_20"] = volume / volume.rolling(20).mean()
    d["drawdown_50"] = close / close.rolling(50).max() - 1
    d["drawdown_200"] = close / close.rolling(200).max() - 1
    d["trend_20_100"] = close.rolling(20).mean() / close.rolling(100).mean() - 1
    d["vol_regime"] = d["vol_20"] / d["vol_100"]
    d["future_return"] = close.shift(-horizon) / close - 1
    d["label"] = (d["future_return"] > 0).astype(int)
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=FEATURES + ["future_return", "label"]).reset_index(drop=True)
    return d


def make_model(name: str, seed: int):
    if name == "random_forest":
        estimator = RandomForestClassifier(n_estimators=300, min_samples_leaf=8, max_features="sqrt", random_state=seed, n_jobs=-1, class_weight="balanced")
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])
    if name == "logistic":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000, C=0.5, random_state=seed))])
    estimator = GradientBoostingClassifier(n_estimators=200, learning_rate=0.03, max_depth=2, min_samples_leaf=10, random_state=seed)
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])


def walk_forward_predict(d: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = d.copy()
    out["prob_up"] = np.nan
    out["prediction"] = np.nan
    out["train_start"] = pd.NaT
    out["train_end"] = pd.NaT
    out["test_start"] = pd.NaT
    out["test_end"] = pd.NaT
    start = cfg.initial_train_days
    while start < len(out):
        end = min(start + cfg.retrain_every_days, len(out))
        train = out.iloc[:start]
        test = out.iloc[start:end]
        if train.label.nunique() < 2:
            start = end
            continue
        model = make_model(cfg.model, cfg.seed)
        model.fit(train[FEATURES], train.label)
        p = model.predict_proba(test[FEATURES])[:, 1]
        out.loc[test.index, "prob_up"] = p
        out.loc[test.index, "prediction"] = (p >= 0.5).astype(int)
        out.loc[test.index, "train_start"] = train.Date.iloc[0]
        out.loc[test.index, "train_end"] = train.Date.iloc[-1]
        out.loc[test.index, "test_start"] = test.Date.iloc[0]
        out.loc[test.index, "test_end"] = test.Date.iloc[-1]
        start = end
    return out.dropna(subset=["prob_up"]).reset_index(drop=True)


def apply_slippage(price: float, side: int, bps: float) -> float:
    return price * (1 + side * bps / 10_000)


def backtest(d: pd.DataFrame, cfg: Config):
    capital = cfg.starting_capital
    peak = capital
    equity_rows, trades = [], []
    cooldown = 0
    consecutive_losses = 0
    total_fees = total_slippage = 0.0
    daily_start = None
    day_trades = 0
    halted_today = False

    for i in range(len(d) - 1):
        row, nxt = d.iloc[i], d.iloc[i + 1]
        day = row.Date.date()
        if daily_start is None or day != d.iloc[i - 1].Date.date():
            daily_start = capital
            day_trades = 0
            halted_today = False
            if cooldown > 0:
                cooldown -= 1
        peak = max(peak, capital)
        drawdown = capital / peak - 1
        if drawdown <= -cfg.max_drawdown_pct:
            halted_today = True
        if (daily_start - capital) / daily_start >= cfg.max_daily_loss_pct:
            halted_today = True
        if cooldown > 0 or halted_today or day_trades >= cfg.max_trades_per_day:
            equity_rows.append({"Date": row.Date, "capital": capital, "drawdown": drawdown, "halted": True})
            continue

        p = float(row.prob_up)
        if p >= cfg.probability_threshold:
            direction = 1
        elif (not cfg.long_only) and p <= 1 - cfg.probability_threshold:
            direction = -1
        else:
            direction = 0
        if direction == 0:
            equity_rows.append({"Date": row.Date, "capital": capital, "drawdown": drawdown, "halted": False})
            continue

        raw_entry = float(nxt.Open)
        entry = apply_slippage(raw_entry, direction, cfg.slippage_bps)
        stop = entry * (1 - cfg.stop_loss_pct) if direction == 1 else entry * (1 + cfg.stop_loss_pct)
        target = entry * (1 + cfg.take_profit_pct) if direction == 1 else entry * (1 - cfg.take_profit_pct)
        risk_cash = capital * cfg.risk_per_trade
        stop_distance = abs(entry - stop)
        qty_by_risk = risk_cash / stop_distance if stop_distance else 0
        qty_by_cap = (capital * cfg.max_position_pct) / entry if entry else 0
        qty = min(qty_by_risk, qty_by_cap)
        if qty <= 0:
            continue

        hi, lo = float(nxt.High), float(nxt.Low)
        exit_price = float(nxt.Close)
        exit_reason = "close"
        if direction == 1:
            stop_hit, target_hit = lo <= stop, hi >= target
            if stop_hit and target_hit:
                exit_price, exit_reason = stop, "stop_and_target_same_candle_conservative_stop"
            elif stop_hit:
                exit_price, exit_reason = stop, "stop"
            elif target_hit:
                exit_price, exit_reason = target, "target"
        else:
            stop_hit, target_hit = hi >= stop, lo <= target
            if stop_hit and target_hit:
                exit_price, exit_reason = stop, "stop_and_target_same_candle_conservative_stop"
            elif stop_hit:
                exit_price, exit_reason = stop, "stop"
            elif target_hit:
                exit_price, exit_reason = target, "target"
        exit_exec = apply_slippage(exit_price, -direction, cfg.slippage_bps)
        gross = qty * (exit_exec - entry) * direction
        entry_notional, exit_notional = qty * entry, qty * exit_exec
        entry_fee = entry_notional * cfg.entry_fee_bps / 10_000
        exit_fee = exit_notional * cfg.exit_fee_bps / 10_000
        fees = entry_fee + exit_fee
        slippage_cost = qty * abs(raw_entry - entry) + qty * abs(exit_price - exit_exec)
        net = gross - fees
        capital += net
        total_fees += fees
        total_slippage += slippage_cost
        day_trades += 1
        if net > 0:
            consecutive_losses = 0
        else:
            consecutive_losses += 1
            if consecutive_losses >= cfg.cooldown_after_losses:
                cooldown = cfg.cooldown_days
                consecutive_losses = 0
        trades.append({
            "signal_date": row.Date, "entry_date": nxt.Date, "exit_date": nxt.Date,
            "direction": "LONG" if direction == 1 else "SHORT", "prob_up": p,
            "entry": entry, "stop": stop, "target": target, "exit": exit_exec,
            "qty": qty, "gross_pnl": gross, "fees": fees, "slippage_cost": slippage_cost,
            "net_pnl": net, "exit_reason": exit_reason, "capital_after": capital
        })
        peak = max(peak, capital)
        equity_rows.append({"Date": nxt.Date, "capital": capital, "drawdown": capital / peak - 1, "halted": False})

    equity = pd.DataFrame(equity_rows).drop_duplicates("Date").sort_values("Date").reset_index(drop=True)
    if equity.empty:
        raise RuntimeError("Backtest produced no equity observations.")
    trades_df = pd.DataFrame(trades)
    summary = performance_summary(equity, trades_df, cfg.starting_capital, total_fees, total_slippage)
    return equity, trades_df, summary


def performance_summary(equity, trades, starting, fees, slippage):
    final = float(equity.capital.iloc[-1])
    total = final / starting - 1
    days = max((equity.Date.iloc[-1] - equity.Date.iloc[0]).days, 1)
    years = days / 365.25
    cagr = (final / starting) ** (1 / years) - 1 if final > 0 else -1
    daily = equity.capital.pct_change().dropna()
    vol = daily.std() * math.sqrt(365.25) if len(daily) > 1 else 0
    sharpe = daily.mean() / daily.std() * math.sqrt(365.25) if daily.std() > 0 else np.nan
    downside = daily[daily < 0].std()
    sortino = daily.mean() / downside * math.sqrt(365.25) if downside and downside > 0 else np.nan
    max_dd = float(equity.drawdown.min())
    wins = int((trades.net_pnl > 0).sum()) if not trades.empty else 0
    losses = int((trades.net_pnl <= 0).sum()) if not trades.empty else 0
    gross_profit = float(trades.loc[trades.net_pnl > 0, "net_pnl"].sum()) if not trades.empty else 0
    gross_loss = float(-trades.loc[trades.net_pnl <= 0, "net_pnl"].sum()) if not trades.empty else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan
    return {
        "starting_capital": starting, "final_capital": round(final, 2),
        "total_return_pct": round(total * 100, 2), "cagr_pct": round(cagr * 100, 2),
        "annualized_volatility_pct": round(vol * 100, 2),
        "sharpe": round(float(sharpe), 3) if pd.notna(sharpe) else None,
        "sortino": round(float(sortino), 3) if pd.notna(sortino) else None,
        "max_drawdown_pct": round(max_dd * 100, 2), "calmar": round(float(calmar), 3) if pd.notna(calmar) else None,
        "trades": len(trades), "wins": wins, "losses": losses,
        "win_rate_pct": round(wins / len(trades) * 100, 2) if len(trades) else None,
        "average_win": round(float(trades.loc[trades.net_pnl > 0, "net_pnl"].mean()), 2) if wins else None,
        "average_loss": round(float(trades.loc[trades.net_pnl <= 0, "net_pnl"].mean()), 2) if losses else None,
        "profit_factor": round(pf, 3) if np.isfinite(pf) else None,
        "expectancy": round(float(trades.net_pnl.mean()), 2) if len(trades) else None,
        "largest_win": round(float(trades.net_pnl.max()), 2) if len(trades) else None,
        "largest_loss": round(float(trades.net_pnl.min()), 2) if len(trades) else None,
        "fees_paid": round(fees, 2), "slippage_cost": round(slippage, 2),
    }


def buy_and_hold(raw, starting):
    first, last = float(raw.Close.iloc[0]), float(raw.Close.iloc[-1])
    final = starting * last / first
    return {"starting_capital": starting, "final_capital": final, "total_return_pct": (final / starting - 1) * 100}


def save_outputs(raw, features, predictions, equity, trades, summary, benchmark):
    features.to_csv(OUTPUTS / "features.csv", index=False)
    predictions.to_csv(OUTPUTS / "predictions.csv", index=False)
    equity.to_csv(OUTPUTS / "equity_curve.csv", index=False)
    trades.to_csv(OUTPUTS / "trades.csv", index=False)
    report = {"strategy": summary, "buy_and_hold": benchmark}
    (OUTPUTS / "performance_summary.json").write_text(json.dumps(report, indent=2, default=str))
    yearly = equity.set_index("Date").capital.resample("YE").last().pct_change().dropna().rename("return").reset_index()
    yearly.to_csv(OUTPUTS / "yearly_performance.csv", index=False)
    monthly = equity.set_index("Date").capital.resample("ME").last().pct_change().dropna().rename("return").reset_index()
    monthly.to_csv(OUTPUTS / "monthly_performance.csv", index=False)

    plt.figure(figsize=(12, 6)); plt.plot(raw.Date, raw.Close); plt.title("Bitcoin Price"); plt.xlabel("Date"); plt.ylabel("BTC-USD"); plt.tight_layout(); plt.savefig(OUTPUTS / "btc_price.png", dpi=150); plt.close()
    plt.figure(figsize=(12, 6)); plt.plot(equity.Date, equity.capital, label="Strategy"); bh = np.full(len(equity), summary["starting_capital"]); bh = bh * raw.Close.iloc[0:len(equity)].to_numpy() / raw.Close.iloc[0]; plt.plot(equity.Date, bh, label="Buy & Hold"); plt.legend(); plt.title("Strategy vs Buy & Hold"); plt.tight_layout(); plt.savefig(OUTPUTS / "strategy_vs_buy_hold.png", dpi=150); plt.close()
    plt.figure(figsize=(12, 4)); plt.plot(equity.Date, equity.drawdown); plt.title("Strategy Drawdown"); plt.tight_layout(); plt.savefig(OUTPUTS / "drawdown.png", dpi=150); plt.close()
    if not trades.empty:
        plt.figure(figsize=(10, 5)); plt.hist(trades.net_pnl, bins=40); plt.title("Trade Net P&L Distribution"); plt.tight_layout(); plt.savefig(OUTPUTS / "trade_pnl_distribution.png", dpi=150); plt.close()


def run_backtest(cfg: Config):
    raw = download_data(cfg)
    features = build_features(raw, cfg.target_horizon_days)
    features.to_csv(OUTPUTS / "features.csv", index=False)
    predictions = walk_forward_predict(features, cfg)
    predictions.to_csv(OUTPUTS / "predictions.csv", index=False)
    equity, trades, summary = backtest(predictions, cfg)
    benchmark = buy_and_hold(raw, cfg.starting_capital)
    save_outputs(raw, features, predictions, equity, trades, summary, benchmark)
    (OUTPUTS / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    log.info("Backtest complete: strategy %.2f%% vs buy-and-hold %.2f%%", summary["total_return_pct"], benchmark["total_return_pct"])
    print(json.dumps({"strategy": summary, "buy_and_hold": benchmark}, indent=2, default=str))


def paper_mode(cfg: Config):
    raise NotImplementedError("Paper trading is intentionally scaffolded but disabled until a live data/order adapter is reviewed and configured.")


def main():
    parser = argparse.ArgumentParser(description="Bitcoin ML trading research system")
    parser.add_argument("--mode", choices=["backtest", "data", "paper"], default="backtest")
    parser.add_argument("--model", choices=["gradient_boosting", "random_forest", "logistic"], default="gradient_boosting")
    args = parser.parse_args()
    cfg = Config(model=args.model)
    if args.mode == "data":
        download_data(cfg)
    elif args.mode == "paper":
        paper_mode(cfg)
    else:
        run_backtest(cfg)


if __name__ == "__main__":
    main()
